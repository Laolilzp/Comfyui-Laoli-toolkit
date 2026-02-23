import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

console.log("%c[老李-工具箱] Sigma编辑器 已加载", "color:white; background:#d9230f; padding:4px");

app.registerExtension({
    name: "Laoli.SigmaEditor.FinalFix",
    async setup() {
        api.addEventListener("laoli_sigma_update_event", ({ detail }) => {
            if (!detail || !detail.node_id || !detail.text) return;
            const node = app.graph.getNodeById(detail.node_id);
            if (!node || node.type !== "LaoliSigmaEditor") return;

            const widget = node.widgets.find(w => w.name === "sigma_string");
            if (widget && widget.value !== detail.text) widget.value = detail.text;
            if (node.refreshCurve) node.refreshCurve(detail.text);
        });
    },

    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (nodeData.name === "LaoliSigmaEditor") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                const node = this;

                node.setSize([450, 500]);

                // --- 1. 核心状态初始化 (仿照 Visual Sampler) ---
                node.laoli_sigmas = [];
                node.dragIndex = -1;       // 当前拖拽点的索引
                node.hoverIndex = -1;      // 当前悬停点的索引
                node.isDragging = false;   // 是否发生了有效拖拽位移
                node.dragStartPos = null;  // 鼠标按下的起始坐标
                node.chartArea = { y: 0, height: 0 };
                node.maxVal = 1.0;

                // --- 2. Widget 设置 ---
                const sigmaWidget = node.widgets.find(w => w.name === "sigma_string");
                if (sigmaWidget) {
                    const TEXTAREA_HEIGHT = 55;
                    sigmaWidget.computeSize = function() { return [0, TEXTAREA_HEIGHT]; };
                    const applyTextareaStyle = () => {
                        if (sigmaWidget.inputEl) {
                            Object.assign(sigmaWidget.inputEl.style, {
                                height: "50px", maxHeight: "50px", minHeight: "50px",
                                overflowY: "auto", resize: "none"
                            });
                        }
                    };
                    setTimeout(applyTextareaStyle, 100);
                    const origOnConfigure = node.onConfigure;
                    node.onConfigure = function(info) {
                        if (origOnConfigure) origOnConfigure.apply(this, arguments);
                        setTimeout(applyTextareaStyle, 100);
                        setTimeout(() => {
                            const sw = node.widgets.find(w => w.name === "sigma_string");
                            if (sw && sw.value) node.refreshCurve(sw.value);
                        }, 300);
                    };
                    sigmaWidget.callback = (v) => node.refreshCurve(v);
                }

                // --- 自动生成逻辑 ---
                let generateTimer = null;
                node.requestGenerate = function() {
                    if (generateTimer) clearTimeout(generateTimer);
                    generateTimer = setTimeout(() => {
                        const stepsW = node.widgets.find(w => w.name === "steps");
                        const denoiseW = node.widgets.find(w => w.name === "denoise");
                        const schedulerW = node.widgets.find(w => w.name === "scheduler");
                        if (!stepsW) return;

                        const payload = {
                            steps: stepsW.value,
                            denoise: denoiseW ? denoiseW.value : 1.0,
                            scheduler: schedulerW ? schedulerW.value : "normal",
                            node_id: String(node.id)
                        };

                        api.fetchApi("/laoli/sigmas/generate", {
                            method: "POST",
                            body: JSON.stringify(payload)
                        }).then(r => r.json()).then(data => {
                            if (data.status === "success" && data.sigmas) {
                                if (sigmaWidget && sigmaWidget.value !== data.sigmas) {
                                    sigmaWidget.value = data.sigmas;
                                    node.refreshCurve(data.sigmas);
                                }
                            }
                        }).catch(e => { console.warn("[老李-工具箱] 生成sigma失败:", e); });
                    }, 300);
                };

                setTimeout(() => {
                    const watchWidgets = ["steps", "scheduler", "denoise"];
                    for (const wName of watchWidgets) {
                        const w = node.widgets.find(ww => ww.name === wName);
                        if (!w) continue;
                        const origCallback = w.callback;
                        w.callback = function(value) {
                            if (origCallback) origCallback.apply(this, arguments);
                            if (sigmaWidget) sigmaWidget.value = "";
                            node.requestGenerate();
                        };
                    }
                }, 100);

                // --- 预设按钮 ---
                const presetWidget = node.addWidget("combo", "预设列表", "选择预设...", (name) => {
                    if (name === "选择预设...") return;
                    api.fetchApi("/laoli/sigmas/presets").then(r=>r.json()).then(presets => {
                        if (presets[name]) {
                            const val = presets[name];
                            sigmaWidget.value = val;
                            node.refreshCurve(val);
                            node.setDirtyCanvas(true, true);
                        }
                    });
                }, { values: ["选择预设..."] });

                const refreshPresets = async () => {
                    try {
                        const r = await api.fetchApi("/laoli/sigmas/presets");
                        const d = await r.json();
                        if (d) presetWidget.options.values = ["选择预设...", ...Object.keys(d)];
                    } catch(e){}
                };
                refreshPresets();

                node.addWidget("button", "💾 保存预设", null, () => {
                    if (!sigmaWidget || !sigmaWidget.value) { alert("当前没有sigma数据可保存"); return; }
                    const n = prompt("输入预设名称:");
                    if (n) {
                        api.fetchApi("/laoli/sigmas/save", {
                            method: "POST", headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ name: n, values: sigmaWidget.value })
                        }).then(r => r.json()).then(data => {
                            if (data.status === "success") refreshPresets();
                            else alert("保存失败");
                        });
                    }
                });
                node.addWidget("button", "🗑️ 删除选中", null, () => {
                    const n = presetWidget.value;
                    if (n && n !== "选择预设..." && confirm(`删除 "${n}"?`)) {
                        api.fetchApi("/laoli/sigmas/delete", {
                            method: "POST", headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ name: n })
                        }).then(() => { presetWidget.value = "选择预设..."; refreshPresets(); });
                    }
                });
                node.addWidget("button", "🔄 清空数据", null, () => {
                    sigmaWidget.value = ""; node.laoli_sigmas = []; node.setDirtyCanvas(true, true);
                });

                node.refreshCurve = function(str) {
                    if (!str || typeof str !== 'string') {
                        node.laoli_sigmas = []; node.setDirtyCanvas(true, true); return;
                    }
                    const matches = str.match(/-?\d*\.?\d+(?:[eE][-+]?\d+)?/g);
                    node.laoli_sigmas = matches ? matches.map(v => Math.max(0, Math.min(1, Number(v)))) : [];
                    node.setDirtyCanvas(true, true);
                };
                setTimeout(() => {
                    if(sigmaWidget && sigmaWidget.value) node.refreshCurve(sigmaWidget.value);
                }, 200);

                const origOnExecuted = node.onExecuted;
                node.onExecuted = function(message) {
                    if (origOnExecuted) origOnExecuted.apply(this, arguments);
                    if (message && message.sigma_string && message.sigma_string[0]) {
                        const newVal = message.sigma_string[0];
                        if (sigmaWidget) sigmaWidget.value = newVal;
                        node.refreshCurve(newVal);
                    }
                };

// --- 3. 绘图组件 (Visualizer) ---
                const MIN_CHART_HEIGHT = 100;
                node.addCustomWidget({
                    name: "Visualizer",
                    type: "VISUALIZER",
                    
                    // ★★★ 核心修复 1：将控件逻辑高度设为 0 ★★★
                    // 这样 LiteGraph 就不会认为鼠标点到了控件上，
                    // 所有点击都会直接穿透给 node.onMouseDown，完美解决坐标和拦截问题。
                    computeSize: function(width) { 
                        return [width, 0]; 
                    },

                    // ★★★ 核心修复 2：移除 mouse 函数 ★★★
                    // 既然高度为0，且我们要让节点处理事件，就不需要这个函数了。
                    // mouse: function(...) { ... }, 

                    draw: function(ctx, node, widgetWidth, _y, _height) {
                        const bottomMargin = 10;
                        // 虽然 widget 只有 0 高度，但我们计算画布剩余空间来绘图
                        const realHeight = Math.max(MIN_CHART_HEIGHT, node.size[1] - _y - bottomMargin);
                        
                        node.chartArea = { y: _y, height: realHeight };
                        
                        // 之前调整过的左边距 (35)
                        const margin = { top: 10, right: 10, bottom: 20, left: 35 };
                        const chartX = margin.left, chartY = _y + margin.top;
                        const chartW = widgetWidth - margin.left - margin.right;
                        const chartH = realHeight - margin.top - margin.bottom;
                        node.chartLayout = { x: chartX, y: chartY, w: chartW, h: chartH };

                        // 背景
                        ctx.save(); ctx.beginPath(); ctx.rect(0, _y, widgetWidth, realHeight); ctx.clip();
                        ctx.fillStyle = "#121212"; ctx.fillRect(chartX, chartY, chartW, chartH);
                        ctx.strokeStyle = "#333"; ctx.strokeRect(chartX, chartY, chartW, chartH);

                        const sigmas = node.laoli_sigmas;
                        if (!sigmas || sigmas.length === 0) {
                            ctx.fillStyle = "#666"; ctx.font = "14px Arial";
                            ctx.fillText("无数据 - 调整步数/调度器后自动生成", chartX + 20, chartY + chartH / 2);
                            ctx.restore(); return;
                        }

                        // 绘制网格
                        ctx.font = "10px Arial"; ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.lineWidth = 1;
                        for (let v = 0; v <= 1.001; v += 0.1) {
                            const lineY = chartY + chartH - (v * chartH);
                            const isMajor = Math.abs(v % 0.5) < 0.01 || v < 0.01;
                            ctx.strokeStyle = isMajor ? "#555" : "#222";
                            ctx.beginPath(); ctx.moveTo(chartX, lineY); ctx.lineTo(chartX + chartW, lineY); ctx.stroke();
                            ctx.fillStyle = "#aaa"; ctx.fillText(v.toFixed(1), chartX - 5, lineY);
                        }
                        ctx.textAlign = "center"; ctx.textBaseline = "top";
                        const totalSteps = sigmas.length;
                        let stepInterval = totalSteps > 50 ? (totalSteps > 100 ? 20 : 10) : 5;
                        for (let i = 0; i < totalSteps; i += stepInterval) {
                            const lineX = chartX + (i / (totalSteps - 1)) * chartW;
                            ctx.strokeStyle = "#222";
                            ctx.beginPath(); ctx.moveTo(lineX, chartY); ctx.lineTo(lineX, chartY + chartH); ctx.stroke();
                            ctx.fillStyle = "#666"; ctx.fillText(i.toString(), lineX, chartY + chartH + 2);
                        }

                        // 生成点坐标
                        const points = [];
                        const denominator = sigmas.length > 1 ? sigmas.length - 1 : 1;
                        for (let i = 0; i < sigmas.length; i++) {
                            const px = chartX + (i / denominator) * chartW;
                            const py = chartY + chartH - (sigmas[i] / node.maxVal) * chartH;
                            points.push({ x: px, y: py, val: sigmas[i], idx: i });
                        }
                        node.points = points;

                        // 绘制连线
                        ctx.save();
                        ctx.strokeStyle = "rgba(0, 204, 255, 0.15)"; ctx.lineWidth = 6; ctx.lineJoin = "round";
                        ctx.beginPath(); points.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
                        ctx.stroke(); ctx.restore();
                        ctx.strokeStyle = "#00ccff"; ctx.lineWidth = 1.2; ctx.lineJoin = "round";
                        ctx.beginPath(); points.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
                        ctx.stroke();

                        // 绘制点
                        const activeIdx = node.dragIndex !== -1 ? node.dragIndex : node.hoverIndex;
                        if (points.length < 120) {
                            const baseR = points.length < 30 ? 3.5 : (points.length < 60 ? 2.5 : 1.8);
                            points.forEach((p, i) => {
                                if (i === activeIdx) return;
                                ctx.fillStyle = "#00ccff"; ctx.beginPath(); ctx.arc(p.x, p.y, baseR, 0, Math.PI * 2); ctx.fill();
                            });
                        }

                        // 绘制激活点
                        if (activeIdx !== -1 && points[activeIdx]) {
                            const p = points[activeIdx];
                            const isDrag = node.dragIndex !== -1;
                            ctx.save();
                            ctx.strokeStyle = isDrag ? "rgba(255, 180, 50, 0.6)" : "rgba(255, 200, 80, 0.4)";
                            ctx.lineWidth = 0.8; ctx.setLineDash([3, 3]);
                            ctx.beginPath(); ctx.moveTo(chartX, p.y); ctx.lineTo(chartX + chartW, p.y);
                            ctx.moveTo(p.x, chartY); ctx.lineTo(p.x, chartY + chartH); ctx.stroke();
                            ctx.setLineDash([]); ctx.restore();

                            const labelStep = `Step: ${p.idx}`, labelVal = `Val: ${p.val.toFixed(4)}`;
                            ctx.font = "bold 11px Arial";
                            const textW = Math.max(ctx.measureText(labelStep).width, ctx.measureText(labelVal).width);
                            let lx = p.x + 12; if (lx + textW + 10 > chartX + chartW) lx = p.x - textW - 18;
                            let ly = p.y - 38; if (ly < chartY + 2) ly = p.y + 16;
                            ctx.fillStyle = "rgba(30, 30, 30, 0.85)"; ctx.fillRect(lx - 4, ly - 4, textW + 12, 30);
                            ctx.fillStyle = "#ffc850"; ctx.textAlign = "left";
                            ctx.fillText(labelStep, lx, ly + 8); ctx.fillText(labelVal, lx, ly + 21);

                            ctx.beginPath(); ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
                            ctx.fillStyle = isDrag ? "rgba(255, 180, 50, 0.25)" : "rgba(0, 204, 255, 0.2)"; ctx.fill();
                            ctx.beginPath(); ctx.arc(p.x, p.y, 4.5, 0, Math.PI * 2);
                            ctx.fillStyle = isDrag ? "#ffb432" : "#fff"; ctx.fill();
                            ctx.strokeStyle = isDrag ? "#ffb432" : "#00ccff"; ctx.lineWidth = 1.5; ctx.stroke();
                        }
                        ctx.restore();
                    }
                });                // --- 4. 辅助函数：结束拖拽 ---
                node._finishDrag = function() {
                    const wasDragging = node.isDragging;
                    const index = node.dragIndex;
                    
                    // 如果发生了有效拖拽，更新 Widget
                    if (index !== -1 && wasDragging) {
                        const w = node.widgets.find(w => w.name === "sigma_string");
                        if (w) {
                            const newStr = "[" + node.laoli_sigmas.map(v => v.toFixed(4)).join(", ") + "]";
                            if (w.value !== newStr) {
                                w.value = newStr;
                                if (w.callback) w.callback(newStr);
                            }
                        }
                    }
                    
                    // 重置状态
                    node.dragIndex = -1;
                    node.dragStartPos = null;
                    node.isDragging = false;
                    node.hoverIndex = -1;
                    node.setDirtyCanvas(true, true);
                    return wasDragging;
                };

                // 命中检测
                node.getPointFromEvent = function(localPos) {
                    if (!node.chartLayout || !node.points || !node.points.length) return null;
                    const mx = localPos[0], my = localPos[1];
                    const L = node.chartLayout;
                    // 扩大一点判定范围
                    if (mx < L.x - 15 || mx > L.x + L.w + 15 || my < L.y - 15 || my > L.y + L.h + 20) return null;
                    
                    if (node.points.length === 1) {
                        const p = node.points[0];
                        return Math.hypot(mx - p.x, my - p.y) < 30 ? 0 : -1;
                    }
                    let minD = Infinity, close = -1;
                    const xR = Math.max(10, (L.w / (node.points.length - 1)) * 0.6);
                    for (let i = 0; i < node.points.length; i++) {
                        const p = node.points[i];
                        const dx = Math.abs(mx - p.x);
                        if (dx < xR && dx < minD) { minD = dx; close = i; }
                    }
                    return close;
                };

                // --- 5. 交互事件处理 ---

                const origOnMouseDown = node.onMouseDown;
                node.onMouseDown = function(event, pos, graphCanvas) {
                    // 安全网：清理残留
                    if (node.dragIndex !== -1) node._finishDrag();

                    const idx = node.getPointFromEvent(pos);
                    if (idx !== null && idx >= 0) {
                        node.dragIndex = idx;
                        node.dragStartPos = { x: pos[0], y: pos[1] };
                        node.isDragging = false;
                        node.setDirtyCanvas(true, true);
                        return true; // 捕获事件
                    }
                    // 点击图表区域但没点中点，也捕获，防止拖动节点
                    if (node.chartLayout) {
                        const L = node.chartLayout;
                        if (pos[0] >= L.x && pos[0] <= L.x + L.w && pos[1] >= L.y && pos[1] <= L.y + L.h) return true;
                    }
                    return origOnMouseDown ? origOnMouseDown.apply(this, arguments) : undefined;
                };

                const origOnMouseMove = node.onMouseMove;
                node.onMouseMove = function(event, pos, graphCanvas) {
                    // ★★★ 核心修复：检测鼠标左键状态 ★★★
                    // 如果处于拖拽模式，但 buttons & 1 为 0 (左键未按)，说明鼠标已松开
                    if (node.dragIndex !== -1 && !(event.buttons & 1)) {
                        node._finishDrag();
                        return true;
                    }

                    if (node.dragIndex !== -1 && node.chartLayout) {
                        // 拖拽逻辑
                        if (!node.isDragging && node.dragStartPos) {
                            if (Math.abs(pos[0] - node.dragStartPos.x) > 2 || Math.abs(pos[1] - node.dragStartPos.y) > 2) {
                                node.isDragging = true;
                            }
                        }
                        if (node.isDragging) {
                            const L = node.chartLayout;
                            let clampedY = Math.max(L.y, Math.min(L.y + L.h, pos[1]));
                            let ratio = (L.y + L.h - clampedY) / L.h;
                            node.laoli_sigmas[node.dragIndex] = Math.max(0, Math.min(1, ratio));
                            node.setDirtyCanvas(true, true);
                        }
                        return true;
                    }

                    // Hover 逻辑
                    const idx = node.getPointFromEvent(pos);
                    const newHover = (idx !== null && idx >= 0) ? idx : -1;
                    if (node.hoverIndex !== newHover) {
                        node.hoverIndex = newHover;
                        node.setDirtyCanvas(true, false);
                    }
                    return origOnMouseMove ? origOnMouseMove.apply(this, arguments) : undefined;
                };

                const origOnMouseUp = node.onMouseUp;
                node.onMouseUp = function(event, pos, graphCanvas) {
                    if (node.dragIndex !== -1) {
                        node._finishDrag();
                        return true;
                    }
                    return origOnMouseUp ? origOnMouseUp.apply(this, arguments) : undefined;
                };

                const origOnMouseLeave = node.onMouseLeave;
                node.onMouseLeave = function(event) {
                    node.hoverIndex = -1;
                    node.setDirtyCanvas(true, false);
                    if (origOnMouseLeave) origOnMouseLeave.apply(this, arguments);
                };

                // --- 6. 全局安全监听 (Global Safety Net) ---
                const canvas = app.canvas?.canvas || document.querySelector("canvas");
                if (canvas) {
                    const globalUp = () => { if (node.dragIndex !== -1) node._finishDrag(); };
                    // 使用 pointerup 以获得更好的兼容性
                    document.addEventListener("pointerup", globalUp);
                    document.addEventListener("pointercancel", globalUp);
                    
                    const origOnRemoved = node.onRemoved;
                    node.onRemoved = function() {
                        document.removeEventListener("pointerup", globalUp);
                        document.removeEventListener("pointercancel", globalUp);
                        if (origOnRemoved) origOnRemoved.apply(this, arguments);
                    };
                }

                // --- 7. 尺寸控制 ---
                const origOnResize = node.onResize;
                node.onResize = function(size) {
                    let fixedHeight = LiteGraph.NODE_TITLE_HEIGHT || 30;
                    if (node.widgets) {
                        for (const w of node.widgets) {
                            if (w.type === "VISUALIZER") continue;
                            const wh = w.computeSize ? w.computeSize(size[0])[1] : (LiteGraph.NODE_WIDGET_HEIGHT || 20);
                            fixedHeight += wh + 4;
                        }
                    }
                    const minNodeHeight = fixedHeight + MIN_CHART_HEIGHT + 20;
                    if (size[1] < minNodeHeight) size[1] = minNodeHeight;
                    if (origOnResize) origOnResize.apply(this, arguments);
                    node.setDirtyCanvas(true, true);
                };

                return r;
            };
        }
    }
});