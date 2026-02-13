import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// 启动标志
console.log("%c[老李-工具箱] 已加载", "color:white; background:#d9230f; padding:4px");

app.registerExtension({
    name: "Laoli.SigmaEditor.FinalFix",
    async setup() {
        // 监听后端推送
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

                // 1. 初始大小
                node.setSize([450, 500]);
                node.laoli_sigmas = [];

                // 内部状态
                node.dragIndex = -1;
                node.hoverIndex = -1;
                node.isDragging = false;
                node.chartArea = { y: 0, height: 0 };

                // --- A. 文本框高度锁死 ---
                const sigmaWidget = node.widgets.find(w => w.name === "sigma_string");
                if (sigmaWidget) {
                    const TEXTAREA_HEIGHT = 55;
                    sigmaWidget.computeSize = function() {
                        return [0, TEXTAREA_HEIGHT];
                    };
                    const applyTextareaStyle = () => {
                        if (sigmaWidget.inputEl) {
                            Object.assign(sigmaWidget.inputEl.style, {
                                height: "50px",
                                maxHeight: "50px",
                                minHeight: "50px",
                                overflowY: "auto",
                                resize: "none"
                            });
                        }
                    };
                    setTimeout(applyTextareaStyle, 100);
                    const origOnConfigure = node.onConfigure;
                    node.onConfigure = function(info) {
                        if (origOnConfigure) origOnConfigure.apply(this, arguments);
                        setTimeout(applyTextareaStyle, 100);
                        // 恢复工作流时刷新曲线
                        setTimeout(() => {
                            const sw = node.widgets.find(w => w.name === "sigma_string");
                            if (sw && sw.value) node.refreshCurve(sw.value);
                        }, 300);
                    };

                    sigmaWidget.callback = (v) => node.refreshCurve(v);
                }

                // --- B. 请求后端生成 sigma 的函数 ---
                // 防抖定时器
                let generateTimer = null;

                node.requestGenerate = function() {
                    // 清除之前的定时器，防抖 300ms
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
                                // 后端推送事件会自动更新，这里作为 fallback
                                if (sigmaWidget && sigmaWidget.value !== data.sigmas) {
                                    sigmaWidget.value = data.sigmas;
                                    node.refreshCurve(data.sigmas);
                                }
                            }
                        }).catch(e => {
                            console.warn("[老李-工具箱] 生成sigma失败:", e);
                        });
                    }, 300);
                };

                // --- C. 监听 steps / scheduler / denoise 变化 ---
                const watchWidgets = ["steps", "scheduler", "denoise"];
                // 使用 setTimeout 确保所有 widget 已经创建完毕
                setTimeout(() => {
                    for (const wName of watchWidgets) {
                        const w = node.widgets.find(ww => ww.name === wName);
                        if (!w) continue;
                        const origCallback = w.callback;
                        w.callback = function(value) {
                            if (origCallback) origCallback.apply(this, arguments);
                            // 参数变化时，清空当前 sigma_string 并请求后端重新生成
                            if (sigmaWidget) sigmaWidget.value = "";
                            node.requestGenerate();
                        };
                    }
                }, 100);

                // --- D. 预设 & 按钮 ---
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
                    if (!sigmaWidget || !sigmaWidget.value) {
                        alert("当前没有sigma数据可保存");
                        return;
                    }
                    const n = prompt("输入预设名称:");
                    if (n) {
                        api.fetchApi("/laoli/sigmas/save", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ name: n, values: sigmaWidget.value })
                        }).then(r => r.json()).then(data => {
                            if (data.status === "success") {
                                refreshPresets();
                            } else {
                                alert("保存失败");
                            }
                        });
                    }
                });
                node.addWidget("button", "🗑️ 删除选中", null, () => {
                    const n = presetWidget.value;
                    if (n && n !== "选择预设..." && confirm(`删除 "${n}"?`)) {
                        api.fetchApi("/laoli/sigmas/delete", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ name: n })
                        }).then(() => {
                            presetWidget.value = "选择预设...";
                            refreshPresets();
                        });
                    }
                });
                node.addWidget("button", "🔄 清空数据", null, () => {
                    sigmaWidget.value = "";
                    node.laoli_sigmas = [];
                    node.setDirtyCanvas(true, true);
                });

                // --- E. 数据解析 ---
                node.refreshCurve = function(str) {
                    if (!str || typeof str !== 'string') {
                        node.laoli_sigmas = [];
                        node.setDirtyCanvas(true, true);
                        return;
                    }
                    const matches = str.match(/-?\d*\.?\d+(?:[eE][-+]?\d+)?/g);
                    // 所有值 clamp 到 [0, 1]
                    node.laoli_sigmas = matches ? matches.map(v => Math.max(0, Math.min(1, Number(v)))) : [];
                    node.setDirtyCanvas(true, true);
                };
                setTimeout(() => { if(sigmaWidget && sigmaWidget.value) node.refreshCurve(sigmaWidget.value); }, 200);

                // --- F. onExecuted：队列执行完成后后端返回的 UI 值 ---
                const origOnExecuted = node.onExecuted;
                node.onExecuted = function(message) {
                    if (origOnExecuted) origOnExecuted.apply(this, arguments);
                    // 后端 get_sigmas 返回的 ui.sigma_string
                    if (message && message.sigma_string && message.sigma_string[0]) {
                        const newVal = message.sigma_string[0];
                        if (sigmaWidget) sigmaWidget.value = newVal;
                        node.refreshCurve(newVal);
                    }
                };

                // --- G. 绘图组件 (Visualizer) ---
                const MIN_CHART_HEIGHT = 100;

                node.addCustomWidget({
                    name: "Visualizer",
                    type: "VISUALIZER",

                    computeSize: function(width) {
                        return [width, MIN_CHART_HEIGHT];
                    },

                    // mouse 方法：拦截 widget bounds 内的事件
                    mouse: function(event, pos, node) {
                        const etype = event.type;

                        if (etype === "pointerdown" || etype === "mousedown") {
                            const idx = node.getPointFromEvent ? node.getPointFromEvent(pos) : null;
                            if (idx === null) return false;
                            if (idx >= 0) {
                                node.dragIndex = idx;
                                node.isDragging = true;
                                node.setDirtyCanvas(true, true);
                            }
                            return true;
                        }

                        if (etype === "pointermove" || etype === "mousemove") {
                            if (node.isDragging && node.dragIndex !== -1 && node.chartLayout) {
                                const layout = node.chartLayout;
                                const bottomY = layout.y + layout.h;
                                const topY = layout.y;
                                let clampedY = Math.max(topY, Math.min(bottomY, pos[1]));
                                let ratio = (bottomY - clampedY) / layout.h;
                                ratio = Math.max(0, Math.min(1, ratio));
                                node.laoli_sigmas[node.dragIndex] = Math.max(0, Math.min(1, ratio * node.maxVal));
                                node.setDirtyCanvas(true, true);
                                return true;
                            }
                            const idx = node.getPointFromEvent ? node.getPointFromEvent(pos) : null;
                            const newHover = (idx !== null && idx >= 0) ? idx : -1;
                            if (node.hoverIndex !== newHover) {
                                node.hoverIndex = newHover;
                                node.setDirtyCanvas(true, true);
                            }
                            return false;
                        }

                        if (etype === "pointerup" || etype === "mouseup") {
                            if (node._finishDrag) node._finishDrag();
                            return true;
                        }

                        return false;
                    },

                    draw: function(ctx, node, widgetWidth, _y, _height) {
                        const bottomMargin = 10;
                        const realHeight = Math.max(MIN_CHART_HEIGHT, node.size[1] - _y - bottomMargin);
                        node.chartArea = { y: _y, height: realHeight };

                        const margin = { top: 10, right: 10, bottom: 20, left: 45 };
                        const chartX = margin.left;
                        const chartY = _y + margin.top;
                        const chartW = widgetWidth - margin.left - margin.right;
                        const chartH = realHeight - margin.top - margin.bottom;
                        node.chartLayout = { x: chartX, y: chartY, w: chartW, h: chartH };

                        ctx.save();
                        ctx.beginPath();
                        ctx.rect(0, _y, widgetWidth, realHeight);
                        ctx.clip();

                        // 背景
                        ctx.fillStyle = "#121212";
                        ctx.fillRect(chartX, chartY, chartW, chartH);
                        ctx.strokeStyle = "#333";
                        ctx.strokeRect(chartX, chartY, chartW, chartH);

                        const sigmas = node.laoli_sigmas;
                        if (!sigmas || sigmas.length === 0) {
                            ctx.fillStyle = "#666"; ctx.font = "14px Arial";
                            ctx.fillText("无数据 - 调整步数/调度器后自动生成", chartX + 20, chartY + chartH/2);
                            ctx.restore();
                            return;
                        }

                        // 坐标系 — Y 轴固定 [0, 1]
                        const maxVal = 1.0;
                        node.maxVal = maxVal;

                        // Y轴
                        ctx.font = "10px Arial"; ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.lineWidth = 1;
                        for (let v = 0; v <= 1.001; v += 0.1) {
                            const ratio = v;
                            const lineY = chartY + chartH - (ratio * chartH);
                            const isMajor = (Math.abs(v % 0.5) < 0.01 || v < 0.01);
                            ctx.strokeStyle = isMajor ? "#555" : "#222";
                            ctx.beginPath(); ctx.moveTo(chartX, lineY); ctx.lineTo(chartX + chartW, lineY); ctx.stroke();
                            ctx.fillStyle = "#aaa"; ctx.fillText(v.toFixed(1), chartX - 5, lineY);
                        }

                        // X轴
                        ctx.textAlign = "center"; ctx.textBaseline = "top";
                        const totalSteps = sigmas.length;
                        let stepInterval = totalSteps > 50 ? (totalSteps > 100 ? 20 : 10) : 5;
                        for (let i = 0; i < totalSteps; i += stepInterval) {
                            const ratio = i / (totalSteps - 1);
                            const lineX = chartX + ratio * chartW;
                            ctx.strokeStyle = "#222";
                            ctx.beginPath(); ctx.moveTo(lineX, chartY); ctx.lineTo(lineX, chartY + chartH); ctx.stroke();
                            ctx.fillStyle = "#666"; ctx.fillText(i.toString(), lineX, chartY + chartH + 2);
                        }

                        // 曲线 — 发光底层
                        const points = [];
                        const denominator = sigmas.length > 1 ? sigmas.length - 1 : 1;
                        for (let i = 0; i < sigmas.length; i++) {
                            const px = chartX + (i / denominator) * chartW;
                            const py = chartY + chartH - (sigmas[i] / maxVal) * chartH;
                            points.push({ x: px, y: py, val: sigmas[i], idx: i });
                        }

                        ctx.save();
                        ctx.strokeStyle = "rgba(0, 204, 255, 0.15)";
                        ctx.lineWidth = 6;
                        ctx.lineJoin = "round";
                        ctx.beginPath();
                        for (let i = 0; i < points.length; i++) {
                            if (i === 0) ctx.moveTo(points[i].x, points[i].y);
                            else ctx.lineTo(points[i].x, points[i].y);
                        }
                        ctx.stroke();
                        ctx.restore();

                        // 主曲线
                        ctx.strokeStyle = "#00ccff";
                        ctx.lineWidth = 1.2;
                        ctx.lineJoin = "round";
                        ctx.beginPath();
                        for (let i = 0; i < points.length; i++) {
                            if (i === 0) ctx.moveTo(points[i].x, points[i].y);
                            else ctx.lineTo(points[i].x, points[i].y);
                        }
                        ctx.stroke();

                        // 数据点
                        node.points = points;
                        const activeIdx = (node.dragIndex !== -1) ? node.dragIndex : node.hoverIndex;

                        if (points.length < 120) {
                            const baseR = points.length < 30 ? 3.5 : (points.length < 60 ? 2.5 : 1.8);
                            for (let i = 0; i < points.length; i++) {
                                const p = points[i];
                                if (i === activeIdx) continue;
                                ctx.fillStyle = "#00ccff";
                                ctx.beginPath();
                                ctx.arc(p.x, p.y, baseR, 0, Math.PI * 2);
                                ctx.fill();
                            }
                        }

                        // 交互高亮
                        if (activeIdx !== -1 && points[activeIdx]) {
                            const p = points[activeIdx];
                            const isDrag = node.dragIndex !== -1;

                            ctx.save();
                            ctx.strokeStyle = isDrag ? "rgba(255, 180, 50, 0.6)" : "rgba(255, 200, 80, 0.4)";
                            ctx.lineWidth = 0.8;
                            ctx.setLineDash([3, 3]);
                            ctx.beginPath();
                            ctx.moveTo(chartX, p.y); ctx.lineTo(chartX + chartW, p.y);
                            ctx.moveTo(p.x, chartY); ctx.lineTo(p.x, chartY + chartH);
                            ctx.stroke();
                            ctx.setLineDash([]);
                            ctx.restore();

                            const labelStep = `Step: ${p.idx}`;
                            const labelVal = `Val: ${p.val.toFixed(4)}`;
                            ctx.font = "bold 11px Arial";
                            const textW = Math.max(ctx.measureText(labelStep).width, ctx.measureText(labelVal).width);
                            let lx = p.x + 12;
                            if (lx + textW + 10 > chartX + chartW) lx = p.x - textW - 18;
                            let ly = p.y - 38;
                            if (ly < chartY + 2) ly = p.y + 16;

                            ctx.fillStyle = "rgba(30, 30, 30, 0.85)";
                            const pad = 4;
                            ctx.fillRect(lx - pad, ly - pad, textW + pad * 2 + 4, 30);
                            ctx.fillStyle = "#ffc850"; ctx.textAlign = "left";
                            ctx.fillText(labelStep, lx, ly + 8);
                            ctx.fillText(labelVal, lx, ly + 21);

                            ctx.beginPath();
                            ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
                            ctx.fillStyle = isDrag ? "rgba(255, 180, 50, 0.25)" : "rgba(0, 204, 255, 0.2)";
                            ctx.fill();
                            ctx.beginPath();
                            ctx.arc(p.x, p.y, 4.5, 0, Math.PI * 2);
                            ctx.fillStyle = isDrag ? "#ffb432" : "#fff";
                            ctx.fill();
                            ctx.strokeStyle = isDrag ? "#ffb432" : "#00ccff";
                            ctx.lineWidth = 1.5;
                            ctx.stroke();
                        }

                        ctx.restore();
                    }
                });


                // --- H. 智能交互 ---

                node.getPointFromEvent = function(localPos) {
                    if (!node.chartLayout) return null;
                    const mx = localPos[0];
                    const my = localPos[1];
                    const layout = node.chartLayout;

                    const area = node.chartArea || { y: layout.y, height: layout.h };
                    if (mx < 0 || mx > layout.x + layout.w + 15 ||
                        my < area.y || my > area.y + area.height) return null;

                    if (!node.points || !node.points.length) return -1;

                    if (node.points.length === 1) {
                        const p = node.points[0];
                        const dist = Math.sqrt((mx - p.x) ** 2 + (my - p.y) ** 2);
                        return dist < 30 ? 0 : -1;
                    }

                    let minD = Infinity;
                    let close = -1;
                    const xSpacing = layout.w / (node.points.length - 1);
                    const xRadius = Math.max(10, xSpacing * 0.6);

                    for (let i = 0; i < node.points.length; i++) {
                        const p = node.points[i];
                        const dx = Math.abs(mx - p.x);
                        if (dx < xRadius && dx < minD) {
                            minD = dx;
                            close = i;
                        }
                    }
                    return close;
                };

                // --- 鼠标事件 (widget bounds 外的区域) ---
                const origOnMouseDown = node.onMouseDown;
                node.onMouseDown = function(event, pos, graphCanvas) {
                    const idx = node.getPointFromEvent(pos);
                    if (idx === null) {
                        return origOnMouseDown ? origOnMouseDown.apply(this, arguments) : undefined;
                    }
                    if (idx >= 0) {
                        node.dragIndex = idx;
                        node.isDragging = true;
                        node.setDirtyCanvas(true, true);
                    }
                    return true;
                };

                const origOnMouseMove = node.onMouseMove;
                node.onMouseMove = function(event, pos, graphCanvas) {
                    if (node.isDragging && node.dragIndex !== -1) {
                        if (!node.chartLayout) return true;
                        const layout = node.chartLayout;
                        const bottomY = layout.y + layout.h;
                        const topY = layout.y;
                        let clampedY = Math.max(topY, Math.min(bottomY, pos[1]));
                        let ratio = (bottomY - clampedY) / layout.h;
                        ratio = Math.max(0, Math.min(1, ratio));
                        node.laoli_sigmas[node.dragIndex] = Math.max(0, Math.min(1, ratio * node.maxVal));
                        node.setDirtyCanvas(true, true);
                        return true;
                    }

                    const idx = node.getPointFromEvent(pos);
                    const newHover = (idx !== null && idx >= 0) ? idx : -1;
                    if (node.hoverIndex !== newHover) {
                        node.hoverIndex = newHover;
                        node.setDirtyCanvas(true, true);
                    }

                    return origOnMouseMove ? origOnMouseMove.apply(this, arguments) : undefined;
                };

                // 结束拖拽
                node._finishDrag = function() {
                    if (!node.isDragging) return false;
                    node.isDragging = false;
                    if (node.dragIndex !== -1) {
                        const w = node.widgets.find(w => w.name === "sigma_string");
                        if (w) w.value = "[" + node.laoli_sigmas.map(v => v.toFixed(4)).join(", ") + "]";
                        node.dragIndex = -1;
                    }
                    node.hoverIndex = -1;
                    node.setDirtyCanvas(true, true);
                    return true;
                };

                const origOnMouseUp = node.onMouseUp;
                node.onMouseUp = function(event, pos, graphCanvas) {
                    if (node._finishDrag()) return true;
                    return origOnMouseUp ? origOnMouseUp.apply(this, arguments) : undefined;
                };

                // 全局 pointerup 监听
                const canvas = app.canvas?.canvas || document.querySelector("canvas");
                if (canvas) {
                    const globalUp = () => { node._finishDrag(); };
                    canvas.addEventListener("pointerup", globalUp);
                    canvas.addEventListener("pointercancel", globalUp);
                    const origOnRemoved = node.onRemoved;
                    node.onRemoved = function() {
                        canvas.removeEventListener("pointerup", globalUp);
                        canvas.removeEventListener("pointercancel", globalUp);
                        if (origOnRemoved) origOnRemoved.apply(this, arguments);
                    };
                }

                // --- I. 控制节点最小高度 ---
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
