import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "Laoli.VisualSampler.TwoCol",
    async setup() {
        api.addEventListener("laoli_vs_update", ({ detail }) => {
            if (!detail || !detail.node_id) return;
            const node = app.graph.getNodeById(detail.node_id);
            if (!node || node.type !== "LaoliVisualSampler") return;
            const w = node.widgets?.find(w => w.name === detail.target);
            if (w && w.value !== detail.text) w.value = detail.text;
            if (node.refreshCurve) node.refreshCurve(detail.target, detail.text || "");
        });

        api.addEventListener("laoli_vs_preview", ({ detail }) => {
            if (!detail || !detail.node_id) return;
            const node = app.graph.getNodeById(detail.node_id);
            if (!node || node.type !== "LaoliVisualSampler") return;
            if (!node._previewImages) node._previewImages = {};
            if (!node._previewMeta) node._previewMeta = {};
            const b64 = detail.image;
            if (b64) {
                const img = new window.Image();
                img.onload = () => {
                    node._previewImages[detail.stage] = img;
                    node._previewMeta[detail.stage] = {
                        step: detail.step, total: detail.total, final: !!detail.final,
                        lat_w: detail.lat_w, lat_h: detail.lat_h,
                        nat_w: img.naturalWidth, nat_h: img.naturalHeight,
                    };
                    node.setDirtyCanvas(true, true);
                };
                img.src = "data:image/jpeg;base64," + b64;
            }
        });

        api.addEventListener("laoli_vs_status", ({ detail }) => {
            if (!detail || !detail.node_id) return;
            const node = app.graph.getNodeById(detail.node_id);
            if (!node || node.type !== "LaoliVisualSampler") return;
            if (!node._statusMsg) node._statusMsg = {};
            node._statusMsg.text = detail.msg || "";
            node._statusMsg.time = Date.now();
            node.setDirtyCanvas(true, true);
        });
    },

    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== "LaoliVisualSampler") return;
        const orig = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = orig?.apply(this, arguments);
            try { buildUI(this); } catch (e) { console.error(e); }
            return r;
        };
    }
});

// ════════════════════════════════════════════════════════════════════════
// 核心算法：软笔刷 + 强推挤(Push)逻辑
// ════════════════════════════════════════════════════════════════════════

const EPSILON = 0.0001;

// 推挤逻辑：High -> Low
function applyPushConstraint(sigmas, changedIdx, newVal, anchors) {
    const len = sigmas.length;
    sigmas[changedIdx] = newVal;

    // 1. 向左扫描 (上游)
    let currentFloor = newVal;
    for (let i = changedIdx - 1; i >= 0; i--) {
        if (anchors && i === anchors.first) {
            if (sigmas[i+1] > sigmas[i] - EPSILON) {
                sigmas[i+1] = Math.max(0, sigmas[i] - EPSILON);
            }
            break; 
        }
        if (sigmas[i] < currentFloor + EPSILON) {
            sigmas[i] = Math.min(1, currentFloor + EPSILON);
            currentFloor = sigmas[i];
        } else {
            break; 
        }
    }

    // 2. 向右扫描 (下游)
    let currentCeiling = newVal;
    for (let i = changedIdx + 1; i < len; i++) {
        if (anchors && i === anchors.last) {
            if (sigmas[i-1] < sigmas[i] + EPSILON) {
                sigmas[i-1] = Math.min(1, sigmas[i] + EPSILON);
            }
            break;
        }
        if (sigmas[i] > currentCeiling - EPSILON) {
            sigmas[i] = Math.max(0, currentCeiling - EPSILON);
            currentCeiling = sigmas[i];
        } else {
            break;
        }
    }
}

// 软笔刷模式 (增加 strength 参数)
function applySoftDrag(sigmas, dragIdx, newValue, radius, strength, anchors) {
    const result = [...sigmas];
    const oldValue = sigmas[dragIdx];
    const delta = newValue - oldValue;
    if (Math.abs(delta) < 1e-6) return result;
    
    const len = sigmas.length;
    const radiusSq2 = 2 * radius * radius;

    for (let i = 0; i < len; i++) {
        if (anchors && (i === anchors.first || i === anchors.last)) continue;
        
        const dist = Math.abs(i - dragIdx);
        const weight = Math.exp(-(dist * dist) / radiusSq2);
        
        // 核心修改：如果是邻居点，应用强度系数
        // 强度 1.0 = 完全跟随高斯曲线
        // 强度 0.0 = 邻居不动
        let effectiveWeight = weight;
        if (i !== dragIdx) {
            effectiveWeight *= strength;
        }

        if (effectiveWeight < 0.001) continue;

        let candidate = sigmas[i] + delta * effectiveWeight;
        candidate = Math.max(0, Math.min(1, candidate));
        result[i] = candidate;
    }

    enforceMonotonicity(result, anchors);
    return result;
}

function enforceMonotonicity(sigmas, anchors) {
    const len = sigmas.length;
    for (let i = 1; i < len; i++) {
        if (anchors && i === anchors.first) continue;
        const maxVal = sigmas[i - 1] - EPSILON;
        if (sigmas[i] > maxVal) sigmas[i] = Math.max(0, maxVal);
    }
    for (let i = len - 2; i >= 0; i--) {
        if (anchors && i === anchors.last) continue;
        const minVal = sigmas[i + 1] + EPSILON;
        if (sigmas[i] < minVal) sigmas[i] = Math.min(1, minVal);
    }
}

function calculateWeights(sigmas, dragIdx, radius) {
    const weights = new Array(sigmas.length).fill(0);
    const radiusSq2 = 2 * radius * radius;
    for (let i = 0; i < sigmas.length; i++) {
        const dist = Math.abs(i - dragIdx);
        weights[i] = Math.exp(-(dist * dist) / radiusSq2);
    }
    return weights;
}

function validateSigmaSequence(sigmas) {
    const issues = [];
    for (let i = 1; i < sigmas.length; i++) {
        if (sigmas[i-1] - sigmas[i] < EPSILON - 1e-7) {
            issues.push(`步骤 ${i}: 间距过小或逆增`);
        }
    }
    return issues;
}

function getStage2Guidance(sigmas) {
    if (!sigmas || !sigmas.length) return null;
    const first = sigmas[0];
    if (first > 0.95) return "⚠ 二阶段起始 sigma 过高，建议 0.7-0.9";
    if (first < 0.3) return "⚠ 二阶段起始 sigma 过低，精修效果可能不明显";
    return null;
}

// ════════════════════════════════════════════════════════════════════════
// 主 UI
// ════════════════════════════════════════════════════════════════════════

function buildUI(node) {
    const C = {
        s1: { hdr: "#1a2f28", hdrTx: "#5ae8a8", accent: "#38c888", row: "#1c2824", rowH: "#243830", tagDot: "#38c888" },
        up: { hdr: "#2c2818", hdrTx: "#e8c860", accent: "#c8a840", row: "#282418", rowH: "#383220", tagDot: "#c8a840" },
        s2: { hdr: "#1a2238", hdrTx: "#60a8f8", accent: "#4888e0", row: "#1c2030", rowH: "#283048", tagDot: "#4888e0" },
        sync: { row: "#241a24", rowH: "#302030", tagDot: "#b060b0", accent: "#b060b0" },
        label: "#a0aab8", value: "#e0e8f0", valueDim: "#90a0b0", sep: "#282c34",
    };
    const PC = {
        hdr: "#241c30", hdrTx: "#c0a0e0", accent: "#9070c0",
        row: "#201a28", rowH: "#2c2438",
        btnSave: "#1e2e22", btnSaveH: "#284030", btnSaveTx: "#70e088",
        btnDel: "#2e1e1e", btnDelH: "#3c2828", btnDelTx: "#e07070",
    };

    const ROW = 26, GAP = 1, HDR = 26, PAD = 10, SEC = 8;
    const LR = 0.38, MIN_CHART_H = 195, SIGMA_TEXT_H = 50;
    const EDGE_PAD = 80, CENTER_GAP = 12, PREVIEW_GAP = 6;
    const PREVIEW_H = 280;
    const SLIDER_H = 20;
    const CONTROLS_H = 46; // Radius + Strength area height

    const SC_LIST = ["fixed", "increment", "decrement", "randomize"];
    const SC_LBL = { fixed: "固定", increment: "递增", decrement: "递减", randomize: "随机" };
    const SC_ICON = { fixed: "●", increment: "▲", decrement: "▼", randomize: "◆" };
    const ONE_DECIMAL = ["CFG_1", "CFG_2", "放大倍数"];

    node._cr = {};
    node._hover = null;
    node._activeSigma = null;
    node._suppressRegen = false;
    node._previewImages = {};
    node._previewMeta = {};
    node._statusMsg = { text: "", time: 0 };
    node._minNodeH = 1380;

    if (!node.properties) node.properties = {};
    if (!node.properties._sc1) node.properties._sc1 = "randomize";
    if (!node.properties._sc2) node.properties._sc2 = "randomize";
    if (!node.properties._selectedPreset) node.properties._selectedPreset = "";
    
    // 半径默认 2.0
    if (node.properties._softSelectionRadius1 == null) node.properties._softSelectionRadius1 = 2.0;
    if (node.properties._softSelectionRadius2 == null) node.properties._softSelectionRadius2 = 2.0;
    // 强度默认 1.0
    if (node.properties._softSelectionStrength1 == null) node.properties._softSelectionStrength1 = 1.0;
    if (node.properties._softSelectionStrength2 == null) node.properties._softSelectionStrength2 = 1.0;

    const MAX_RADIUS = 20.0;

    function getRadiusForTarget(target) {
        return target === "曲线_1" ? (node.properties._softSelectionRadius1 ?? 2.0) : (node.properties._softSelectionRadius2 ?? 2.0);
    }
    function setRadiusForTarget(target, v) {
        const nv = Math.max(0.1, Math.min(MAX_RADIUS, Math.round(v * 10) / 10));
        if (target === "曲线_1") node.properties._softSelectionRadius1 = nv;
        else node.properties._softSelectionRadius2 = nv;
        return nv;
    }

    function getStrengthForTarget(target) {
        return target === "曲线_1" ? (node.properties._softSelectionStrength1 ?? 1.0) : (node.properties._softSelectionStrength2 ?? 1.0);
    }
    function setStrengthForTarget(target, v) {
        const nv = Math.max(0.0, Math.min(1.0, Math.round(v * 100) / 100));
        if (target === "曲线_1") node.properties._softSelectionStrength1 = nv;
        else node.properties._softSelectionStrength2 = nv;
        return nv;
    }

    node.size[0] = Math.max(node.size[0], 640);
    node.size[1] = Math.max(node.size[1], node._minNodeH);

    node._sigmaData = {
        "曲线_1": { sigmas: [], dragIndex: -1, hoverIndex: -1, isDragging: false, dragStartPos: null, chartLayout: null, points: [], weights: null },
        "曲线_2": { sigmas: [], dragIndex: -1, hoverIndex: -1, isDragging: false, dragStartPos: null, chartLayout: null, points: [], weights: null }
    };

    const MAX_UNDO = 20;
    node._sigmaUndo = { "曲线_1": [], "曲线_2": [] };

    function pushUndo(target) {
        const sd = node._sigmaData[target];
        const stack = node._sigmaUndo[target];
        stack.push([...sd.sigmas]);
        if (stack.length > MAX_UNDO) stack.shift();
    }

    function popUndo(target) {
        const stack = node._sigmaUndo[target];
        if (!stack.length) return false;
        const sd = node._sigmaData[target];
        sd.sigmas = stack.pop();
        const w = fw(target);
        if (w) {
            w.value = "[" + sd.sigmas.map(v => v.toFixed(4)).join(", ") + "]";
            try { w.callback?.(w.value); } catch (_e) {}
        }
        node.setDirtyCanvas(true, true);
        return true;
    }

    const hideAllWidgets = () => {
        if (!node.widgets) return;
        for (const w of node.widgets) {
            w.hidden = true;
            w.computeSize = () => [0, 0];
            w.draw = function () {};
            if (w.name && w.name.includes("control_after")) w.value = "fixed";
        }
    };
    hideAllWidgets();

    const hideDom = () => {
        if (!node.widgets) return;
        for (const w of node.widgets) {
            const el = w.inputEl || w.element;
            if (el) Object.assign(el.style, {
                display: "none", pointerEvents: "none", overflow: "hidden",
                height: "0px", position: "absolute", left: "-9999px"
            });
        }
    };
    setTimeout(hideDom, 200);

    const fw = n => node.widgets?.find(w => w.name === n);
    const wmeta = w => {
        const o = w?.options || {};
        return { min: o.min ?? 0, max: o.max ?? 100, step: o.step ?? (o.round ? 1 : 0.01) };
    };
    const rr = (ctx, x, y, w, h, r) => {
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.arcTo(x + w, y, x + w, y + h, r);
        ctx.arcTo(x + w, y + h, x, y + h, r);
        ctx.arcTo(x, y + h, x, y, r);
        ctx.arcTo(x, y, x + w, y, r);
        ctx.closePath();
    };

    node.refreshCurve = function (target, str) {
        const sd = node._sigmaData[target];
        if (!sd) return;
        if (!str || typeof str !== "string") {
            sd.sigmas = [];
            node.setDirtyCanvas(true, true);
            return;
        }
        const m = str.match(/-?\d*\.?\d+(?:[eE][-+]?\d+)?/g);
        sd.sigmas = m
            ? m.map(v => Math.max(0, Math.min(1, parseFloat(v)))).filter(v => !isNaN(v))
            : [];
        node.setDirtyCanvas(true, true);
    };

    node._finishSigmaDrag = function (target) {
        const sd = node._sigmaData[target];
        if (!sd || !sd.isDragging) return false;
        sd.isDragging = false;
        sd.weights = null;
        if (sd.dragIndex !== -1) {
            const w = node.widgets?.find(w => w.name === target);
            if (w) {
                const nv = "[" + sd.sigmas.map(v => v.toFixed(4)).join(", ") + "]";
                if (w.value !== nv) {
                    w.value = nv;
                    try { w.callback?.(nv); } catch (_e) {}
                }
            }
            sd.dragIndex = -1;
        }
        sd.hoverIndex = -1;
        node.setDirtyCanvas(true, true);
        return true;
    };

    let _genTimer = {};
    node.requestGenerate = function (target) {
        if (node._suppressRegen) return;
        if (_genTimer[target]) clearTimeout(_genTimer[target]);
        _genTimer[target] = setTimeout(() => {
            if (node._suppressRegen) return;
            const cw = fw(target);
            if (cw && cw.value && cw.value.trim()) return;
            const s1 = target === "曲线_1";
            api.fetchApi("/laoli/vsampler/generate_sigmas", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    steps: fw(s1 ? "步数_1" : "步数_2")?.value ?? 20,
                    scheduler: fw(s1 ? "调度器_1" : "调度器_2")?.value ?? "normal",
                    denoise: fw(s1 ? "降噪_1" : "降噪_2")?.value ?? 1.0,
                    node_id: String(node.id),
                    target
                })
            })
                .then(r => r.json())
                .then(d => {
                    if (d.status === "success" && d.sigmas) {
                        const w = fw(target);
                        if (w) {
                            w.value = d.sigmas;
                            node.refreshCurve(target, d.sigmas);
                        }
                    }
                })
                .catch(() => {});
        }, 300);
    };

    node._clearAndRegenerate = function(target) {
        const w = fw(target);
        if (w) w.value = "";
        node.refreshCurve(target, "");
        if (_genTimer[target]) clearTimeout(_genTimer[target]);
        node.requestGenerate(target);
    };

    node._cancelPendingGen = function() {
        for (const t in _genTimer) {
            if (_genTimer[t]) { clearTimeout(_genTimer[t]); _genTimer[t] = null; }
        }
    };

    setTimeout(() => {
        hideAllWidgets();
        hideDom();
        for (const t of ["曲线_1", "曲线_2"]) {
            const w = fw(t);
            if (w && w.value) node.refreshCurve(t, w.value);
        }
        const wm = {
            "步数_1": "曲线_1", "调度器_1": "曲线_1", "降噪_1": "曲线_1",
            "步数_2": "曲线_2", "调度器_2": "曲线_2", "降噪_2": "曲线_2"
        };
        for (const [n, t] of Object.entries(wm)) {
            const w = fw(n);
            if (!w) continue;
            const oc = w.callback;
            w.callback = function (v) {
                oc?.apply(this, arguments);
                node._clearAndRegenerate(t);
            };
        }
        for (const t of ["曲线_1", "曲线_2"]) {
            const w = fw(t);
            if (!w || !w.value) node.requestGenerate(t);
        }
        
        const seedW1 = fw("随机种_1");
        if (seedW1) {
            seedW1.beforeQueued = function() {
                if (node.properties["_sc1"] === "randomize") {
                    this.value = Math.floor(Math.random() * 999999999999999);
                }
                const syncW = fw("种子同步");
                if (syncW && syncW.value === "enable") {
                    const w2 = fw("随机种_2");
                    if (w2) w2.value = this.value;
                }
            };
        }
        const seedW2 = fw("随机种_2");
        if (seedW2) {
            seedW2.beforeQueued = function() {
                const syncW = fw("种子同步");
                const isSync = syncW && syncW.value === "enable";
                if (isSync) {
                    const w1 = fw("随机种_1");
                    if (w1) this.value = w1.value;
                } else if (node.properties["_sc2"] === "randomize") {
                    this.value = Math.floor(Math.random() * 999999999999999);
                }
            };
        }
    }, 500);

    let _presetsCache = null;
    const loadPresets = cb => {
        if (_presetsCache) { cb(_presetsCache); return; }
        api.fetchApi("/laoli/vsampler/presets")
            .then(r => r.json())
            .then(d => { _presetsCache = d; cb(d || {}); })
            .catch(() => cb({}));
    };
    const savePreset = (name, config) => {
        api.fetchApi("/laoli/vsampler/save", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, config })
        })
            .then(r => r.json())
            .then(d => { if (d.status === "success") _presetsCache = null; })
            .catch(() => {});
    };
    const deletePreset = name => {
        api.fetchApi("/laoli/vsampler/delete", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name })
        })
            .then(r => r.json())
            .then(d => { if (d.status === "success") _presetsCache = null; })
            .catch(() => {});
    };

    function localToScreen(lx, ly) {
        const cv = app.canvas;
        if (!cv?.canvas) return null;
        const rect = cv.canvas.getBoundingClientRect(), ds = cv.ds;
        const gx = node.pos[0] + lx, gy = node.pos[1] + ly;
        if (ds?.convertOffsetToCanvas) {
            const [cx, cy] = ds.convertOffsetToCanvas([gx, gy]);
            return [rect.left + cx, rect.top + cy, ds.scale];
        }
        const s = ds?.scale || 1, ox = ds?.offset?.[0] || 0, oy = ds?.offset?.[1] || 0;
        return [rect.left + (gx + ox) * s, rect.top + (gy + oy) * s, s];
    }

    function showInput(cr, curVal, onOK) {
        const coords = localToScreen(cr.vx, cr.y);
        if (!coords) {
            const v = prompt("输入:", curVal);
            if (v !== null) onOK(v);
            return;
        }
        const [sx, sy, scale] = coords;
        const sw = Math.max(80, cr.vw * scale), sh = Math.max(22, cr.h * scale);
        const fs = Math.max(12, 12 * scale);

        const mask = document.createElement("div");
        Object.assign(mask.style, {
            position: "fixed", inset: "0", zIndex: "9998", background: "rgba(0,0,0,0.12)"
        });
        const inp = document.createElement("input");
        inp.type = "text";
        inp.value = String(curVal);
        Object.assign(inp.style, {
            position: "fixed", left: sx + "px", top: sy + "px",
            width: sw + "px", height: sh + "px", fontSize: fs + "px",
            fontWeight: "600", background: "#0c1018", color: "#f0f4ff",
            border: "1.5px solid #4080e0", borderRadius: "3px",
            padding: "0 6px", outline: "none", boxSizing: "border-box", zIndex: "9999"
        });
        document.body.append(mask, inp);
        requestAnimationFrame(() => { inp.focus(); inp.select(); });

        let done = false;
        const close = ok => {
            if (done) return;
            done = true;
            if (ok) onOK(inp.value);
            mask.remove();
            inp.remove();
        };
        inp.addEventListener("keydown", e => {
            if (e.key === "Enter") { e.preventDefault(); close(true); }
            if (e.key === "Escape") { e.preventDefault(); close(false); }
        });
        inp.addEventListener("blur", () => close(true));
        mask.addEventListener("pointerdown", e => { e.preventDefault(); close(true); });
    }

    function showSigmaTextInput(cr, target) {
        const w = fw(target), origVal = w ? (w.value || "") : "";
        const coords = localToScreen(cr.x, cr.y);
        if (!coords) {
            const v = prompt("Sigma:", origVal);
            if (v !== null) {
                if (w) w.value = v;
                node.refreshCurve(target, v);
            }
            return;
        }
        const [sx, sy, scale] = coords;
        const sw = Math.max(240, cr.w * scale), sh = Math.max(60, cr.h * scale);
        const fs = Math.max(11, 11 * scale);

        const mask = document.createElement("div");
        Object.assign(mask.style, {
            position: "fixed", inset: "0", zIndex: "9998", background: "rgba(0,0,0,0.18)"
        });
        const ta = document.createElement("textarea");
        ta.value = origVal;
        Object.assign(ta.style, {
            position: "fixed", left: sx + "px", top: sy + "px",
            width: sw + "px", height: sh + "px", fontSize: fs + "px",
            fontFamily: "'Consolas',monospace", background: "#0c1018", color: "#c0d8f0",
            border: "1.5px solid #4080e0", borderRadius: "4px", padding: "4px 6px",
            outline: "none", boxSizing: "border-box", zIndex: "9999",
            resize: "both", lineHeight: "1.4"
        });
        document.body.append(mask, ta);
        requestAnimationFrame(() => {
            ta.focus();
            ta.setSelectionRange(ta.value.length, ta.value.length);
        });

        ta.addEventListener("input", () => {
            if (w) w.value = ta.value;
            node.refreshCurve(target, ta.value);
        });

        let done = false;
        const close = ok => {
            if (done) return;
            done = true;
            if (!ok) {
                if (w) w.value = origVal;
                node.refreshCurve(target, origVal);
            }
            mask.remove();
            ta.remove();
            node.setDirtyCanvas(true, true);
        };
        ta.addEventListener("keydown", e => {
            if (e.key === "Escape") { e.preventDefault(); close(false); }
            if (e.key === "Enter" && e.ctrlKey) { e.preventDefault(); close(true); }
        });
        mask.addEventListener("pointerdown", e => { e.preventDefault(); close(true); });
    }

    function showSavePresetDialog(btnCr) {
        const doSave = nm => {
            const cfg = {};
            for (const w of node.widgets) {
                if (w.name) cfg[w.name] = w.value;
            }
            cfg._softSelectionRadius1 = node.properties._softSelectionRadius1;
            cfg._softSelectionRadius2 = node.properties._softSelectionRadius2;
            cfg._softSelectionStrength1 = node.properties._softSelectionStrength1;
            cfg._softSelectionStrength2 = node.properties._softSelectionStrength2;
            savePreset(nm, cfg);
            node.properties._selectedPreset = nm;
            node.setDirtyCanvas(true, true);
        };
        const coords = localToScreen(btnCr.x + btnCr.w + 6, btnCr.y);
        if (!coords) {
            const nm = prompt("预设名称:");
            if (nm?.trim()) doSave(nm.trim());
            return;
        }
        const [sx, sy, scale] = coords;
        const fs = Math.max(12, 12 * scale);
        const inW = Math.max(150, 150 * scale), inH = Math.max(24, ROW * scale);

        const mask = document.createElement("div");
        Object.assign(mask.style, {
            position: "fixed", inset: "0", zIndex: "9998", background: "rgba(0,0,0,0.18)"
        });
        const panel = document.createElement("div");
        Object.assign(panel.style, {
            position: "fixed", left: sx + "px", top: (sy - 2) + "px",
            display: "flex", gap: (4 * scale) + "px", alignItems: "center",
            padding: (5 * scale) + "px " + (8 * scale) + "px",
            background: "#1a1e28", border: "1.5px solid #9070c0", borderRadius: "5px", zIndex: "9999"
        });
        const lbl = document.createElement("span");
        lbl.textContent = "名称:";
        Object.assign(lbl.style, { color: "#a0aab8", fontSize: fs + "px", whiteSpace: "nowrap" });
        const inp = document.createElement("input");
        inp.type = "text";
        Object.assign(inp.style, {
            width: inW + "px", height: inH + "px", fontSize: fs + "px", fontWeight: "600",
            background: "#0c1018", color: "#f0f4ff", border: "1px solid #4060a0",
            borderRadius: "3px", padding: "0 6px", outline: "none"
        });
        const okBtn = document.createElement("button");
        okBtn.textContent = "✓";
        Object.assign(okBtn.style, {
            height: inH + "px", background: "#1e3020", color: "#70e088",
            border: "1px solid #408050", borderRadius: "3px", cursor: "pointer",
            padding: "0 " + (8 * scale) + "px"
        });
        panel.append(lbl, inp, okBtn);
        document.body.append(mask, panel);
        requestAnimationFrame(() => inp.focus());

        let done = false;
        const close = ok => {
            if (done) return;
            done = true;
            if (ok && inp.value.trim()) doSave(inp.value.trim());
            mask.remove();
            panel.remove();
        };
        okBtn.addEventListener("click", () => close(true));
        inp.addEventListener("keydown", e => {
            if (e.key === "Enter") { e.preventDefault(); close(true); }
            if (e.key === "Escape") { e.preventDefault(); close(false); }
        });
        mask.addEventListener("pointerdown", e => { e.preventDefault(); close(false); });
    }

    function drawRow(ctx, r, x, y, cw, th) {
        const hov = node._hover === r.n || (node._hover && node._hover.startsWith(r.n + "_"));
        const lw = Math.floor(cw * LR), vx = x + lw, vw = cw - lw;
        const syncW = fw("种子同步");
        const isSync = syncW && syncW.value === "enable";
        const isLocked = isSync && (r.n === "随机种_2" || r.n === "_sc2");

        ctx.fillStyle = hov && !isLocked ? th.rowH : th.row;
        rr(ctx, x, y, cw, ROW, 3);
        ctx.fill();

        if (isLocked) {
            ctx.fillStyle = "rgba(0,0,0,0.25)";
            rr(ctx, x, y, cw, ROW, 3);
            ctx.fill();
        }

        ctx.fillStyle = C.sep;
        ctx.fillRect(x + 8, y + ROW - 0.5, cw - 16, 0.5);

        ctx.fillStyle = th.tagDot;
        ctx.globalAlpha = hov ? 1 : 0.5;
        ctx.beginPath();
        ctx.arc(x + 7, y + ROW / 2, 2, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;

        ctx.font = "12px 'Microsoft YaHei','PingFang SC',sans-serif";
        ctx.fillStyle = C.label;
        ctx.textBaseline = "middle";
        ctx.fillText(r.l, x + 14, y + ROW / 2);

        let txt = "", tc = C.value;
        if (r.k === "seedctrl") {
            if (isLocked) {
                txt = "🔗 跟随一阶段";
                tc = "#d080d0";
            } else {
                const m = node.properties[r.n] || "fixed";
                txt = (SC_ICON[m] || "●") + " " + (SC_LBL[m] || m);
                tc = C.valueDim;
            }
        } else if (r.n === "随机种_2" && isSync) {
            const w1 = fw("随机种_1");
            const sv = w1 ? String(w1.value) : "—";
            txt = "🔗 " + (sv.length > 14 ? sv.slice(0, 14) + "…" : sv);
            tc = "#d080d0";
        } else {
            const w = fw(r.n);
            if (!w) {
                txt = "—";
                tc = C.valueDim;
            } else if (r.k === "seed") {
                txt = String(w.value);
                if (txt.length > 16) txt = txt.slice(0, 16) + "…";
            } else if (r.k === "combo") {
                txt = String(w.value);
            } else if (r.k === "float") {
                txt = typeof w.value === "number"
                    ? (ONE_DECIMAL.includes(r.n) ? w.value.toFixed(1) : w.value.toFixed(2))
                    : String(w.value);
            } else {
                txt = String(w.value);
            }
        }

        const arrow = (!isLocked && (r.k === "combo" || r.k === "seedctrl")) ? " ▾" : "";
        ctx.save();
        ctx.beginPath();
        
        let rightMargin = 4;
        if (r.k === "seed" && !isLocked) rightMargin = 28; 
        
        const canStep = !isLocked && (r.k === "int" || r.k === "float" || r.k === "seed");
        
        let stepperW = 0;
        if (hov && canStep) {
            stepperW = 50; 
            rightMargin += stepperW;
            
            const stepY = y + ROW / 2;
            const stepX = vx + vw - rightMargin + (r.k === "seed" ? 24 : 0);
            
            // 左箭头
            const decrX = stepX; 
            const decrHover = node._hover === r.n + "_decr";
            ctx.fillStyle = decrHover ? th.accent : "#606878";
            ctx.beginPath();
            ctx.moveTo(decrX + 6, stepY);
            ctx.lineTo(decrX + 6 + 8, stepY - 8);
            ctx.lineTo(decrX + 6 + 8, stepY + 8);
            ctx.fill();
            node._cr[r.n + "_decr"] = { x: decrX, y: y, w: 22, h: ROW, k: "step_decr", name: r.n, type: r.k };

            // 右箭头
            const incrX = stepX + 28;
            const incrHover = node._hover === r.n + "_incr";
            ctx.fillStyle = incrHover ? th.accent : "#606878";
            ctx.beginPath();
            ctx.moveTo(incrX + 16, stepY);
            ctx.lineTo(incrX + 16 - 8, stepY - 8);
            ctx.lineTo(incrX + 16 - 8, stepY + 8);
            ctx.fill();
            node._cr[r.n + "_incr"] = { x: incrX, y: y, w: 22, h: ROW, k: "step_incr", name: r.n, type: r.k };
        }

        ctx.rect(vx, y, vw - rightMargin, ROW);
        ctx.clip();

        ctx.font = "600 12px 'Segoe UI',system-ui,sans-serif";
        ctx.fillStyle = tc;
        ctx.fillText(txt, vx + 4, y + ROW / 2);

        if (arrow) {
            const tw2 = ctx.measureText(txt).width;
            ctx.fillStyle = C.label;
            ctx.font = "10px sans-serif";
            ctx.fillText(arrow, vx + 4 + tw2 + 2, y + ROW / 2);
        }
        ctx.restore();

        if (r.k === "seed" && !isLocked) {
            const btnW = 20, btnX = vx + vw - btnW - 4;
            const isHoverDice = node._hover === "seed_btn" + r.n;
            ctx.fillStyle = C.sep;
            ctx.fillRect(btnX - 1, y + 4, 1, ROW - 8);
            ctx.font = "14px Segoe UI Emoji, sans-serif";
            ctx.textAlign = "center";
            ctx.fillStyle = isHoverDice ? "#ffffff" : "#a0aab8";
            ctx.fillText("🎲", btnX + btnW / 2, y + ROW / 2);
            ctx.textAlign = "left";
            node._cr["seed_btn" + r.n] = { x: btnX, y: y, w: btnW, h: ROW, k: "dice", seedName: r.n };
            node._cr[r.n] = { x, y, w: cw - btnW - 6 - stepperW, h: ROW, vx, vw: vw - btnW - 6 - stepperW, k: r.k };
        } else {
            if (hov && !isLocked && !canStep) {
                ctx.fillStyle = th.accent;
                ctx.globalAlpha = 0.6;
                rr(ctx, x + cw - 3, y + 6, 2, ROW - 12, 1);
                ctx.fill();
                ctx.globalAlpha = 1;
            }
            node._cr[r.n] = { x, y, w: cw - stepperW, h: ROW, vx, vw: vw - stepperW, k: r.k };
        }
    }

    function drawSection(ctx, sec, x, y, cw) {
        const th = sec.th;
        ctx.fillStyle = th.hdr;
        rr(ctx, x, y, cw, HDR, 4);
        ctx.fill();
        ctx.fillStyle = th.accent;
        rr(ctx, x, y, 3, HDR, 2);
        ctx.fill();

        ctx.font = "bold 12px 'Microsoft YaHei','PingFang SC',sans-serif";
        ctx.fillStyle = th.hdrTx;
        ctx.textBaseline = "middle";
        ctx.fillText(sec.title, x + 12, y + HDR / 2);

        let cy = y + HDR + GAP;
        for (const r of sec.rows) {
            drawRow(ctx, r, x, cy, cw, th);
            cy += ROW + GAP;
        }
        return cy;
    }

    function drawSigmaChart(ctx, target, x, y, w, h, themeColor, isCtrlMode = false) {
        const sd = node._sigmaData[target];
        const mg = { top: 10, right: 10, bottom: 20, left: 38 };
        const cX = x + mg.left, cY = y + mg.top;
        const cW = w - mg.left - mg.right, cH = h - mg.top - mg.bottom;
        sd.chartLayout = { x: cX, y: cY, w: cW, h: cH };

        ctx.fillStyle = "#0e1018";
        rr(ctx, x, y, w, h, 5);
        ctx.fill();
        ctx.strokeStyle = "#2a2e38";
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.fillStyle = "#111318";
        ctx.fillRect(cX, cY, cW, cH);

        const sigmas = sd.sigmas;
        if (!sigmas || !sigmas.length) {
            ctx.fillStyle = "#555";
            ctx.font = "11px 'Microsoft YaHei',sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("等待生成曲线…", x + w / 2, y + h / 2);
            ctx.textAlign = "left";
            ctx.textBaseline = "alphabetic";
            return;
        }

        const activeIdx = sd.dragIndex !== -1 ? sd.dragIndex : sd.hoverIndex;
        if (activeIdx !== -1 && sd.weights) {
            const radius = getRadiusForTarget(target);
            const spreadPx = (radius / Math.max(1, sigmas.length - 1)) * cW;
            const p = sd.points[activeIdx];
            if (p) {
                const grad = ctx.createRadialGradient(
                    p.x, cY + cH / 2, 0,
                    p.x, cY + cH / 2, spreadPx * 2
                );
                grad.addColorStop(0, themeColor.replace("rgb", "rgba").replace(")", ",0.15)"));
                grad.addColorStop(0.5, themeColor.replace("rgb", "rgba").replace(")", ",0.06)"));
                grad.addColorStop(1, "transparent");

                ctx.save();
                ctx.beginPath();
                ctx.rect(cX, cY, cW, cH);
                ctx.clip();
                ctx.fillStyle = grad;
                ctx.fillRect(p.x - spreadPx * 2, cY, spreadPx * 4, cH);
                ctx.restore();
            }
        }

        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        
        for (let i = 0; i <= 10; i++) {
            const v = i / 10, lineY = cY + cH - v * cH;
            const isKey = (i === 0 || i === 5 || i === 10);
            
            ctx.beginPath();
            ctx.moveTo(cX, lineY);
            ctx.lineTo(cX + cW, lineY);
            
            if (isKey) {
                ctx.strokeStyle = "#404858"; 
                ctx.lineWidth = 1;
                ctx.stroke();
                ctx.font = "bold 9px Arial";
                ctx.fillStyle = "#aab0c0";
            } else {
                ctx.strokeStyle = "#1e2228"; 
                ctx.lineWidth = 0.8;
                ctx.stroke();
                ctx.font = "8px Arial";
                ctx.fillStyle = "#505868";
            }
            ctx.fillText(v.toFixed(1), cX - 5, lineY);
        }

        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        const tot = sigmas.length;
        let si = tot > 50 ? (tot > 100 ? 20 : 10) : 5;
        if (tot <= 10) si = 1;
        for (let i = 0; i < tot; i += si) {
            const lx2 = cX + (tot > 1 ? i / (tot - 1) : 0) * cW;
            ctx.strokeStyle = "#1e2228";
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(lx2, cY);
            ctx.lineTo(lx2, cY + cH);
            ctx.stroke();
            ctx.fillStyle = "#606878";
            ctx.font = "8px Arial";
            ctx.fillText(i.toString(), lx2, cY + cH + 3);
        }

        const pts = [], den = sigmas.length > 1 ? sigmas.length - 1 : 1;
        for (let i = 0; i < sigmas.length; i++) {
            const px = cX + (i / den) * cW;
            const py = cY + cH - Math.max(0, Math.min(1, sigmas[i])) * cH;
            pts.push({ x: px, y: py, val: sigmas[i], idx: i });
        }
        sd.points = pts;

        ctx.save();
        ctx.strokeStyle = themeColor.replace(")", ",0.12)").replace("rgb", "rgba");
        ctx.lineWidth = 6;
        ctx.lineJoin = "round";
        ctx.beginPath();
        pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
        ctx.stroke();
        ctx.restore();

        ctx.strokeStyle = themeColor;
        ctx.lineWidth = 1.5;
        ctx.lineJoin = "round";
        ctx.beginPath();
        pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
        ctx.stroke();

        if (pts.length < 120) {
            const br = pts.length < 30 ? 3 : pts.length < 60 ? 2 : 1.5;
            const weights = sd.weights || [];
            pts.forEach((p, i) => {
                if (i !== activeIdx) {
                    let pointColor = themeColor;
                    let pointRadius = br;
                    if (weights.length > 0 && weights[i] > 0.01) {
                        const weight = weights[i];
                        const alpha = 0.3 + weight * 0.7;
                        pointColor = themeColor.replace("rgb", "rgba").replace(")", `,${alpha})`);
                        pointRadius = br + weight * 2;
                    }
                    ctx.fillStyle = pointColor;
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, pointRadius, 0, Math.PI * 2);
                    ctx.fill();
                }
            });
        }

        if (activeIdx !== -1 && pts[activeIdx]) {
            const p = pts[activeIdx], drag = sd.dragIndex !== -1;

            ctx.save();
            ctx.strokeStyle = drag ? "rgba(255,180,50,0.6)" : "rgba(255,200,80,0.4)";
            ctx.lineWidth = 0.8;
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.moveTo(cX, p.y);
            ctx.lineTo(cX + cW, p.y);
            ctx.moveTo(p.x, cY);
            ctx.lineTo(p.x, cY + cH);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.restore();

            const ls = "Step: " + p.idx, lv = "Val: " + p.val.toFixed(4);
            ctx.font = "bold 10px Arial";
            const tw3 = Math.max(ctx.measureText(ls).width, ctx.measureText(lv).width);
            let lx3 = p.x + 10;
            if (lx3 + tw3 + 10 > cX + cW) lx3 = p.x - tw3 - 16;
            let ly3 = p.y - 34;
            if (ly3 < cY + 2) ly3 = p.y + 14;

            ctx.fillStyle = "rgba(16,18,28,0.92)";
            rr(ctx, lx3 - 4, ly3 - 4, tw3 + 12, 28, 4);
            ctx.fill();
            ctx.strokeStyle = drag ? "rgba(255,180,50,0.4)" : "rgba(255,200,80,0.3)";
            ctx.lineWidth = 0.5;
            ctx.stroke();

            ctx.fillStyle = "#ffc850";
            ctx.textAlign = "left";
            ctx.textBaseline = "top";
            ctx.fillText(ls, lx3, ly3);
            ctx.fillText(lv, lx3, ly3 + 12);

            ctx.beginPath();
            ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
            ctx.fillStyle = drag ? "rgba(255,180,50,0.25)" : themeColor.replace(")", ",0.2)").replace("rgb", "rgba");
            ctx.fill();
            ctx.beginPath();
            ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
            ctx.fillStyle = drag ? "#ffb432" : "#fff";
            ctx.fill();
            ctx.strokeStyle = drag ? "#ffb432" : themeColor;
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }

        ctx.save();
        ctx.font = "bold 9px 'Segoe UI',sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        if (isCtrlMode) {
            ctx.fillStyle = "rgba(100,180,255,0.8)";
            ctx.fillText("✏ 推挤模式 [松开Ctrl=软笔刷]", cX + 6, cY + 4);
        } else {
            const radius = getRadiusForTarget(target);
            const strength = getStrengthForTarget(target);
            ctx.fillStyle = "rgba(255,200,80,0.8)";
            // 显示强度信息
            ctx.fillText(`🖌 软笔刷 R:${radius.toFixed(1)} S:${(strength*100).toFixed(0)}% [Ctrl=推挤]`, cX + 6, cY + 4);
        }
        ctx.restore();

        ctx.textAlign = "left";
        ctx.textBaseline = "alphabetic";
    }

    // 新的组合控件绘制函数 (半径 + 强度)
    function drawBrushControls(ctx, target, x, y, w, h, themeColor) {
        // 背景
        ctx.fillStyle = "#14181e";
        rr(ctx, x, y, w, h, 3);
        ctx.fill();
        ctx.strokeStyle = "#2a2e38";
        ctx.lineWidth = 0.5;
        ctx.stroke();

        const rowH = 20;
        const gap = 4;
        const labelW = 58;
        
        // --- 第一行：半径 ---
        let ry = y + (h/2 - rowH) - 1; // 稍微偏上
        const radius = getRadiusForTarget(target);
        const minR = 0.1, maxR = 20.0;
        const rRatio = (radius - minR) / (maxR - minR);

        ctx.font = "9px 'Microsoft YaHei',sans-serif";
        ctx.fillStyle = "#a0aab8";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText("影响半径:", x + 6, ry + rowH / 2);

        let sliderX = x + labelW;
        let sliderW = w - labelW - 40;
        let sliderY = ry + (rowH - 6) / 2;

        ctx.fillStyle = "#222630";
        rr(ctx, sliderX, sliderY, sliderW, 6, 3);
        ctx.fill();

        let fillW = Math.max(4, sliderW * rRatio);
        ctx.fillStyle = themeColor.replace("rgb", "rgba").replace(")", ",0.6)");
        rr(ctx, sliderX, sliderY, fillW, 6, 3);
        ctx.fill();

        let handleX = sliderX + sliderW * rRatio;
        const handleR = 6;
        let isHoverR = node._hover === "_radius_slider_" + target;

        ctx.beginPath();
        ctx.arc(handleX, ry + rowH / 2, handleR, 0, Math.PI * 2);
        ctx.fillStyle = isHoverR ? "#fff" : themeColor;
        ctx.fill();
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.font = "bold 10px 'Segoe UI',sans-serif";
        ctx.fillStyle = "#e0e8f0";
        ctx.textAlign = "left";
        ctx.fillText(radius.toFixed(1), x + w - 34, ry + rowH / 2);

        node._cr["_radius_slider_" + target] = {
            x: sliderX, y: ry, w: sliderW, h: rowH,
            k: "slider_radius", target,
            min: minR, max: maxR, sliderX, sliderW
        };

        // --- 第二行：强度 ---
        let sy = y + h/2 + 1; // 偏下
        const strength = getStrengthForTarget(target);
        const minS = 0.0, maxS = 1.0;
        const sRatio = (strength - minS) / (maxS - minS);

        ctx.font = "9px 'Microsoft YaHei',sans-serif";
        ctx.fillStyle = "#a0aab8";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText("影响强度:", x + 6, sy + rowH / 2);

        sliderY = sy + (rowH - 6) / 2;

        ctx.fillStyle = "#222630";
        rr(ctx, sliderX, sliderY, sliderW, 6, 3);
        ctx.fill();

        fillW = Math.max(4, sliderW * sRatio);
        // 强度条稍微暗一点
        ctx.fillStyle = themeColor.replace("rgb", "rgba").replace(")", ",0.4)"); 
        rr(ctx, sliderX, sliderY, fillW, 6, 3);
        ctx.fill();

        handleX = sliderX + sliderW * sRatio;
        let isHoverS = node._hover === "_strength_slider_" + target;

        ctx.beginPath();
        ctx.arc(handleX, sy + rowH / 2, handleR, 0, Math.PI * 2);
        ctx.fillStyle = isHoverS ? "#fff" : themeColor.replace("rgb", "rgba").replace(")", ",0.8)");
        ctx.fill();
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.font = "bold 10px 'Segoe UI',sans-serif";
        ctx.fillStyle = "#e0e8f0";
        ctx.textAlign = "left";
        ctx.fillText((strength * 100).toFixed(0) + "%", x + w - 34, sy + rowH / 2);

        node._cr["_strength_slider_" + target] = {
            x: sliderX, y: sy, w: sliderW, h: rowH,
            k: "slider_strength", target,
            min: minS, max: maxS, sliderX, sliderW
        };
    }

    function drawSigmaTextBar(ctx, target, x, y, w, themeColor) {
        const barH = SIGMA_TEXT_H, wgt = fw(target);
        const hov = node._hover === target + "_text";

        ctx.fillStyle = hov ? "#1c2030" : "#14181e";
        rr(ctx, x, y, w, barH, 3);
        ctx.fill();
        ctx.strokeStyle = "#2a2e38";
        ctx.lineWidth = 0.5;
        ctx.stroke();

        ctx.fillStyle = hov ? themeColor : "#666";
        ctx.font = "11px sans-serif";
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillText("✏", x + w - 8, y + barH / 2);
        ctx.textAlign = "left";

        ctx.font = "bold 10px 'Microsoft YaHei',sans-serif";
        ctx.fillStyle = themeColor;
        ctx.textBaseline = "top";
        ctx.fillText(target === "曲线_1" ? "σ₁ " : "σ₂ ", x + 8, y + 5);

        const val = wgt ? String(wgt.value || "") : "";
        const tX = x + 26, tMW = w - 42, lH = 13, mL = 3;

        ctx.save();
        ctx.beginPath();
        ctx.rect(tX - 2, y + 2, tMW + 4, barH - 4);
        ctx.clip();
        ctx.font = "9px 'Consolas',monospace";
        ctx.textBaseline = "top";

        if (!val) {
            ctx.fillStyle = "#555";
            ctx.fillText("点击编辑 sigma 值…", tX, y + 18);
        } else {
            ctx.fillStyle = "#8aaccc";
            const cW2 = 5.4, mC = Math.floor(tMW / cW2);
            let rem = val;
            for (let l = 0; l < mL && rem.length > 0; l++) {
                let ch = rem.substring(0, mC);
                rem = rem.substring(mC);
                if (l === mL - 1 && rem.length > 0) ch = ch.substring(0, Math.max(0, ch.length - 1)) + "…";
                ctx.fillText(ch, tX, y + 5 + l * lH);
            }
        }
        ctx.restore();

        ctx.textBaseline = "middle";
        ctx.textAlign = "left";

        node._cr[target + "_text"] = { x, y, w, h: barH, vx: x, vw: w, k: "sigma_text", target };
        return barH;
    }

    function drawPreviewArea(ctx, x, y, w, h) {
        const halfW = (w - PREVIEW_GAP) / 2;
        const HEADER_H = 24;
        const INFO_H = 16;
        const PBAR_H = 14;
        const PBAR_PAD = 6;
        const IMG_TOP_PAD = 2;
        const BOTTOM_PAD = 4;

        const stages = [
            { key: "stage1", label: "一阶段预览", ox: x, col: C.s1 },
            { key: "stage2", label: "二阶段预览", ox: x + halfW + PREVIEW_GAP, col: C.s2 }
        ];

        for (const st of stages) {
            const ox = st.ox;
            const mt = node._previewMeta?.[st.key];
            const img = node._previewImages?.[st.key];

            ctx.fillStyle = "#0c0e14";
            rr(ctx, ox, y, halfW, h, 5);
            ctx.fill();
            ctx.strokeStyle = st.col.accent + "40";
            ctx.lineWidth = 1;
            ctx.stroke();

            ctx.fillStyle = st.col.hdr;
            rr(ctx, ox, y, halfW, HEADER_H, 5);
            ctx.fill();

            ctx.font = "bold 10px 'Microsoft YaHei',sans-serif";
            ctx.fillStyle = st.col.hdrTx;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";

            let lbl = st.label;
            if (mt && !mt.final && mt.step !== undefined) {
                lbl += " " + (mt.step + 1) + "/" + mt.total;
            } else if (mt && mt.final) {
                lbl += " ✓";
            }
            ctx.fillText(lbl, ox + halfW / 2, y + HEADER_H / 2);

            const imgAreaY = y + HEADER_H + IMG_TOP_PAD;
            const imgAreaH = h - HEADER_H - IMG_TOP_PAD - INFO_H - PBAR_H - BOTTOM_PAD;
            const imgAreaW = halfW - PBAR_PAD * 2;

            if (img) {
                ctx.save();
                ctx.imageSmoothingEnabled = true;
                ctx.imageSmoothingQuality = "high";

                const sc = Math.min(imgAreaW / img.naturalWidth, imgAreaH / img.naturalHeight);
                const dw = img.naturalWidth * sc;
                const dh = img.naturalHeight * sc;
                const dx = ox + (halfW - dw) / 2;
                const dy = imgAreaY + (imgAreaH - dh) / 2;

                ctx.drawImage(img, dx, dy, dw, dh);
                ctx.restore();
            } else {
                ctx.fillStyle = "#333";
                ctx.font = "11px 'Microsoft YaHei',sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText("等待执行…", ox + halfW / 2, imgAreaY + imgAreaH / 2);
            }

            const infoY = y + h - INFO_H - PBAR_H - BOTTOM_PAD;
            ctx.font = "9px 'Consolas','Segoe UI',monospace";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";

            if (img) {
                let dimText = "";
                if (mt && mt.lat_w && mt.lat_h) {
                    dimText = "Latent " + mt.lat_w + "×" + mt.lat_h + " → " + (mt.lat_w * 8) + "×" + (mt.lat_h * 8) + "px";
                } else if (mt && !mt.final) {
                    dimText = "Latent " + img.naturalWidth + "×" + img.naturalHeight;
                } else {
                    dimText = img.naturalWidth + "×" + img.naturalHeight + "px";
                }
                ctx.fillStyle = "#607888";
                ctx.fillText(dimText, ox + halfW / 2, infoY + INFO_H / 2);
            }

            const pbarY = y + h - PBAR_H - BOTTOM_PAD;
            const pbarX = ox + PBAR_PAD;
            const pbarW = halfW - PBAR_PAD * 2;

            ctx.fillStyle = "#181c24";
            rr(ctx, pbarX, pbarY, pbarW, PBAR_H, 4);
            ctx.fill();
            ctx.strokeStyle = "#2a3040";
            ctx.lineWidth = 0.5;
            ctx.stroke();

            if (mt && mt.total > 0) {
                let progress = 0;
                if (mt.final) {
                    progress = 1.0;
                } else if (mt.step !== undefined) {
                    progress = Math.min(1.0, (mt.step + 1) / mt.total);
                }
                if (progress > 0) {
                    const fillW = Math.max(6, Math.round(pbarW * progress));
                    ctx.save();
                    rr(ctx, pbarX, pbarY, fillW, PBAR_H, 4);
                    ctx.clip();
                    const grad = ctx.createLinearGradient(pbarX, pbarY, pbarX + fillW, pbarY);
                    grad.addColorStop(0, st.col.accent + "90");
                    grad.addColorStop(1, st.col.accent + "d0");
                    ctx.fillStyle = grad;
                    ctx.fillRect(pbarX, pbarY, fillW, PBAR_H);
                    ctx.restore();
                }

                ctx.font = "bold 9px 'Segoe UI',Arial,sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                if (mt.final) {
                    ctx.fillStyle = st.col.hdrTx;
                    ctx.fillText("✓ 完成", pbarX + pbarW / 2, pbarY + PBAR_H / 2);
                } else {
                    const pct = Math.round(progress * 100);
                    ctx.fillStyle = "#c0c8d8";
                    ctx.fillText(pct + "% (" + (mt.step + 1) + "/" + mt.total + ")", pbarX + pbarW / 2, pbarY + PBAR_H / 2);
                }
            } else {
                ctx.font = "8px 'Segoe UI',sans-serif";
                ctx.fillStyle = "#404858";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText("等待中", pbarX + pbarW / 2, pbarY + PBAR_H / 2);
            }
        }

        ctx.textAlign = "left";
        ctx.textBaseline = "alphabetic";
    }

    function sigmaHitTest(target, pos) {
        const sd = node._sigmaData[target];
        if (!sd?.chartLayout) return null;
        const L = sd.chartLayout, mx = pos[0], my = pos[1];
        if (mx < L.x - 5 || mx > L.x + L.w + 5 || my < L.y || my > L.y + L.h + 15) return null;
        if (!sd.points?.length) return null;

        if (sd.points.length === 1) {
            return {
                inChart: true,
                idx: Math.hypot(mx - sd.points[0].x, my - sd.points[0].y) < 25 ? 0 : -1
            };
        }

        let minD = Infinity, close = -1;
        const xR = Math.max(8, L.w / (sd.points.length - 1) * 0.6);
        for (let i = 0; i < sd.points.length; i++) {
            const dx = Math.abs(mx - sd.points[i].x);
            if (dx < xR && dx < minD) { minD = dx; close = i; }
        }
        return { inChart: true, idx: close };
    }

    const origFG = node.onDrawForeground;
    node.onDrawForeground = function (ctx) {
        if (origFG) origFG.apply(this, arguments);
        try {
            hideAllWidgets();
            hideDom();
            node._cr = {};

            const W = this.size[0], mid = W / 2;
            const colW = (W - EDGE_PAD * 2 - CENTER_GAP) / 2;
            const lColX = EDGE_PAD, rColX = EDGE_PAD + colW + CENTER_GAP;
            const fullW = W - PAD * 2;

            ctx.save();
            ctx.textBaseline = "middle";
            let sy = 30, yL = sy, yR = sy;

            const LEFT_SECTIONS = [{
                title: "一阶段 · 构图", th: C.s1, rows: [
                    { n: "随机种_1", l: "种子", k: "seed" },
                    { n: "_sc1", l: "生成控制", k: "seedctrl" },
                    { n: "步数_1", l: "步数", k: "int" },
                    { n: "CFG_1", l: "引导系数", k: "float" },
                    { n: "采样器_1", l: "采样器", k: "combo" },
                    { n: "调度器_1", l: "调度器", k: "combo" },
                    { n: "降噪_1", l: "降噪强度", k: "float" },
                    { n: "添加噪波_1", l: "添加噪波", k: "combo" },
                    { n: "开始步数_1", l: "开始步数", k: "int" },
                    { n: "结束步数_1", l: "结束步数", k: "int" },
                    { n: "返回噪波_1", l: "返回噪波", k: "combo" }
                ]
            }];

            const SYNC_ROW = { n: "种子同步", l: "种子同步", k: "combo" };

            const RIGHT_SECTIONS = [
                {
                    title: "放大设置", th: C.up, rows: [
                        { n: "放大倍数", l: "放大倍数", k: "float" },
                        { n: "放大算法", l: "放大算法", k: "combo" },
                        { n: "放大模式", l: "放大模式", k: "combo" }
                    ]
                },
                {
                    title: "二阶段 · 精修", th: C.s2, rows: [
                        { n: "随机种_2", l: "种子", k: "seed" },
                        { n: "_sc2", l: "生成控制", k: "seedctrl" },
                        { n: "步数_2", l: "步数", k: "int" },
                        { n: "CFG_2", l: "引导系数", k: "float" },
                        { n: "采样器_2", l: "采样器", k: "combo" },
                        { n: "调度器_2", l: "调度器", k: "combo" },
                        { n: "降噪_2", l: "降噪强度", k: "float" }
                    ]
                }
            ];

            for (const s of LEFT_SECTIONS) { yL = drawSection(ctx, s, lColX, yL, colW); yL += GAP; }
            drawRow(ctx, SYNC_ROW, lColX, yL, colW, C.sync);
            yL += ROW + GAP + SEC;

            for (const s of RIGHT_SECTIONS) { yR = drawSection(ctx, s, rColX, yR, colW); yR += GAP; }

            const paramBottom = Math.max(yL, yR);

            ctx.strokeStyle = C.sep;
            ctx.lineWidth = 1;
            ctx.setLineDash([2, 3]);
            ctx.beginPath();
            ctx.moveTo(mid, sy);
            ctx.lineTo(mid, paramBottom);
            ctx.stroke();
            ctx.setLineDash([]);

            let statusY = paramBottom + 4;
            const statusMsg = node._statusMsg;
            if (statusMsg && statusMsg.text && (Date.now() - statusMsg.time < 15000)) {
                const isWarn = statusMsg.text.includes("⚠");
                ctx.fillStyle = isWarn ? "#2e2010" : "#102e18";
                rr(ctx, lColX, statusY, colW * 2 + CENTER_GAP, ROW, 3);
                ctx.fill();
                ctx.strokeStyle = isWarn ? "#c8a04060" : "#40c08060";
                ctx.lineWidth = 0.5;
                ctx.stroke();
                ctx.font = "11px 'Microsoft YaHei',sans-serif";
                ctx.fillStyle = isWarn ? "#e8c060" : "#60e888";
                ctx.textAlign = "left";
                ctx.fillText(statusMsg.text, lColX + 8, statusY + ROW / 2);
            }
            statusY += ROW + GAP + 2;

            const presetY = yR;
            const btnW = (colW - 4) / 3;

            const saveHov = node._hover === "_save_btn";
            ctx.fillStyle = saveHov ? PC.btnSaveH : PC.btnSave;
            rr(ctx, rColX, presetY, btnW, ROW, 3);
            ctx.fill();
            ctx.font = "12px 'Microsoft YaHei',sans-serif";
            ctx.fillStyle = PC.btnSaveTx;
            ctx.textAlign = "center";
            ctx.fillText("💾 保存", rColX + btnW / 2, presetY + ROW / 2);
            node._cr["_save_btn"] = { x: rColX, y: presetY, w: btnW, h: ROW, k: "save_btn" };

            const loadHov = node._hover === "_preset_sel";
            ctx.fillStyle = loadHov ? PC.rowH : PC.row;
            rr(ctx, rColX + btnW + 2, presetY, btnW, ROW, 3);
            ctx.fill();
            let pl_short = node.properties._selectedPreset || "📁 加载...";
            if (ctx.measureText(pl_short).width > btnW - 10) pl_short = "📁 预设";
            ctx.font = "12px 'Microsoft YaHei',sans-serif";
            ctx.fillStyle = C.value;
            ctx.fillText(pl_short, rColX + btnW + 2 + btnW / 2, presetY + ROW / 2);
            node._cr["_preset_sel"] = { x: rColX + btnW + 2, y: presetY, w: btnW, h: ROW, k: "preset" };

            const delHov = node._hover === "_del_btn";
            ctx.fillStyle = delHov ? PC.btnDelH : PC.btnDel;
            rr(ctx, rColX + (btnW + 2) * 2, presetY, btnW, ROW, 3);
            ctx.fill();
            ctx.fillStyle = PC.btnDelTx;
            ctx.fillText("🗑 删除", rColX + (btnW + 2) * 2 + btnW / 2, presetY + ROW / 2);
            node._cr["_del_btn"] = { x: rColX + (btnW + 2) * 2, y: presetY, w: btnW, h: ROW, k: "del_btn" };

            let sigmaY = Math.max(statusY, presetY + ROW + SEC);
            ctx.textAlign = "left";

            const isCtrlPressed = node._ctrlPressed || false;

            ctx.fillStyle = C.s1.hdr;
            rr(ctx, PAD, sigmaY, fullW, HDR, 4);
            ctx.fill();
            ctx.fillStyle = C.s1.accent;
            rr(ctx, PAD, sigmaY, 3, HDR, 2);
            ctx.fill();
            ctx.font = "bold 11px 'Microsoft YaHei','PingFang SC',sans-serif";
            ctx.fillStyle = C.s1.hdrTx;
            ctx.fillText("一阶段 · Sigma 曲线（工作流第一次运行：点运行-到本节点-中断-改调度器；鼠标左键按下再按ctrl调单点）", PAD + 10, sigmaY + HDR / 2);
            sigmaY += HDR + 3;

            sigmaY += drawSigmaTextBar(ctx, "曲线_1", PAD, sigmaY, fullW, "#38c888") + 3;
            // 【UI修改】绘制组合控件：半径 + 强度
            drawBrushControls(ctx, "曲线_1", PAD, sigmaY, fullW, CONTROLS_H, "rgb(56,200,136)");
            sigmaY += CONTROLS_H + 6;

            const extraSliders = (CONTROLS_H + 6) * 2;
            const chartH = Math.max(
                MIN_CHART_H,
                (this.size[1] - (sigmaY + 8 + HDR + 3 + SIGMA_TEXT_H + 3 + extraSliders + PREVIEW_H + 10)) / 2
            );

            drawSigmaChart(ctx, "曲线_1", PAD, sigmaY, fullW, chartH, "rgb(56,200,136)", isCtrlPressed);
            node._cr["_chart_1"] = { x: PAD, y: sigmaY, w: fullW, h: chartH, k: "chart", target: "曲线_1" };
            sigmaY += chartH + 8;

            ctx.fillStyle = C.s2.hdr;
            rr(ctx, PAD, sigmaY, fullW, HDR, 4);
            ctx.fill();
            ctx.fillStyle = C.s2.accent;
            rr(ctx, PAD, sigmaY, 3, HDR, 2);
            ctx.fill();
            ctx.font = "bold 11px 'Microsoft YaHei','PingFang SC',sans-serif";
            ctx.fillStyle = C.s2.hdrTx;
            ctx.textBaseline = "middle";
            ctx.fillText("二阶段 · Sigma 曲线（建议 0 步设为 0.7-0.9）", PAD + 10, sigmaY + HDR / 2);
            sigmaY += HDR + 3;

            sigmaY += drawSigmaTextBar(ctx, "曲线_2", PAD, sigmaY, fullW, "#4888e0") + 3;
            // 【UI修改】绘制组合控件
            drawBrushControls(ctx, "曲线_2", PAD, sigmaY, fullW, CONTROLS_H, "rgb(72,136,224)");
            sigmaY += CONTROLS_H + 6;

            drawSigmaChart(ctx, "曲线_2", PAD, sigmaY, fullW, chartH, "rgb(72,136,224)", isCtrlPressed);
            node._cr["_chart_2"] = { x: PAD, y: sigmaY, w: fullW, h: chartH, k: "chart", target: "曲线_2" };
            sigmaY += chartH + 5;

            drawPreviewArea(ctx, PAD, sigmaY, fullW, PREVIEW_H);
            node._cr["_preview_area"] = { x: PAD, y: sigmaY, w: fullW, h: PREVIEW_H, k: "preview" };

            ctx.restore();
        } catch (e) {
            console.error("Visual Sampler Draw Error", e);
        }
    };

    const hit = pos => {
        let best = null;
        for (const n in node._cr) {
            const r = node._cr[n];
            if (pos[0] >= r.x && pos[0] <= r.x + r.w && pos[1] >= r.y && pos[1] <= r.y + r.h) {
                if (r.k === "dice" || r.k === "save_btn" || r.k === "del_btn" || r.k === "preset" || r.k === "step_decr" || r.k === "step_incr") return n;
                best = n;
            }
        }
        return best;
    };

    node.onMouseDown = function (e, pos) {
        if (e && typeof e.ctrlKey !== 'undefined') node._ctrlPressed = e.ctrlKey;
        
        const name = hit(pos);
        if (name) {
            const cr = node._cr[name];

            if (cr?.k === "sigma_text") { showSigmaTextInput(cr, cr.target); return true; }

            if (cr?.k === "step_decr" || cr?.k === "step_incr") {
                const w = fw(cr.name);
                if (w) {
                    const m = wmeta(w);
                    let delta = 0;
                    
                    if (ONE_DECIMAL.includes(cr.name)) {
                        delta = 0.1;
                    } else if (cr.type === "int" || cr.type === "seed") {
                        delta = 1;
                    } else {
                        delta = 0.01;
                    }

                    let nv = w.value + (cr.k === "step_incr" ? delta : -delta);
                    nv = Math.max(m.min, Math.min(m.max, nv));
                    
                    if (cr.type === "int" || cr.type === "seed") nv = Math.round(nv);
                    else nv = parseFloat(nv.toFixed(4));

                    w.value = nv;
                    w.callback?.(nv);
                    node.setDirtyCanvas(true, true);
                }
                return true; 
            }

            if (cr?.k === "dice") {
                const w = fw(cr.seedName);
                if (w) {
                    w.value = Math.floor(Math.random() * 999999999999999);
                    const syncW = fw("种子同步");
                    if (syncW && syncW.value === "enable" && cr.seedName === "随机种_1") {
                        const w2 = fw("随机种_2");
                        if (w2) w2.value = w.value;
                    }
                    node.setDirtyCanvas(true, true);
                }
                return true;
            }

            // 【滑动条点击逻辑更新】区分 Radius 和 Strength
            if (cr?.k === "slider_radius" || cr?.k === "slider_strength") {
                node._draggingSlider = cr; // 记录正在拖动的滑块对象
                updateSliderByMouse(pos[0], cr);
                node.setDirtyCanvas(true, true);
                return true;
            }
        }

        for (const target of ["曲线_1", "曲线_2"]) {
            const result = sigmaHitTest(target, pos);
            if (result?.inChart && result.idx >= 0) {
                const sd = node._sigmaData[target];
                sd.dragIndex = result.idx;
                sd.dragStartPos = { x: pos[0], y: pos[1] };
                sd.isDragging = false;
                node._activeSigma = target;

                pushUndo(target);

                const isCtrl = e.ctrlKey || node._ctrlPressed;

                if (isCtrl) {
                    sd.weights = new Array(sd.sigmas.length).fill(0);
                    sd.weights[sd.dragIndex] = 1;
                } else {
                    const radius = getRadiusForTarget(target);
                    sd.weights = calculateWeights(sd.sigmas, sd.dragIndex, radius);
                }

                node.setDirtyCanvas(true, true);
                
                if (e.ctrlKey) {
                    e.preventDefault(); 
                    e.stopPropagation();
                    if (e.stopImmediatePropagation) e.stopImmediatePropagation();
                }
                return true;
            }
        }
        
        if (!name) return false;
        const cr = node._cr[name];
        if (!cr) return false;
        if (cr.k === "chart" || cr.k === "preview") return false;

        const syncW = fw("种子同步");
        const isSync = syncW && syncW.value === "enable";
        if (isSync && (name === "随机种_2" || name === "_sc2")) return true;

        if (cr.k === "seedctrl") {
            new LiteGraph.ContextMenu(
                SC_LIST.map(m => ({
                    content: SC_ICON[m] + " " + SC_LBL[m],
                    callback: () => {
                        node.properties[name] = m;
                        if (m === "randomize") {
                            const seedName = name === "_sc1" ? "随机种_1" : "随机种_2";
                            const w = fw(seedName);
                            if (w && w.value === 0) w.value = Math.floor(Math.random() * 999999999999999);
                        }
                        node.setDirtyCanvas(true, true);
                    }
                })),
                { event: e }
            );
            return true;
        }

        if (cr.k === "preset") {
            loadPresets(presets => {
                const names = Object.keys(presets);
                if (!names.length) {
                    new LiteGraph.ContextMenu([{ content: "(暂无预设)" }], { event: e });
                    return;
                }
                new LiteGraph.ContextMenu(
                    names.map(pn => ({
                        content: pn,
                        callback: () => {
                            node.properties._selectedPreset = pn;
                            const config = presets[pn];
                            node._suppressRegen = true;

                            const ck = ["曲线_1", "曲线_2"];
                            for (const [k, v] of Object.entries(config)) {
                                if (ck.includes(k)) continue;
                                if (k === "_softSelectionRadius1" || k === "_softSelectionRadius2") continue;
                                const w = fw(k);
                                if (w) { w.value = v; w.callback?.(v); }
                            }
                            node._cancelPendingGen();

                            for (const c of ck) {
                                if (config[c] !== undefined) {
                                    const w = fw(c);
                                    if (w) w.value = config[c];
                                    node.refreshCurve(c, config[c]);
                                }
                            }

                            if (config._softSelectionRadius1 != null)
                                node.properties._softSelectionRadius1 = config._softSelectionRadius1;
                            if (config._softSelectionRadius2 != null)
                                node.properties._softSelectionRadius2 = config._softSelectionRadius2;
                            if (config._softSelectionStrength1 != null)
                                node.properties._softSelectionStrength1 = config._softSelectionStrength1;
                            if (config._softSelectionStrength2 != null)
                                node.properties._softSelectionStrength2 = config._softSelectionStrength2;

                            node._suppressRegen = false;
                            node.setDirtyCanvas(true, true);
                        }
                    })),
                    { event: e }
                );
            });
            return true;
        }

        if (cr.k === "save_btn") { showSavePresetDialog(cr); return true; }

        if (cr.k === "del_btn") {
            loadPresets(presets => {
                const names = Object.keys(presets);
                if (!names.length) {
                    new LiteGraph.ContextMenu([{ content: "(暂无)" }], { event: e });
                    return;
                }
                new LiteGraph.ContextMenu(
                    names.map(pn => ({
                        content: "🗑 " + pn,
                        callback: () => {
                            setTimeout(() => {
                                new LiteGraph.ContextMenu(
                                    [
                                        { content: `确认删除「${pn}」？` },
                                        null,
                                        {
                                            content: "✅确认",
                                            callback: () => {
                                                deletePreset(pn);
                                                if (node.properties._selectedPreset === pn)
                                                    node.properties._selectedPreset = "";
                                                node.setDirtyCanvas(true, true);
                                            }
                                        },
                                        { content: "❌取消", callback: () => {} }
                                    ],
                                    { event: e, title: "删除确认" }
                                );
                            }, 60);
                        }
                    })),
                    { event: e }
                );
            });
            return true;
        }

        if (cr.k === "combo") {
            const w = fw(name);
            if (!w) return false;
            const vals = w.options?.values || [];
            if (!vals.length) return false;
            new LiteGraph.ContextMenu(
                vals.map(v => ({
                    content: String(v),
                    callback: () => {
                        w.value = v;
                        w.callback?.(v);
                        node.setDirtyCanvas(true, true);
                    }
                })),
                { event: e }
            );
            return true;
        }

        if (cr.k === "seed") {
            const w = fw(name);
            if (!w) return false;
            showInput(cr, w.value, val => {
                const n2 = parseInt(val);
                if (!isNaN(n2) && n2 >= 0) {
                    w.value = n2;
                    w.callback?.(n2);
                    node.setDirtyCanvas(true, true);
                }
            });
            return true;
        }

        if (cr.k === "int" || cr.k === "float") {
            const w = fw(name);
            if (!w) return false;
            showInput(cr, w.value, val => {
                const n2 = parseFloat(val);
                if (!isNaN(n2)) {
                    const m = wmeta(w);
                    let nv = Math.max(m.min, Math.min(m.max, n2));
                    if (cr.k === "int") nv = Math.round(nv);
                    w.value = nv;
                    w.callback?.(nv);
                    node.setDirtyCanvas(true, true);
                }
            });
            return true;
        }

        return false;
    };

    node.onMouseMove = function (e, pos) {
        if (typeof e.ctrlKey !== 'undefined') node._ctrlPressed = e.ctrlKey;

        const name = hit(pos);
        if (node._hover !== name) { node._hover = name; node.setDirtyCanvas(true, false); }

        if (node._draggingSlider && (e.buttons & 1)) {
            updateSliderByMouse(pos[0], node._draggingSlider);
            node.setDirtyCanvas(true, true);
            return true;
        }

        // 【核心修复】防止粘连：如果鼠标松开且没有按Ctrl，强制重置 activeSigma
        if (e.buttons === 0 && !e.ctrlKey) {
             if (node._activeSigma) {
                 node._activeSigma = null;
                 node.setDirtyCanvas(true, false);
             }
        }

        if (node._activeSigma) {
            const sd = node._sigmaData[node._activeSigma];
            if (sd?.dragIndex !== -1 && sd.chartLayout) {
                if (!sd.isDragging && sd.dragStartPos) {
                    if (Math.abs(pos[0] - sd.dragStartPos.x) > 3 || Math.abs(pos[1] - sd.dragStartPos.y) > 3)
                        sd.isDragging = true;
                }
                if (sd.isDragging) {
                    const L = sd.chartLayout;
                    const cy = Math.max(L.y, Math.min(L.y + L.h, pos[1]));
                    const newVal = Math.max(0, Math.min(1, (L.y + L.h - cy) / L.h));

                    const sigmas = sd.sigmas;
                    const idx = sd.dragIndex;
                    
                    // 【核心修复】检测 Ctrl 单点调节
                    const isCtrl = e.ctrlKey || node._ctrlPressed;
                    const anchors = { first: 0, last: sigmas.length - 1 };
                    const isAnchor = (idx === anchors.first || idx === anchors.last);

                    if (isCtrl || isAnchor) {
                        applyPushConstraint(sigmas, idx, newVal, anchors);
                        sd.weights = new Array(sigmas.length).fill(0);
                        sd.weights[idx] = 1; 
                    } else {
                        const radius = getRadiusForTarget(node._activeSigma);
                        const strength = getStrengthForTarget(node._activeSigma);
                        const updated = applySoftDrag(sigmas, idx, newVal, radius, strength, anchors);
                        sd.sigmas = updated;
                        sd.weights = calculateWeights(updated, idx, radius);
                    }

                    if (node._activeSigma === "曲线_2") {
                        const hint = getStage2Guidance(sd.sigmas);
                        if (hint) {
                            node._statusMsg = { text: hint, time: Date.now() };
                        }
                    }

                    node.setDirtyCanvas(true, true);
                    return true;
                }
            }
        }
        
        for (const t of ["曲线_1", "曲线_2"]) {
            const r = sigmaHitTest(t, pos);
            if (r?.inChart) {
                const sd = node._sigmaData[t];
                const nh = r.idx >= 0 ? r.idx : -1;
                if (sd.hoverIndex !== nh) {
                    sd.hoverIndex = nh;
                    if (nh !== -1) {
                        const isCtrl = e.ctrlKey || node._ctrlPressed;
                        if (isCtrl) {
                            sd.weights = new Array(sd.sigmas.length).fill(0);
                            sd.weights[nh] = 1;
                        } else {
                            const radius = getRadiusForTarget(t);
                            sd.weights = calculateWeights(sd.sigmas, nh, radius);
                        }
                    } else {
                        sd.weights = null;
                    }
                    node.setDirtyCanvas(true, true);
                }
            }
        }
        return false;
    };

    node.onMouseUp = function (e, pos) {
        if (e && typeof e.ctrlKey !== 'undefined') node._ctrlPressed = e.ctrlKey;

        if (node._draggingSlider) {
            node._draggingSlider = null;
            node.setDirtyCanvas(true, true);
            return true;
        }

        if (node._activeSigma) {
            const sd = node._sigmaData[node._activeSigma];
            if (sd?.isDragging) {
                node._finishSigmaDrag(node._activeSigma);
                const issues = validateSigmaSequence(sd.sigmas);
                if (issues.length) {
                    node._statusMsg = { text: "⚠ " + issues[0], time: Date.now() };
                }
            }
            if (sd) { sd.isDragging = false; sd.dragStartPos = null; sd.dragIndex = -1; sd.weights = null; }
            node._activeSigma = null;
            return true;
        }
        return false;
    };

    node.onMouseLeave = function () {
        node._hover = null;
        for (const t of ["曲线_1", "曲线_2"]) {
            node._sigmaData[t].hoverIndex = -1;
            node._sigmaData[t].weights = null;
        }
        node.setDirtyCanvas(true, false);
    };

    node.onMouseWheel = function (e, pos) {
        for (const target of ["曲线_1", "曲线_2"]) {
            const chartCr = node._cr["_chart_" + (target === "曲线_1" ? "1" : "2")];
            if (chartCr && pos[0] >= chartCr.x && pos[0] <= chartCr.x + chartCr.w
                && pos[1] >= chartCr.y && pos[1] <= chartCr.y + chartCr.h) {
                if (e.preventDefault) e.preventDefault();
                if (e.stopPropagation) e.stopPropagation();
                return true;
            }
        }

        const nm = hit(pos);
        if (!nm) return false;
        const cr = node._cr[nm], w = fw(nm);
        if (!w || !cr || (cr.k !== "int" && cr.k !== "float")) return false;

        const syncW = fw("种子同步");
        if (nm === "随机种_2" && syncW?.value === "enable") return true;

        const m = wmeta(w), dir = -Math.sign(e.deltaY);
        let nv = w.value + dir * m.step;
        nv = Math.max(m.min, Math.min(m.max, nv));
        if (cr.k === "float") nv = parseFloat(nv.toFixed(4));
        else nv = Math.round(nv);

        if (w.value !== nv) {
            w.value = nv;
            w.callback?.(nv);
            node.setDirtyCanvas(true, true);
        }
        return true;
    };

    node.onKeyDown = function(e) {
        if (e.key === "Control") {
            node._ctrlPressed = true;
            node.setDirtyCanvas(true, false);
        }

        if (e.ctrlKey && (e.key === 'z' || e.key === 'Z')) {
            const target = node._activeSigma || "曲线_1";
            if (popUndo(target)) {
                node._statusMsg = { text: "↩ 已撤销", time: Date.now() };
                if (e.preventDefault) e.preventDefault();
                return true;
            }
        }
        return false;
    };

    node.onKeyUp = function(e) {
        if (e.key === "Control") {
            node._ctrlPressed = false;
            node.setDirtyCanvas(true, false);
        }
        return false;
    };

    // 通用滑块更新函数（处理半径和强度）
    function updateSliderByMouse(mouseX, cr) {
        const minV = cr.min, maxV = cr.max;
        const t = (mouseX - cr.sliderX) / cr.sliderW;
        const ratio = Math.max(0, Math.min(1, t));
        const raw = minV + ratio * (maxV - minV);
        
        if (cr.k === "slider_radius") {
            const nv = setRadiusForTarget(cr.target, raw);
            node._statusMsg = { text: `🖌 ${cr.target} 半径: ${nv.toFixed(1)}`, time: Date.now() };
        } else if (cr.k === "slider_strength") {
            const nv = setStrengthForTarget(cr.target, raw);
            node._statusMsg = { text: `💪 ${cr.target} 强度: ${(nv*100).toFixed(0)}%`, time: Date.now() };
        }
    }

    const origMenu = node.getExtraMenuOptions;
    node.getExtraMenuOptions = function(_, options) {
        origMenu?.apply(this, arguments);

        options.unshift(
            {
                content: "设置软笔刷参数",
                has_submenu: true,
                callback: (v, opts, e2, menu) => {
                    const mkRadius = (target) => ({
                        content: `${target} 半径 (当前 ${getRadiusForTarget(target).toFixed(1)})`,
                        callback: () => {
                            const cur = getRadiusForTarget(target);
                            const val = prompt(`${target} 软笔刷半径 (0.1-20.0):`, cur);
                            if (val !== null) {
                                const n = parseFloat(val);
                                if (!isNaN(n)) {
                                    setRadiusForTarget(target, n);
                                    node.setDirtyCanvas(true, true);
                                }
                            }
                        }
                    });
                    const mkStrength = (target) => ({
                        content: `${target} 强度 (当前 ${(getStrengthForTarget(target)*100).toFixed(0)}%)`,
                        callback: () => {
                            const cur = getStrengthForTarget(target);
                            const val = prompt(`${target} 软笔刷强度 (0.0-1.0):`, cur);
                            if (val !== null) {
                                const n = parseFloat(val);
                                if (!isNaN(n)) {
                                    setStrengthForTarget(target, n);
                                    node.setDirtyCanvas(true, true);
                                }
                            }
                        }
                    });
                    new LiteGraph.ContextMenu(
                        [ mkRadius("曲线_1"), mkStrength("曲线_1"), mkRadius("曲线_2"), mkStrength("曲线_2") ],
                        { event: e2, parentMenu: menu }
                    );
                }
            },
            {
                content: "重置曲线为调度器默认",
                has_submenu: true,
                callback: (v, opts, e2, menu) => {
                    new LiteGraph.ContextMenu(
                        [
                            { content: "重置 曲线_1", callback: () => node._clearAndRegenerate("曲线_1") },
                            { content: "重置 曲线_2", callback: () => node._clearAndRegenerate("曲线_2") }
                        ],
                        { event: e2, parentMenu: menu }
                    );
                }
            },
            null
        );
    };

    const canvas = app.canvas?.canvas || document.querySelector("canvas");
    if (canvas) {
        const globalUp = () => {
            node._draggingSlider = null;
            if (node._activeSigma) {
                node._finishSigmaDrag(node._activeSigma);
                node._activeSigma = null;
            }
        };
        canvas.addEventListener("pointerup", globalUp);
        canvas.addEventListener("pointercancel", globalUp);

        const origRemoved = node.onRemoved;
        node.onRemoved = function () {
            canvas.removeEventListener("pointerup", globalUp);
            canvas.removeEventListener("pointercancel", globalUp);
            origRemoved?.apply(this, arguments);
        };
    }

    const origExec = node.onExecuted;
    node.onExecuted = function (output) {
        origExec?.apply(this, arguments);
        doPostRunSeed("_sc1", "随机种_1");
        const syncW = fw("种子同步");
        if (syncW?.value === "enable") {
            const w1 = fw("随机种_1"), w2 = fw("随机种_2");
            if (w1 && w2) w2.value = w1.value;
        } else {
            doPostRunSeed("_sc2", "随机种_2");
        }
        node.setDirtyCanvas(true, true);
    };

    function doPostRunSeed(prop, seedName) {
        const mode = node.properties[prop] || "fixed", w = fw(seedName);
        if (!w) return;
        if (mode === "increment") w.value += 1;
        if (mode === "decrement") w.value = Math.max(0, w.value - 1);
    }

    const origResize = node.onResize;
    node.onResize = function (sz) {
        sz[0] = Math.max(sz[0], 640);
        sz[1] = Math.max(sz[1], node._minNodeH || 1340);
        origResize?.apply(this, arguments);
        node.setDirtyCanvas(true, true);
    };
}