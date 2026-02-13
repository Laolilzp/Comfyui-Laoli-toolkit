import math
import torch
import numpy
import os
import json
import comfy.samplers
import comfy.model_sampling
import comfy.model_management
from server import PromptServer
from aiohttp import web


# 模块级缓存：存储每个节点执行时的真实 model_sampling
# key = node_id (str), value = model_sampling 对象
_cached_model_sampling = {}


def _build_sigma_table():
    """
    构建默认 sigma 查找表，模拟 ComfyUI ModelSamplingDiscrete 的行为。
    返回的 sigma_table 从小到大排列（index 0 = 最小 sigma，index 999 = 最大 sigma），
    与 ComfyUI 的 model_sampling.sigmas 顺序一致。
    """
    total_timesteps = 1000
    beta_start = 0.00085
    beta_end = 0.012
    betas = torch.linspace(beta_start**0.5, beta_end**0.5, total_timesteps, dtype=torch.float64)**2
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sigma_table = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
    return sigma_table.float()


def _preview_sigmas(scheduler, total_steps, steps, denoise, node_id=None):
    """
    根据调度器名称生成 [0,1] 范围的预览 sigma 曲线。

    策略：
    1. 优先使用缓存的真实 model_sampling（执行过一次后就有）
    2. 其次使用 ComfyUI ModelSamplingDiscrete 默认值
    3. 最后使用模拟兜底（用于第三方调度器或 ComfyUI 调用失败）
    """
    n = total_steps
    sigmas = None

    # ===== 第一优先：使用缓存的真实 model_sampling =====
    cached_ms = _cached_model_sampling.get(str(node_id)) if node_id else None
    if cached_ms is not None:
        try:
            print(f"[老李-后端] _preview_sigmas: 找到缓存 node_id={node_id}, model_sampling类型={type(cached_ms).__name__}")
            sigmas = comfy.samplers.calculate_sigmas(cached_ms, scheduler, n)
            print(f"[老李-后端] _preview_sigmas: 使用缓存的真实模型 ({scheduler}), "
                  f"长度={len(sigmas)}, 范围=[{sigmas[-1].item():.4f}, {sigmas[0].item():.4f}]")
        except Exception as e:
            import traceback
            print(f"[老李-后端] _preview_sigmas: 缓存模型计算失败 ({scheduler}): {e}")
            print(f"[老李-后端] 完整错误追踪:\n{traceback.format_exc()}")

    # ===== 第二优先：使用 ComfyUI 默认 ModelSamplingDiscrete =====
    if sigmas is None:
        try:
            ms = comfy.model_sampling.ModelSamplingDiscrete()
            sigmas = comfy.samplers.calculate_sigmas(ms, scheduler, n)
            print(f"[老李-后端] _preview_sigmas: ComfyUI 默认计算 ({scheduler}), "
                  f"长度={len(sigmas)}, 范围=[{sigmas[-1].item():.4f}, {sigmas[0].item():.4f}]")
        except Exception as e:
            print(f"[老李-后端] _preview_sigmas: ComfyUI 默认计算失败 ({scheduler}): {e}")

    # ===== 第三优先：模拟兜底 =====
    if sigmas is None:
        sigmas = _simulate_sigmas(scheduler, n)
        print(f"[老李-后端] _preview_sigmas: 使用模拟计算 ({scheduler}), "
              f"长度={len(sigmas)}, 范围=[{sigmas[-1].item():.4f}, {sigmas[0].item():.4f}]")

    # ===== 归一化到 [0,1] 并应用 denoise =====
    sig_max_val = sigmas.max()
    if sig_max_val > 0:
        sigmas = sigmas / sig_max_val

    if denoise < 1.0:
        sigmas = sigmas[-(steps + 1):]

    return sigmas.clamp(0.0, 1.0)


def _simulate_sigmas(scheduler, n):
    """
    不依赖 ComfyUI 的模拟计算，用于第三方调度器或 ComfyUI 不可用时。
    模拟 ModelSamplingDiscrete 的默认 sigma 表（SD1.5/SDXL linear beta schedule）。
    """
    sigma_table = _build_sigma_table()
    total_timesteps = len(sigma_table)
    sigma_max = float(sigma_table[-1])
    sigma_min = float(sigma_table[0])
    log_sigma_table = torch.log(torch.clamp(sigma_table, min=1e-10))

    # 模拟 ComfyUI 的 timestep() — log-space 插值
    def sigma_to_timestep(sigma):
        log_sigma = math.log(max(float(sigma), 1e-10))
        dists = log_sigma - log_sigma_table
        mask = (dists >= 0)
        if not mask.any():
            return 0.0
        low_idx = int(mask.float().cumsum(dim=0).argmax().item())
        low_idx = min(low_idx, total_timesteps - 2)
        high_idx = low_idx + 1
        low = float(log_sigma_table[low_idx])
        high = float(log_sigma_table[high_idx])
        w = (low - log_sigma) / (low - high) if low != high else 0.0
        w = max(0.0, min(1.0, w))
        return (1.0 - w) * low_idx + w * high_idx

    # 模拟 ComfyUI 的 sigma() — log-space 插值
    def timestep_to_sigma(t):
        t = max(0.0, min(float(total_timesteps - 1), float(t)))
        low_idx = int(math.floor(t))
        high_idx = min(low_idx + 1, total_timesteps - 1)
        w = t - low_idx
        log_sigma = (1.0 - w) * float(log_sigma_table[low_idx]) + w * float(log_sigma_table[high_idx])
        return math.exp(log_sigma)

    # ===== 各调度器的模拟实现 =====
    if scheduler == "simple":
        sigs = []
        ss = total_timesteps / n
        for x in range(n):
            sigs.append(float(sigma_table[-(1 + int(x * ss))]))
        sigs.append(0.0)
        return torch.FloatTensor(sigs)

    elif scheduler == "normal":
        start_t = sigma_to_timestep(sigma_max)
        end_t = sigma_to_timestep(sigma_min)
        if abs(timestep_to_sigma(end_t)) < 0.00001:
            timesteps = torch.linspace(start_t, end_t, n + 1)
            sigs = [timestep_to_sigma(float(t)) for t in timesteps]
        else:
            timesteps = torch.linspace(start_t, end_t, n)
            sigs = [timestep_to_sigma(float(t)) for t in timesteps]
            sigs.append(0.0)
        return torch.FloatTensor(sigs)

    elif scheduler == "sgm_uniform":
        start_t = sigma_to_timestep(sigma_max)
        end_t = sigma_to_timestep(sigma_min)
        timesteps = torch.linspace(start_t, end_t, n + 1)[:-1]
        sigs = [timestep_to_sigma(float(t)) for t in timesteps]
        sigs.append(0.0)
        return torch.FloatTensor(sigs)

    elif scheduler == "ddim_uniform":
        sigs = [0.0]
        ss = max(total_timesteps // n, 1)
        x = 1
        while x < total_timesteps:
            sigs.append(float(sigma_table[x]))
            x += ss
        sigs = sigs[::-1]
        return torch.FloatTensor(sigs)

    elif scheduler == "beta":
        try:
            import scipy.stats
            ts = 1 - numpy.linspace(0, 1, n, endpoint=False)
            indices = scipy.stats.beta.ppf(ts, 0.6, 0.6) * (total_timesteps - 1)
            indices = numpy.rint(indices).astype(int)
            sigs = []
            last_t = -1
            for idx in indices:
                t = min(max(int(idx), 0), total_timesteps - 1)
                if t != last_t:
                    sigs.append(float(sigma_table[t]))
                last_t = t
            sigs.append(0.0)
            return torch.FloatTensor(sigs)
        except ImportError:
            pass

    elif scheduler == "karras":
        rho = 7.0
        ramp = torch.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1.0 / rho)
        max_inv_rho = sigma_max ** (1.0 / rho)
        sigs = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return torch.cat([sigs, torch.zeros(1)])

    elif scheduler == "exponential":
        sigs = torch.linspace(math.log(sigma_max), math.log(sigma_min), n).exp()
        return torch.cat([sigs, torch.zeros(1)])

    elif scheduler == "kl_optimal":
        adj_idxs = torch.linspace(0, 1, n)
        sigs = (adj_idxs * math.atan(sigma_min) + (1 - adj_idxs) * math.atan(sigma_max)).tan()
        return torch.cat([sigs, torch.zeros(1)])

    elif scheduler == "linear_quadratic":
        if n == 1:
            return torch.FloatTensor([1.0, 0.0])
        threshold_noise = 0.025
        linear_steps = n // 2
        linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
        threshold_noise_step_diff = linear_steps - threshold_noise * n
        quadratic_steps = n - linear_steps
        quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps ** 2)
        linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps ** 2)
        const = quadratic_coef * (linear_steps ** 2)
        quadratic_sigma_schedule = [
            quadratic_coef * (i ** 2) + linear_coef * i + const
            for i in range(linear_steps, n)
        ]
        sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
        sigma_schedule = [1.0 - x for x in sigma_schedule]
        return torch.FloatTensor(sigma_schedule) * sigma_max

    elif scheduler == "beta57":
        try:
            import scipy.stats
            ts = 1 - numpy.linspace(0, 1, n, endpoint=False)
            indices = scipy.stats.beta.ppf(ts, 0.5, 0.7) * (total_timesteps - 1)
            indices = numpy.rint(indices).astype(int)
            sigs = []
            for idx in indices:
                t = min(max(int(idx), 0), total_timesteps - 1)
                sigs.append(float(sigma_table[t]))
            sigs.append(0.0)
            return torch.FloatTensor(sigs)
        except ImportError:
            pass

    elif scheduler in ("capitanZIT", "capitanZiT"):
        return torch.linspace(1.0, 0.0, n + 1)

    # 未知调度器 — 线性兜底
    return torch.linspace(sigma_max, 0.0, n + 1)

# --- 预设系统 ---
CURRENT_DIR = os.path.dirname(__file__)
PRESET_DIR = os.path.join(CURRENT_DIR, "presets")
PRESET_FILE = os.path.join(PRESET_DIR, "sigmas.json")
if not os.path.exists(PRESET_DIR): os.makedirs(PRESET_DIR)
if not os.path.exists(PRESET_FILE):
    with open(PRESET_FILE, 'w', encoding='utf-8') as f: json.dump({}, f)

routes = PromptServer.instance.routes

@routes.get("/laoli/sigmas/presets")
async def get_presets(request):
    if os.path.exists(PRESET_FILE):
        with open(PRESET_FILE, 'r', encoding='utf-8') as f: return web.json_response(json.load(f))
    return web.json_response({})

@routes.post("/laoli/sigmas/save")
async def save_preset(request):
    try:
        data = await request.json()
        name = data.get("name")
        values = data.get("values")
        if not name or not values: return web.json_response({"status": "error"})
        current = {}
        if os.path.exists(PRESET_FILE):
            with open(PRESET_FILE, 'r', encoding='utf-8') as f: current = json.load(f)
        current[name] = values
        with open(PRESET_FILE, 'w', encoding='utf-8') as f: json.dump(current, f, indent=4, ensure_ascii=False)
        return web.json_response({"status": "success"})
    except: return web.json_response({"status": "error"})

@routes.post("/laoli/sigmas/delete")
async def delete_preset(request):
    try:
        data = await request.json()
        name = data.get("name")
        if os.path.exists(PRESET_FILE):
            with open(PRESET_FILE, 'r', encoding='utf-8') as f: current = json.load(f)
            if name in current:
                del current[name]
                with open(PRESET_FILE, 'w', encoding='utf-8') as f: json.dump(current, f, indent=4, ensure_ascii=False)
        return web.json_response({"status": "success"})
    except: return web.json_response({"status": "error"})

@routes.post("/laoli/sigmas/generate")
async def generate_sigmas(request):
    """前端请求：根据 steps/scheduler/denoise 生成默认 sigma 曲线（不需要 model）"""
    try:
        data = await request.json()
        steps = int(data.get("steps", 20))
        denoise = float(data.get("denoise", 1.0))
        scheduler = data.get("scheduler", "normal")
        node_id = data.get("node_id")

        print(f"[老李-后端] generate_sigmas: 收到请求 scheduler={scheduler}, node_id={node_id}, 缓存数量={len(_cached_model_sampling)}")

        total_steps = steps
        if 0.0 < denoise < 1.0:
            total_steps = int(steps / denoise)
        if total_steps == 0:
            total_steps = steps

        # 计算预览曲线（优先用缓存的真实模型，其次用默认值）
        sigmas = _preview_sigmas(scheduler, total_steps, steps, denoise, node_id=node_id)

        str_values = [f"{x:.4f}" for x in sigmas.tolist()]
        generated_string = "[" + ", ".join(str_values) + "]"

        # 推送给前端
        if node_id:
            PromptServer.instance.send_sync("laoli_sigma_update_event", {
                "node_id": str(node_id),
                "text": generated_string
            })

        return web.json_response({"status": "success", "sigmas": generated_string})
    except Exception as e:
        print(f"[老李-后端] generate_sigmas 出错: {e}")
        return web.json_response({"status": "error", "message": str(e)})


# --- 节点定义 ---

class LaoliSigmaEditor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, ),
                "sigma_string": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": { "unique_id": "UNIQUE_ID" },
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "老李-工具箱/采样"

    def get_sigmas(self, model, steps, denoise, scheduler, sigma_string, unique_id):
        model_sampling = model.get_model_object("model_sampling")

        # 缓存真实 model_sampling，供后续预览使用
        _cached_model_sampling[str(unique_id)] = model_sampling
        print(f"[老李-后端] 缓存 model_sampling: node_id={unique_id}, 类型={type(model_sampling).__name__}")

        # 1. 尝试解析 sigma_string (兼容带[]的格式)
        cleaned_str = sigma_string.replace('[', '').replace(']', '').replace('\n', ',').replace(' ', '')
        values = []
        try:
            for x in cleaned_str.split(','):
                if x: values.append(float(x))
        except ValueError: pass

        if len(values) > 0:
            # 用户编辑过的值是 [0,1] 归一化的，需还原为真实 sigma
            normalized = torch.FloatTensor(values).clamp(0.0, 1.0)

            # 获取模型的真实 sigma 范围
            total_steps = steps
            if denoise < 1.0 and denoise > 0.0:
                total_steps = int(steps / denoise)
            if total_steps == 0: total_steps = steps
            ref_sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, total_steps)
            sig_max = ref_sigmas.max().item()
            sig_min = ref_sigmas[ref_sigmas > 0].min().item() if (ref_sigmas > 0).any() else 0.0

            # [0,1] → [sig_min, sig_max]（保持末尾 0）
            real_sigmas = normalized * (sig_max - sig_min) + sig_min
            # 确保最后一个值为 0（采样器终止条件）
            real_sigmas[-1] = 0.0

            # UI 显示仍用 [0,1]
            str_values = [f"{x:.4f}" for x in normalized.tolist()]
            generated_string = "[" + ", ".join(str_values) + "]"

            print(f"[老李-后端] ID:{unique_id} 使用用户编辑值, sigma范围: [{sig_min:.4f}, {sig_max:.4f}]")
        else:
            # 2. 自动计算 — 使用 ComfyUI 标准流程
            total_steps = steps
            if denoise < 1.0 and denoise > 0.0:
                total_steps = int(steps / denoise)
            if total_steps == 0: total_steps = steps

            real_sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, total_steps)

            # 用完整曲线最大值归一化（denoise 截取前）
            full_sig_max = real_sigmas.max().item()

            if denoise < 1.0:
                real_sigmas = real_sigmas[-(steps + 1):]

            # 归一化用于 UI 显示（使用完整曲线的最大值，不重新归一化）
            if full_sig_max > 0:
                normalized = real_sigmas / full_sig_max
            else:
                normalized = real_sigmas.clone()
            normalized = normalized.clamp(0.0, 1.0)

            str_values = [f"{x:.4f}" for x in normalized.tolist()]
            generated_string = "[" + ", ".join(str_values) + "]"

            print(f"[老李-后端] ID:{unique_id} 自动计算完成, sigma范围: [{real_sigmas[-1].item():.4f}, {real_sigmas[0].item():.4f}]")

        # 主动推送 [0,1] 归一化值给前端显示
        PromptServer.instance.send_sync("laoli_sigma_update_event", {
            "node_id": unique_id,
            "text": generated_string
        })

        return {
            "ui": {"sigma_string": [generated_string]},
            "result": (real_sigmas,)  # 输出真实 sigma 值给采样器
        }

NODE_CLASS_MAPPINGS = { "LaoliSigmaEditor": LaoliSigmaEditor }
NODE_DISPLAY_NAME_MAPPINGS = { "LaoliSigmaEditor": "老李-可视化Sigma编辑器" }
