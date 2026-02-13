import torch
import os
import json
import comfy.samplers
import comfy.model_management
from server import PromptServer
from aiohttp import web

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

        total_steps = steps
        if 0.0 < denoise < 1.0:
            total_steps = int(steps / denoise)
        if total_steps == 0:
            total_steps = steps

        # 使用 comfy 的 sigma 计算 (用默认 model_sampling 作近似)
        # 注意：没有实际 model 时，用 comfy 的默认 model_sampling
        try:
            model_sampling = comfy.model_sampling.ModelSamplingDiscrete()
            sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, total_steps)
            if denoise < 1.0:
                sigmas = sigmas[-(steps + 1):]
        except Exception:
            # 如果 scheduler 不支持或出错，生成线性衰减作为 fallback
            sigmas = torch.linspace(14.6146, 0.0, steps + 1)

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
        sigmas = None
        generated_string = None

        # 1. 尝试解析 sigma_string (兼容带[]的格式)
        cleaned_str = sigma_string.replace('[', '').replace(']', '').replace('\n', ',').replace(' ', '')
        values = []
        try:
            for x in cleaned_str.split(','):
                if x: values.append(float(x))
        except ValueError: pass

        if len(values) > 0:
            sigmas = torch.FloatTensor(values)
            generated_string = sigma_string  # 保持用户输入的原样
        else:
            # 2. 自动计算
            total_steps = steps
            if denoise < 1.0 and denoise > 0.0:
                total_steps = int(steps/denoise)
            if total_steps == 0: total_steps = steps

            model_sampling = model.get_model_object("model_sampling")
            sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, total_steps)
            if denoise < 1.0: sigmas = sigmas[-(steps + 1):]

            str_values = [f"{x:.4f}" for x in sigmas.tolist()]
            generated_string = "[" + ", ".join(str_values) + "]"

            print(f"[老李-后端] ID:{unique_id} 生成完成")

        # 主动推送
        PromptServer.instance.send_sync("laoli_sigma_update_event", {
            "node_id": unique_id,
            "text": generated_string
        })

        return {
            "ui": {"sigma_string": [generated_string]},
            "result": (sigmas,)
        }

NODE_CLASS_MAPPINGS = { "LaoliSigmaEditor": LaoliSigmaEditor }
NODE_DISPLAY_NAME_MAPPINGS = { "LaoliSigmaEditor": "老李-可视化Sigma编辑器" }
