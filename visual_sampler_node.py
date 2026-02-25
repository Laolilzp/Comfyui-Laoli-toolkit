import torch
import comfy.samplers
import comfy.model_sampling
import comfy.sample
import comfy.utils
from server import PromptServer
from aiohttp import web
import json
import os
import traceback
import math
import base64
import gc
from io import BytesIO
import numpy as np
import weakref

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import comfy.model_management
except ImportError:
    pass

try:
    import comfy.latent_preview
except ImportError:
    comfy.latent_preview = None

# 使用弱引用字典，防止 model_sampling (内部持有模型引用) 导致模型永远驻留显存
_vs_cache = weakref.WeakValueDictionary()
_vs_stage1_cache = {}


def _update_status(node_id, msg):
    if node_id:
        try:
            PromptServer.instance.send_sync("laoli_vs_status", {"node_id": node_id, "msg": msg})
        except:
            pass
    print(f" ⏳[VisualSampler] {msg}")


def _soft_clean_vram():
    if hasattr(comfy.model_management, "soft_empty_cache"):
        comfy.model_management.soft_empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _force_clean_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _full_unload_models():
    if hasattr(comfy.model_management, "unload_all_models"):
        comfy.model_management.unload_all_models()
    _force_clean_vram()


def _get_latent_format(model):
    lf = None
    try:
        lf = getattr(model.model, "latent_format", None)
    except:
        pass
    if lf is None:
        try:
            lf = model.get_model_object("latent_format")
        except:
            pass
    return lf


def _safe_vae_encode(vae, pixels, node_id=None):
    _soft_clean_vram()
    with torch.no_grad():
        pixels = pixels.cpu()
        
        if pixels.numel() == 0:
            raise ValueError("⚠️ 输入编码的像素张量为空！")

        old_crop = getattr(vae, "crop_input", False)
        
        if pixels.ndim == 5:
            B, T, H, W, C = pixels.shape
            _update_status(node_id, f"📦 VAE 编码 (像素: {W}x{H})...")
            if old_crop:
                try:
                    ratio = vae.spacial_compression_encode()
                except:
                    ratio = 8
                H_new = (H // ratio) * ratio
                W_new = (W // ratio) * ratio
                if H != H_new or W != W_new:
                    y_off = (H % ratio) // 2
                    x_off = (W % ratio) // 2
                    pixels = pixels[:, :, y_off:y_off + H_new, x_off:x_off + W_new, :]
                vae.crop_input = False
        else:
            B, H, W, C = pixels.shape
            _update_status(node_id, f"📦 VAE 编码 (像素: {W}x{H})...")

        try:
            res = vae.encode(pixels)
        except Exception as e:
            err_msg = str(e).lower()
            if any(k in err_msg for k in["conv2d", "expected", "3d", "4d", "size"]):
                _update_status(node_id, "⚠️ VAE架构自适应降维编码中...")
                if pixels.ndim == 5:
                    pixels_4d = pixels.reshape(B * T, H, W, C)
                else:
                    pixels_4d = pixels
                    
                try:
                    res = vae.encode(pixels_4d)
                except comfy.model_management.OOM_EXCEPTION:
                    _update_status(node_id, "⚠️ VAE编码OOM，切块...")
                    _force_clean_vram()
                    if hasattr(vae, "encode_tiled"):
                        res = vae.encode_tiled(pixels_4d, tile_x=512, tile_y=512)
                    else:
                        res = vae.encode(pixels_4d)
            elif "oom" in err_msg or "memory" in err_msg or "allocate" in err_msg:
                _update_status(node_id, "⚠️ VAE编码OOM，切块...")
                _force_clean_vram()
                if hasattr(vae, "encode_tiled"):
                    res = vae.encode_tiled(pixels, tile_x=512, tile_y=512)
                else:
                    res = vae.encode(pixels)
            else:
                raise e
        finally:
            if old_crop:
                vae.crop_input = old_crop
                
        res_cpu = res.cpu()
        del res, pixels
        _soft_clean_vram()
        return res_cpu


def _safe_vae_decode(vae, latent_samples, node_id=None):
    _soft_clean_vram()
    with torch.no_grad():
        _update_status(node_id,
                       f"🖼️ VAE 解码 (shape: {list(latent_samples.shape)})...")
        try:
            res = vae.decode(latent_samples)
            res_cpu = res.cpu()
            del res
            
            if res_cpu.ndim == 5:
                if res_cpu.shape[1] in [1, 3, 4] and res_cpu.shape[-1] not in [1, 3, 4]:
                    res_cpu = res_cpu.permute(0, 2, 3, 4, 1)
            elif res_cpu.ndim == 4:
                if res_cpu.shape[1] in[1, 3, 4] and res_cpu.shape[-1] not in[1, 3, 4]:
                    res_cpu = res_cpu.permute(0, 2, 3, 1)
                    
            _soft_clean_vram()
            return res_cpu
        except Exception as e:
            err = str(e).lower()
            if any(k in err for k in["memory", "oom", "allocate", "cuda"]):
                _update_status(node_id, "⚠️ 解码OOM，分块...")
                _force_clean_vram()
                if hasattr(vae, "decode_tiled"):
                    res = vae.decode_tiled(latent_samples, tile_x=128, tile_y=128, overlap=32)
                    res_cpu = res.cpu()
                    del res
                    
                    if res_cpu.ndim == 5 and res_cpu.shape[1] in [1, 3, 4] and res_cpu.shape[-1] not in [1, 3, 4]:
                        res_cpu = res_cpu.permute(0, 2, 3, 4, 1)
                    elif res_cpu.ndim == 4 and res_cpu.shape[1] in [1, 3, 4] and res_cpu.shape[-1] not in [1, 3, 4]:
                        res_cpu = res_cpu.permute(0, 2, 3, 1)
                        
                    _soft_clean_vram()
                    return res_cpu
            raise e


def _color_match(target, source, strength=1.0):
    if target is None or source is None:
        return target
    if target.shape[-1] != 3 or source.shape[-1] != 3:
        return target

    try:
        target = target.cpu().float()
        source = source.cpu().float()

        if source.shape[1] != target.shape[1] or source.shape[2] != target.shape[2]:
            source = torch.nn.functional.interpolate(
                source.permute(0, 3, 1, 2),
                size=(target.shape[1], target.shape[2]),
                mode="bilinear", align_corners=False
            ).permute(0, 2, 3, 1).clamp(0, 1)

        def rgb_to_ycbcr(img):
            r, g, b = img[..., 0:1], img[..., 1:2], img[..., 2:3]
            y  =  0.299 * r + 0.587 * g + 0.114 * b
            cb = -0.169 * r - 0.331 * g + 0.500 * b + 0.5
            cr =  0.500 * r - 0.419 * g - 0.081 * b + 0.5
            return torch.cat([y, cb, cr], dim=-1)

        def ycbcr_to_rgb(img):
            y, cb, cr = img[..., 0:1], img[..., 1:2] - 0.5, img[..., 2:3] - 0.5
            r = y + 1.402 * cr
            g = y - 0.344 * cb - 0.714 * cr
            b = y + 1.772 * cb
            return torch.cat([r, g, b], dim=-1)

        target_ycbcr = rgb_to_ycbcr(target)
        source_ycbcr = rgb_to_ycbcr(source)
        result_ycbcr = target_ycbcr.clone()

        for b in range(target.shape[0]):
            src_b = min(b, source.shape[0] - 1)

            for c in[1, 2]:
                tc = target_ycbcr[b, :, :, c]
                sc = source_ycbcr[src_b, :, :, c]

                t_mean, t_std = tc.mean(), tc.std() + 1e-5
                s_mean, s_std = sc.mean(), sc.std() + 1e-5

                std_ratio = torch.clamp(s_std / t_std, 0.2, 3.0)
                matched = (tc - t_mean) * std_ratio + s_mean
                result_ycbcr[b, :, :, c] = tc * (1 - strength) + matched * strength

            y_tc = target_ycbcr[b, :, :, 0]
            y_sc = source_ycbcr[src_b, :, :, 0]
            y_t_mean, y_t_std = y_tc.mean(), y_tc.std() + 1e-5
            y_s_mean, y_s_std = y_sc.mean(), y_sc.std() + 1e-5
            
            y_std_ratio = torch.clamp(y_s_std / y_t_std, 0.5, 2.0)
            y_matched = (y_tc - y_t_mean) * y_std_ratio + y_s_mean

            y_strength = strength * 0.85 
            result_ycbcr[b, :, :, 0] = y_tc * (1 - y_strength) + y_matched * y_strength

        result_rgb = ycbcr_to_rgb(result_ycbcr)
        return result_rgb.clamp(0, 1)

    except Exception as e:
        print(f" ⚠️ 颜色匹配失败: {e}")
        return target


class SafeNativePreviewer:
    def __init__(self, latent_format):
        self.latent_format = latent_format
        self.factors = None
        if hasattr(latent_format, "latent_rgb_factors") and latent_format.latent_rgb_factors is not None:
            self.factors = torch.tensor(latent_format.latent_rgb_factors, dtype=torch.float32, device="cpu")
            if self.factors.ndim == 2 and self.factors.shape[0] == 3 and self.factors.shape[1] != 3:
                self.factors = self.factors.transpose(0, 1)
        self.bias = None
        if hasattr(latent_format, "latent_rgb_factors_bias") and latent_format.latent_rgb_factors_bias is not None:
            self.bias = torch.tensor(latent_format.latent_rgb_factors_bias, dtype=torch.float32, device="cpu")

    def decode_latent_to_preview_image(self, preview_format, x0):
        if self.factors is None:
            return None
        with torch.no_grad():
            try:
                x0_sys = x0[0:1].detach().clone().cpu().float()
                
                if x0_sys.shape[1] != self.factors.shape[0]:
                    if hasattr(self.latent_format, "process_out"):
                        try:
                            out = self.latent_format.process_out(x0_sys)
                            if out is not None:
                                x0_sys = out
                        except:
                            pass
                            
                if x0_sys.shape[1] != self.factors.shape[0]:
                    C_in = x0_sys.shape[1]
                    C_out = self.factors.shape[0]
                    if C_in == C_out * 4:
                        B, _, H, W = x0_sys.shape
                        x0_sys = x0_sys.reshape(B, C_out, 2, 2, H, W) \
                            .permute(0, 1, 4, 2, 5, 3).reshape(B, C_out, H * 2, W * 2)
                            
                rgb = x0_sys.permute(0, 2, 3, 1) @ self.factors
                if self.bias is not None:
                    rgb = rgb + self.bias
                rgb = (rgb[0] + 1.0) / 2.0
                img_np = (rgb.clamp(0, 1).numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)
                w, h = pil_img.size
                scale = min(384 / max(w, h, 1), 1.0)
                if scale < 1.0:
                    pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                buf = BytesIO()
                pil_img.save(buf, format='JPEG', quality=85)
                return ("JPEG", buf.getvalue())
            except:
                return None


def _create_native_previewer(model):
    latent_format = _get_latent_format(model)
    if latent_format is None:
        return None
        
    if comfy.latent_preview is not None:
        try:
            device = getattr(model, 'load_device', "cpu")
            previewer = comfy.latent_preview.get_previewer(device, latent_format)
            if previewer is not None:
                return previewer
        except:
            pass
            
    if hasattr(latent_format, 'latent_rgb_factors') and latent_format.latent_rgb_factors is not None:
        return SafeNativePreviewer(latent_format)
    return None


def _latent_to_preview_base64_fallback(latent_samples, max_size=384):
    try:
        if latent_samples is None or Image is None:
            return ""
        s = latent_samples.cpu().float()
        
        if s.ndim == 5:
            s = s[0, :, 0, :, :]
        elif s.ndim == 4:
            s = s[0]

        c = s.shape[0]
        if c == 4:
            r = s[0] * 0.298 + s[1] * 0.187 - s[2] * 0.158 - s[3] * 0.184
            g = s[0] * 0.207 + s[1] * 0.286 + s[2] * 0.189 - s[3] * 0.271
            b = s[0] * 0.208 + s[1] * 0.173 + s[2] * 0.264 - s[3] * 0.473
            rgb = torch.stack([r, g, b], dim=0)
            
        elif c >= 16:
            factors = torch.tensor([[-0.0346,  0.0244,  0.0681],[ 0.0034,  0.0210,  0.0687],[ 0.0275, -0.0668, -0.0433],[-0.0174,  0.0160,  0.0617],[ 0.0859,  0.0721,  0.0329],[ 0.0004,  0.0383,  0.0115],[ 0.0405,  0.0861,  0.0915],[-0.0236, -0.0185, -0.0259],[-0.0245,  0.0250,  0.0460],[ 0.0406,  0.0528,  0.0606], [-0.0173, -0.0207, -0.0076],[ 0.0416,  0.0458,  0.0519],[ 0.0526,  0.0754,  0.0945],[-0.0337, -0.0142, -0.0187],[-0.0254, -0.0334, -0.0458],[-0.0204, -0.0330, -0.0415]
            ], dtype=torch.float32, device="cpu")
            if c > 16:
                pad = torch.zeros((c - 16, 3), dtype=torch.float32, device="cpu")
                factors = torch.cat([factors, pad], dim=0)
            rgb_img = s.permute(1, 2, 0) @ factors
            rgb = rgb_img.permute(2, 0, 1)
            
        else:
            rgb = s[:3].clone()
            
        mn, mx = rgb.min(), rgb.max()
        if mx - mn > 0.001:
            rgb = (rgb - mn) / (mx - mn)
        else:
            rgb = rgb * 0 + 0.5
            
        rgb = rgb.clamp(0, 1)
        img_np = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        w, h = pil_img.size
        pil_img = pil_img.resize((w * 8, h * 8), Image.BILINEAR)
        w, h = pil_img.size
        scale = min(max_size / max(w, h, 1), 1.0)
        if scale < 1.0:
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = BytesIO()
        pil_img.save(buf, format='JPEG', quality=85)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception:
        return ""


def _tensor_to_base64(image_tensor, max_size=384):
    try:
        if image_tensor is None or Image is None:
            return ""
        img = image_tensor[0, 0].clamp(0, 1) if image_tensor.ndim == 5 else image_tensor[0].clamp(0, 1)
        pil_img = Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8))
        w, h = pil_img.size
        scale = min(max_size / max(w, h, 1), 1.0)
        if scale < 1.0:
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = BytesIO()
        pil_img.save(buf, format='JPEG', quality=85)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except:
        return ""


def _vs_build_sigma_table():
    alphas_cumprod = torch.cumprod(
        1.0 - torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=torch.float64) ** 2, dim=0)
    return ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5


def _vs_simulate_sigmas(scheduler, n):
    sigma_table = _vs_build_sigma_table()
    total_ts = len(sigma_table)
    s_max, s_min = float(sigma_table[0]), float(sigma_table[-1])
    log_st = torch.log(torch.clamp(sigma_table, min=1e-10))
    def ts2s(t):
        t = max(0.0, min(float(total_ts - 1), float(t)))
        lo = int(math.floor(t))
        w = t - lo
        return math.exp((1 - w) * float(log_st[lo]) + w * float(log_st[min(lo + 1, total_ts - 1)]))
    def s2ts(sigma):
        ls = math.log(max(float(sigma), 1e-10))
        mask = (ls - log_st >= 0)
        if not mask.any(): return 0.0
        lo = min(int(mask.float().cumsum(0).argmax().item()), total_ts - 2)
        a, b = float(log_st[lo]), float(log_st[lo + 1])
        w = max(0, min(1, (a - ls) / (a - b) if a != b else 0))
        return (1 - w) * lo + w * (lo + 1)
    if scheduler == "simple":
        return torch.FloatTensor([float(sigma_table[int(x * total_ts / n)]) for x in range(n)] +[0.0])
    elif scheduler == "normal":
        st, et = s2ts(s_max), s2ts(s_min)
        ts = torch.linspace(st, et, n + 1 if abs(ts2s(et)) < 1e-5 else n)
        r =[ts2s(float(t)) for t in ts]
        if abs(ts2s(et)) >= 1e-5: r.append(0.0)
        return torch.FloatTensor(r)
    elif scheduler == "sgm_uniform":
        return torch.FloatTensor([ts2s(float(t)) for t in torch.linspace(s2ts(s_max), s2ts(s_min), n + 1)] +[0.0])
    elif scheduler == "karras":
        return torch.cat([(s_max**(1/7)+torch.linspace(0,1,n)*(s_min**(1/7)-s_max**(1/7)))**7, torch.zeros(1)])
    elif scheduler == "exponential":
        return torch.cat([torch.linspace(math.log(s_max), math.log(s_min), n).exp(), torch.zeros(1)])
    return torch.linspace(s_max, 0.0, n + 1)


def _vs_parse_sigmas(sigma_str, ms, steps, scheduler, denoise=1.0):
    vals =[float(x.strip()) for x in
            (sigma_str or "").replace('[','').replace(']','').replace('\n',',').split(',') if x.strip()]
    total_steps = int(steps / denoise) if 0.0 < denoise < 1.0 else steps
    if total_steps == 0: total_steps = steps
    try: ref = comfy.samplers.calculate_sigmas(ms, scheduler, total_steps)
    except: ref = torch.linspace(1.0, 0.0, total_steps + 1)
    if vals: return torch.FloatTensor(vals).clamp(0.0, 1.0) * ref[0].item()
    return ref[-(steps + 1):] if 0.0 < denoise < 1.0 else ref


def _upscale_with_model(upscale_model, image_tensor, node_id=None):
    _update_status(node_id, "🔍 运行 AI 放大模型...")
    _soft_clean_vram()
    with torch.no_grad():
        try:
            device = comfy.model_management.get_torch_device()
            up_scale = getattr(upscale_model, "scale", 4)
            upscale_model.to(device)
            in_img = image_tensor.movedim(-1, 1).to(device)
            tile, overlap, oom = 256, 32, True
            while oom:
                try:
                    n_steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                        in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                    s = comfy.utils.tiled_scale(
                        in_img, lambda a: upscale_model(a),
                        tile_x=tile, tile_y=tile, overlap=overlap,
                        upscale_amount=up_scale, pbar=comfy.utils.ProgressBar(n_steps))
                    oom = False
                except comfy.model_management.OOM_EXCEPTION:
                    tile //= 2
                    if tile < 64: raise RuntimeError("放大模型显存不足")
            upscale_model.to("cpu")
            s_out = torch.clamp(s.movedim(1, -1), min=0, max=1.0).cpu()
            del in_img, s
            _force_clean_vram()
            return s_out, up_scale
        except Exception as e:
            print(f" 放大失败: {e}")
            traceback.print_exc()
            try: upscale_model.to("cpu")
            except: pass
            _force_clean_vram()
            return image_tensor.cpu(), 1


def _native_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                     positive, negative, latent_image,
                     denoise=1.0, disable_noise=False,
                     start_step=None, last_step=None,
                     sigmas=None, callback=None, node_id=None):
    latent = latent_image.copy()
    latent_samples = latent["samples"].cpu()
    latent_samples = comfy.sample.fix_empty_latent_channels(model, latent_samples)

    if disable_noise:
        noise = torch.zeros_like(latent_samples)
    else:
        batch_inds = latent.get("batch_index", None)
        if batch_inds is not None:
            noise = comfy.sample.prepare_noise(latent_samples, seed, batch_inds)
        else:
            noise = comfy.sample.prepare_noise(latent_samples, seed)

    if sigmas is not None:
        run_sigmas = sigmas.clone()
    else:
        run_sigmas = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), scheduler, steps).cpu()
        if 0.0 < denoise < 1.0:
            total = int(steps / denoise)
            all_sigmas = comfy.samplers.calculate_sigmas(
                model.get_model_object("model_sampling"), scheduler, total).cpu()
            run_sigmas = all_sigmas[-(steps + 1):]
        if start_step is not None or last_step is not None:
            s_start = min(start_step or 0, len(run_sigmas) - 1)
            s_end = min(last_step or len(run_sigmas) - 1, len(run_sigmas) - 1) + 1
            run_sigmas = run_sigmas[s_start:s_end]

    if len(run_sigmas) < 2: return latent

    _update_status(node_id,
                   f"⚙️ 采样 (noise:{list(noise.shape)}, latent:{list(latent_samples.shape)}, {len(run_sigmas)}步)")

    try:
        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)
    except:
        from comfy.samplers import CFGGuider
        guider = CFGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)

    _soft_clean_vram()
    
    # ★ 核心防弹护城河：检测并化解跨架构坍塌 (5D 强行塞入 2D 模型)
    try:
        out_samples = guider.sample(
            noise, latent_samples, comfy.samplers.sampler_object(sampler_name),
            run_sigmas, denoise_mask=latent.get("noise_mask"),
            callback=callback, seed=seed)
    except Exception as e:
        err_msg = str(e).lower()
        if latent_samples.ndim == 5 and any(k in err_msg for k in["unpack", "4d", "mismatch", "shape", "conv2d", "multiplied"]):
            _update_status(node_id, "⚠️ 检测到2D模型接收了3D/5D环境，启动自适应防弹降维重试...")
            _soft_clean_vram()
            
            B, C, T, H, W = latent_samples.shape
            
            latent_4d = latent_samples.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            noise_4d = noise.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            
            denoise_mask_4d = latent.get("noise_mask")
            if denoise_mask_4d is not None:
                if denoise_mask_4d.shape[0] == B:
                    if denoise_mask_4d.ndim == 4 and denoise_mask_4d.shape[1] == T:
                        denoise_mask_4d = denoise_mask_4d.reshape(B * T, denoise_mask_4d.shape[2], denoise_mask_4d.shape[3])
                    else:
                        denoise_mask_4d = denoise_mask_4d.unsqueeze(1).repeat(1, T, 1, 1).reshape(B * T, denoise_mask_4d.shape[-2], denoise_mask_4d.shape[-1])
            
            def flatten_any(obj):
                if isinstance(obj, torch.Tensor):
                    if obj.ndim == 5:
                        _B, _C, _T, _H, _W = obj.shape
                        return obj.permute(0, 2, 1, 3, 4).reshape(_B * _T, _C, _H, _W)
                elif isinstance(obj, dict):
                    return {k: flatten_any(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return[flatten_any(v) for v in obj]
                elif isinstance(obj, tuple):
                    return tuple(flatten_any(v) for v in obj)
                return obj

            def flatten_conds(conds):
                if not conds: return conds
                new_conds =[]
                for t in conds:
                    new_conds.append([flatten_any(t[0]), flatten_any(t[1])])
                return new_conds

            try:
                guider_4d = comfy.samplers.CFGGuider(model)
                guider_4d.set_conds(flatten_conds(positive), flatten_conds(negative))
                guider_4d.set_cfg(cfg)
            except:
                from comfy.samplers import CFGGuider
                guider_4d = CFGGuider(model)
                guider_4d.set_conds(flatten_conds(positive), flatten_conds(negative))
                guider_4d.set_cfg(cfg)
                
            try:
                out_4d = guider_4d.sample(
                    noise_4d, latent_4d, comfy.samplers.sampler_object(sampler_name),
                    run_sigmas, denoise_mask=denoise_mask_4d,
                    callback=callback, seed=seed)
            except Exception as retry_err:
                err_retry = str(retry_err).lower()
                # 终极拦截：如果降维后仍然报错，说明张量通道数与模型强不兼容 (比如用户忘了接 VAE_2)
                if any(k in err_retry for k in["multiplied", "shape", "size", "channel"]):
                    raise RuntimeError(f"\n❌ 跨架构(5D转4D)重构后依然失败！\n\n原因：您很可能没有为「VAE_2」连接正确的 VAE。\n当前送入二阶段的 Latent 有 {C} 个通道，但二阶段模型不接受该通道数。\n\n解决办法：请务必确保为「VAE_2」连上与「模型_2」配套的 VAE！\n底层报错: {str(retry_err)}")
                raise retry_err
            
            out_samples = out_4d.reshape(B, T, out_4d.shape[1], H, W).permute(0, 2, 1, 3, 4)
        else:
            raise e

    out_samples = out_samples.to(comfy.model_management.intermediate_device())
    latent["samples"] = out_samples
    del noise, latent_samples, guider
    _soft_clean_vram()
    return latent


CURRENT_DIR = os.path.dirname(__file__)
PRESET_DIR = os.path.join(CURRENT_DIR, "presets")
PRESET_FILE = os.path.join(PRESET_DIR, "visual_sampler_presets.json")
if not os.path.exists(PRESET_DIR): os.makedirs(PRESET_DIR)
if not os.path.exists(PRESET_FILE):
    with open(PRESET_FILE, 'w', encoding='utf-8') as f: json.dump({}, f)

routes = PromptServer.instance.routes

@routes.get("/laoli/vsampler/presets")
async def get_vs_presets(request):
    try:
        if os.path.exists(PRESET_FILE):
            with open(PRESET_FILE, 'r', encoding='utf-8') as f:
                return web.json_response(json.load(f))
    except: pass
    return web.json_response({})

@routes.post("/laoli/vsampler/save")
async def save_vs_preset(request):
    try:
        data = await request.json()
        if data.get("name") and data.get("config"):
            cur = json.load(open(PRESET_FILE, 'r', encoding='utf-8')) if os.path.exists(PRESET_FILE) else {}
            cur[data["name"]] = data["config"]
            with open(PRESET_FILE, 'w', encoding='utf-8') as f:
                json.dump(cur, f, indent=4, ensure_ascii=False)
            return web.json_response({"status": "success"})
    except: pass
    return web.json_response({"status": "error"})

@routes.post("/laoli/vsampler/delete")
async def delete_vs_preset(request):
    try:
        data = await request.json()
        if os.path.exists(PRESET_FILE):
            cur = json.load(open(PRESET_FILE, 'r', encoding='utf-8'))
            if data.get("name") in cur:
                del cur[data["name"]]
                with open(PRESET_FILE, 'w', encoding='utf-8') as f:
                    json.dump(cur, f, indent=4, ensure_ascii=False)
                return web.json_response({"status": "success"})
    except: pass
    return web.json_response({"status": "error"})

@routes.post("/laoli/vsampler/generate_sigmas")
async def generate_vs_sigmas(request):
    try:
        data = await request.json()
        steps = int(data.get("steps", 20))
        denoise = float(data.get("denoise", 1.0))
        scheduler = data.get("scheduler", "normal")
        total_steps = int(steps / denoise) if 0.0 < denoise < 1.0 else steps
        if total_steps == 0: total_steps = steps
        
        ms = _vs_cache.get(f"{data.get('node_id', '')}_{data.get('target', '曲线_1')}")
        if ms: sigmas = comfy.samplers.calculate_sigmas(ms, scheduler, total_steps)
        else:
            try: sigmas = comfy.samplers.calculate_sigmas(comfy.model_sampling.ModelSamplingDiscrete(), scheduler, total_steps)
            except: sigmas = _vs_simulate_sigmas(scheduler, total_steps)
            
        final = sigmas[-(steps+1):] if 0.0 < denoise < 1.0 else sigmas
        mx = float(sigmas[0]) if len(sigmas)>0 and float(sigmas[0])>0 else 1.0
        normed = (final / mx).clamp(0, 1)
        return web.json_response({"status":"success","sigmas":"["+", ".join([f"{v:.4f}" for v in normed.tolist()])+"]"})
    except:
        return web.json_response({"status":"success","sigmas":""})


class LaoliVisualSampler:
    @classmethod
    def INPUT_TYPES(s):
        preset_choices = ["当前设置"]
        try:
            p = os.path.join(os.path.dirname(__file__), "presets", "visual_sampler_presets.json")
            if os.path.exists(p):
                preset_choices.extend([k for k in json.load(open(p,'r',encoding='utf-8')).keys() if k not in preset_choices])
        except: pass
        return {
            "required": {
                "模型_1": ("MODEL",), "正面_1": ("CONDITIONING",), "负面_1": ("CONDITIONING",),
                "Latent": ("LATENT",), "VAE_1": ("VAE",), "预设选择": (preset_choices,),
                "随机种_1": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "步数_1": ("INT", {"default": 20, "min": 1}),
                "CFG_1": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "采样器_1": (comfy.samplers.SAMPLER_NAMES,),
                "调度器_1": (comfy.samplers.SCHEDULER_NAMES,),
                "降噪_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "添加噪波_1": (["enable", "disable"],),
                "开始步数_1": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "结束步数_1": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "返回噪波_1": (["enable", "disable"],),
                "曲线_1": ("STRING", {"default": "", "multiline": True}),
                "种子同步": (["enable", "disable"],),
                "随机种_2": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "步数_2": ("INT", {"default": 15, "min": 1}),
                "CFG_2": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "采样器_2": (comfy.samplers.SAMPLER_NAMES, {"default": "euler"}),
                "调度器_2": (comfy.samplers.SCHEDULER_NAMES, {"default": "simple"}),
                "降噪_2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "曲线_2": ("STRING", {"default": "", "multiline": True}),
                "放大倍数": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1}),
                "放大算法": (["bicubic", "bilinear", "nearest-exact", "area", "bislerp"],),
                "放大模式": (["latent", "image", "图像+参考latent"],),
            },
            "optional": {
                "模型_2": ("MODEL",), "正面_2": ("CONDITIONING",), "负面_2": ("CONDITIONING",),
                "VAE_2": ("VAE",), "放大模型": ("UPSCALE_MODEL",), "参考图像": ("IMAGE",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "LATENT")
    RETURN_NAMES = ("一阶预览", "最终图像", "分块解码", "终Latent")
    FUNCTION = "sample"
    CATEGORY = "老李-工具箱/采样"

    def _apply_reference_latent(self, conditioning, latent_image):
        if conditioning is None or latent_image is None: return conditioning
        try:
            import nodes
            if "ReferenceLatent" in nodes.NODE_CLASS_MAPPINGS:
                import inspect
                cls = nodes.NODE_CLASS_MAPPINGS["ReferenceLatent"]
                inst = cls()
                func = getattr(inst, getattr(cls, "FUNCTION", "apply"))
                params = inspect.signature(func).parameters
                kw = {}
                for k in params:
                    if k == "conditioning": kw[k] = conditioning
                    elif k in ["latent","latent_image"]: kw[k] = latent_image
                if kw: return func(**kw)[0]
                else: return func(conditioning, latent_image)[0]
        except: pass
        lat = latent_image["samples"].cpu() if isinstance(latent_image, dict) else latent_image.cpu()
        return [[t[0], {**t[1], "latent_image": lat, "concat_latent_image": lat}] for t in conditioning]

    def _flatten_video(self, tensor):
        if tensor is None: return None
        if tensor.ndim == 5:
            B, T, H, W, C = tensor.shape
            return tensor.reshape(B * T, H, W, C)
        return tensor

    def _get_pixel_size(self, tensor):
        if tensor.ndim == 5:
            return tensor.shape[2], tensor.shape[3]
        elif tensor.ndim == 4:
            return tensor.shape[1], tensor.shape[2]
        return 1, 1

    def _pixel_upscale(self, image, target_w, target_h, method):
        img_4d = image
        if img_4d.ndim == 5:
            B, T, H, W, C = img_4d.shape
            img_4d = img_4d.reshape(B * T, H, W, C)
            
        upscaled = comfy.utils.common_upscale(
            img_4d.movedim(-1, 1).cpu(), target_w, target_h, method, "disabled"
        ).movedim(1, -1).clamp(0, 1)
        
        if image.ndim == 5:
            return upscaled.reshape(B, T, target_h, target_w, C)
        return upscaled

    def _latent_upscale(self, latent_samples, target_lat_w, target_lat_h, method):
        s = latent_samples.cpu()
        
        if s.ndim == 5:
            B, C, T, H, W = s.shape
            s_4d = s.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        else:
            s_4d = s
            
        cur_h, cur_w = s_4d.shape[-2], s_4d.shape[-1]
        if cur_h == target_lat_h and cur_w == target_lat_w: 
            return s
            
        s_4d = comfy.utils.common_upscale(s_4d, target_lat_w, target_lat_h, method, "disabled")
        
        if s.ndim == 5:
            return s_4d.reshape(B, T, C, target_lat_h, target_lat_w).permute(0, 2, 1, 3, 4)
        return s_4d

    def sample(self, unique_id, **kwargs):
        try:
            print(f"\n ═══════════════════════════════════")
            model1 = kwargs["模型_1"]
            pos1, neg1 = kwargs["正面_1"], kwargs["负面_1"]
            latent, vae = kwargs["Latent"], kwargs["VAE_1"]
            model2 = kwargs.get("模型_2", model1)
            pos2 = kwargs.get("正面_2")
            neg2 = kwargs.get("负面_2", neg1)
            vae2 = kwargs.get("VAE_2", vae)

            orig_lat_H = latent["samples"].shape[-2]
            orig_lat_W = latent["samples"].shape[-1]

            _update_status(unique_id, f"🟢 输入Latent: {list(latent['samples'].shape)}")

            expected_pixel_H = orig_lat_H * 8
            expected_pixel_W = orig_lat_W * 8
            _update_status(unique_id, f"📐 用户期望像素: {expected_pixel_W}x{expected_pixel_H}")

            ref_image = kwargs.get("参考图像", None)
            if ref_image is not None:
                try:
                    _update_status(unique_id, "⚙️ 处理外部参考图像...")
                    is_5d = latent["samples"].ndim == 5
                    ref_px = ref_image.unsqueeze(1) if is_5d and ref_image.ndim == 4 else ref_image
                    ref_lat = _safe_vae_encode(vae, ref_px, unique_id)
                    pos1 = self._apply_reference_latent(pos1, {"samples": ref_lat})
                    del ref_lat, ref_px
                except Exception as e:
                    _update_status(unique_id, f"⚠️ 参考图像处理失败: {e}")

            ms1 = model1.get_model_object("model_sampling")
            ms2 = model2.get_model_object("model_sampling")
            _vs_cache[f"{unique_id}_曲线_1"] = ms1
            _vs_cache[f"{unique_id}_曲线_2"] = ms2

            seed1 = kwargs.get("随机种_1", 0)
            seed2 = seed1 if kwargs.get("种子同步","disable")=="enable" else kwargs.get("随机种_2", 0)
            steps1 = kwargs.get("步数_1", 20)
            cfg1 = kwargs.get("CFG_1", 8.0)
            denoise1 = kwargs.get("降噪_1", 1.0)
            scheduler1 = kwargs.get("调度器_1", "normal")
            curve1 = kwargs.get("曲线_1", "")
            start1 = kwargs.get("开始步数_1", 0)
            end1 = kwargs.get("结束步数_1", 10000)
            upscale_mode = kwargs.get("放大模式", "latent")
            upscale_factor = kwargs.get("放大倍数", 1.0)
            upscale_model = kwargs.get("放大模型", None)
            upscale_method = kwargs.get("放大算法", "bicubic")

            # ============== 一阶段曲线解析与持久化更新 ==============
            total_steps1 = int(steps1 / denoise1) if 0.0 < denoise1 < 1.0 else steps1
            if total_steps1 == 0: total_steps1 = steps1
            try: ref1 = comfy.samplers.calculate_sigmas(ms1, scheduler1, total_steps1)
            except: ref1 = torch.linspace(1.0, 0.0, total_steps1 + 1)
            
            global_max1 = float(ref1[0].item()) if len(ref1)>0 and ref1[0].item()>0 else 1.0
            
            custom_s1_full = _vs_parse_sigmas(curve1, ms1, steps1, scheduler1, denoise1)
            has_vals1 = bool([x for x in curve1.replace('[','').replace(']','').replace('\n',',').split(',') if x.strip()])
            
            ui1 = curve1 if curve1 and not has_vals1 else "["+", ".join([f"{v:.4f}" for v in (custom_s1_full/global_max1).clamp(0,1).tolist()])+"]"
            PromptServer.instance.send_sync("laoli_vs_update", {"node_id": unique_id, "text": ui1, "target": "曲线_1"})
            
            if curve1:
                custom_s1 = custom_s1_full[min(start1, len(custom_s1_full)-1):min(end1, len(custom_s1_full)-1)+1]
            else:
                custom_s1 = None
            # =========================================================

            def hash_t(t):
                try:
                    if isinstance(t, torch.Tensor) and t.numel() > 0:
                        return f"{'_'.join(map(str, t.shape))}_{float(t.sum()):.4f}_{float(t.std()):.4f}"
                except: pass
                return "none"
                
            pos_t = pos1[0][0] if pos1 and isinstance(pos1, list) and len(pos1)>0 else None
            neg_t = neg1[0][0] if neg1 and isinstance(neg1, list) and len(neg1)>0 else None
            up_name = upscale_model.__class__.__name__ if upscale_model else "none"

            s1_sampler = kwargs.get("采样器_1")
            s1_add_noise = kwargs.get("添加噪波_1", "enable")
            s1_ret_noise = kwargs.get("返回噪波_1", "disable")

            # ★加入 VAE_1 和 VAE_2 的底层ID，完美阻断由于换 VAE 但没换参数造成的错误缓存命中
            s1_key = (f"{unique_id}_{id(model1)}_{id(vae)}_{id(vae2)}_{hash_t(pos_t)}_{hash_t(neg_t)}_"
                      f"{hash_t(latent.get('samples'))}_{seed1}_{steps1}_{cfg1}_{denoise1}_"
                      f"{s1_sampler}_{scheduler1}_{s1_add_noise}_{s1_ret_noise}_"
                      f"{start1}_{end1}_{hash_t(custom_s1)}_"
                      f"{upscale_factor}_{upscale_mode}_{upscale_method}_{up_name}_v22")

            global _vs_stage1_cache
            cached = _vs_stage1_cache.get(unique_id)

            if cached and cached['key'] == s1_key:
                _update_status(unique_id, "⚡ 命中缓存...")
                latent_out_1 = {"samples": cached['latent_out_1'].clone()}
                preview_1 = cached['preview_1'].clone()
                stage1_final_image = cached.get('stage1_final_image', preview_1).clone()
                color_ref_image = cached.get('color_ref_image', preview_1).clone()
                s1_h, s1_w = self._get_pixel_size(preview_1)
                PromptServer.instance.send_sync("laoli_vs_preview", {
                    "node_id": unique_id, "stage": "stage1",
                    "image": _tensor_to_base64(preview_1), "final": True,
                    "pixel_h": int(s1_h), "pixel_w": int(s1_w),
                    "lat_h": int(latent_out_1["samples"].shape[-2]),
                    "lat_w": int(latent_out_1["samples"].shape[-1])
                })
            else:
                _update_status(unique_id, f"🚀 [一阶段] 采样")
                latent_out_1 = _native_ksampler(
                    model1, seed1, steps1, cfg1, s1_sampler, scheduler1,
                    pos1, neg1, latent, denoise=denoise1,
                    disable_noise=(s1_add_noise == "disable"),
                    start_step=start1 if start1>0 else None,
                    last_step=end1 if end1<10000 else None,
                    sigmas=custom_s1,
                    callback=self._make_step_callback(unique_id, "stage1", _create_native_previewer(model1)),
                    node_id=unique_id)

                sampled_shape = latent_out_1["samples"].shape
                _update_status(unique_id, f"📊 采样输出: {list(sampled_shape)}")

                if id(model1) != id(model2):
                    _update_status(unique_id, "🗑️ 卸载一阶段模型...")
                    _full_unload_models()

                raw_decoded = _safe_vae_decode(vae, latent_out_1["samples"], unique_id)
                raw_h, raw_w = self._get_pixel_size(raw_decoded)

                _update_status(unique_id,
                               f"🖼️ VAE原始解码: {raw_w}x{raw_h}, 期望: {expected_pixel_W}x{expected_pixel_H}")

                if raw_w != expected_pixel_W or raw_h != expected_pixel_H:
                    _update_status(unique_id, f"🔧 尺寸校正: {raw_w}x{raw_h} → {expected_pixel_W}x{expected_pixel_H}")
                    preview_1 = self._pixel_upscale(raw_decoded, expected_pixel_W, expected_pixel_H, upscale_method)
                else:
                    preview_1 = raw_decoded
                del raw_decoded

                s1_h, s1_w = self._get_pixel_size(preview_1)
                _update_status(unique_id, f"🖼️ 一阶段最终: {s1_w}x{s1_h}")

                PromptServer.instance.send_sync("laoli_vs_preview", {
                    "node_id": unique_id, "stage": "stage1",
                    "image": _tensor_to_base64(preview_1), "final": True,
                    "pixel_h": int(s1_h), "pixel_w": int(s1_w),
                    "lat_h": int(sampled_shape[-2]), "lat_w": int(sampled_shape[-1])
                })

                color_ref_image = preview_1.clone()
                stage1_final_image = preview_1

                if upscale_mode in ["image", "图像+参考latent"]:
                    if preview_1.ndim == 5:
                        p1_4d = preview_1.reshape(-1, s1_h, s1_w, preview_1.shape[-1])
                    else:
                        p1_4d = preview_1

                    if upscale_model is not None:
                        upscaled_4d, ai_scale = _upscale_with_model(upscale_model, p1_4d, unique_id)
                        ai_h, ai_w = upscaled_4d.shape[1], upscaled_4d.shape[2]
                        _update_status(unique_id,
                                       f"📊 AI放大后: {ai_w}x{ai_h} ({ai_scale}x)")

                        if abs(upscale_factor - 1.0) > 0.05:
                            final_w = max(8, (round(ai_w * upscale_factor) // 8) * 8)
                            final_h = max(8, (round(ai_h * upscale_factor) // 8) * 8)
                            _update_status(unique_id,
                                           f"📐 缩放倍数{upscale_factor}: {ai_w}x{ai_h} → {final_w}x{final_h}")
                            result_4d = self._pixel_upscale(upscaled_4d, final_w, final_h, upscale_method)
                        else:
                            result_4d = upscaled_4d
                        del upscaled_4d

                    elif abs(upscale_factor - 1.0) > 0.05:
                        final_w = max(8, (round(s1_w * upscale_factor) // 8) * 8)
                        final_h = max(8, (round(s1_h * upscale_factor) // 8) * 8)
                        _update_status(unique_id,
                                       f"📐 算法缩放: {s1_w}x{s1_h} → {final_w}x{final_h}")
                        result_4d = self._pixel_upscale(p1_4d, final_w, final_h, upscale_method)
                    else:
                        result_4d = None

                    if result_4d is not None:
                        if preview_1.ndim == 5:
                            final_pixel_h, final_pixel_w = result_4d.shape[1], result_4d.shape[2]
                            stage1_final_image = result_4d.reshape(
                                preview_1.shape[0], preview_1.shape[1],
                                final_pixel_h, final_pixel_w, preview_1.shape[-1])
                        else:
                            stage1_final_image = result_4d
                        del result_4d

                        _update_status(unique_id, f"📦 编码放大图→Latent...")
                        latent_out_1 = {"samples": _safe_vae_encode(vae2, stage1_final_image, unique_id)}
                        _update_status(unique_id, f"📦 编码后Latent: {list(latent_out_1['samples'].shape)}")

                    del p1_4d

                else:
                    sampled_lat_H = latent_out_1["samples"].shape[-2]
                    sampled_lat_W = latent_out_1["samples"].shape[-1]
                    target_lat_W = max(1, round(sampled_lat_W * upscale_factor))
                    target_lat_H = max(1, round(sampled_lat_H * upscale_factor))
                    need_redecode = False

                    if target_lat_W != sampled_lat_W or target_lat_H != sampled_lat_H:
                        _update_status(unique_id,
                                       f"📐 Latent缩放: {sampled_lat_W}x{sampled_lat_H} → {target_lat_W}x{target_lat_H}")
                        latent_out_1 = {"samples": self._latent_upscale(
                            latent_out_1["samples"], target_lat_W, target_lat_H, upscale_method)}
                        need_redecode = True

                    if id(vae) != id(vae2):
                        temp = _safe_vae_decode(vae, latent_out_1["samples"], unique_id)
                        latent_out_1 = {"samples": _safe_vae_encode(vae2, temp, unique_id)}
                        stage1_final_image = temp
                        need_redecode = False

                    if need_redecode:
                        stage1_final_image = _safe_vae_decode(vae, latent_out_1["samples"], unique_id)

                _update_status(unique_id, f"✅ 一阶段完成 → Latent: {list(latent_out_1['samples'].shape)}")

                _vs_stage1_cache[unique_id] = {
                    'key': s1_key,
                    'latent_out_1': latent_out_1["samples"].cpu().clone(),
                    'preview_1': preview_1.cpu().clone(),
                    'stage1_final_image': stage1_final_image.cpu().clone(),
                    'color_ref_image': color_ref_image.cpu().clone(),
                }

            if upscale_mode == "图像+参考latent" and pos2 is not None:
                _update_status(unique_id, "🔗 注入参考Latent...")
                pos2 = self._apply_reference_latent(pos2, latent_out_1)

            if pos2 is None:
                p1 = self._flatten_video(preview_1)
                f_img = self._flatten_video(stage1_final_image)
                sf_h, sf_w = self._get_pixel_size(stage1_final_image)
                PromptServer.instance.send_sync("laoli_vs_preview", {
                    "node_id": unique_id, "stage": "stage2",
                    "image": _tensor_to_base64(stage1_final_image), "final": True,
                    "pixel_h": int(sf_h), "pixel_w": int(sf_w),
                    "lat_h": int(latent_out_1["samples"].shape[-2]),
                    "lat_w": int(latent_out_1["samples"].shape[-1])
                })
                _full_unload_models()
                _update_status(unique_id, "✅ 完毕 (无二阶段)")
                return (p1, f_img, f_img, latent_out_1)

            # ============== 二阶段参数准备与曲线持久化 ==============
            steps2 = kwargs.get("步数_2", 15)
            cfg2 = kwargs.get("CFG_2", 8.0)
            denoise2 = kwargs.get("降噪_2", 0.5)
            scheduler2 = kwargs.get("调度器_2", "simple")
            curve2 = kwargs.get("曲线_2", "")

            total_steps2 = int(steps2 / denoise2) if 0.0 < denoise2 < 1.0 else steps2
            if total_steps2 == 0: total_steps2 = steps2
            try: ref2 = comfy.samplers.calculate_sigmas(ms2, scheduler2, total_steps2)
            except: ref2 = torch.linspace(1.0, 0.0, total_steps2 + 1)
            
            global_max2 = float(ref2[0].item()) if len(ref2)>0 and ref2[0].item()>0 else 1.0
            
            custom_s2_full = _vs_parse_sigmas(curve2, ms2, steps2, scheduler2, denoise2)
            has_vals2 = bool([x for x in curve2.replace('[','').replace(']','').replace('\n',',').split(',') if x.strip()])
            
            ui2 = curve2 if curve2 and not has_vals2 else "["+", ".join([f"{v:.4f}" for v in (custom_s2_full/global_max2).clamp(0,1).tolist()])+"]"
            PromptServer.instance.send_sync("laoli_vs_update", {"node_id": unique_id, "text": ui2, "target": "曲线_2"})
            
            if curve2: custom_s2 = custom_s2_full
            else: custom_s2 = None
            # =========================================================

            _update_status(unique_id,
                           f"🚀 [二阶段] 重绘 (降噪:{denoise2}), Latent: {list(latent_out_1['samples'].shape)}")

            latent_out_final = _native_ksampler(
                model2, seed2, steps2, cfg2, kwargs.get("采样器_2"), scheduler2,
                pos2, neg2, latent_out_1, denoise=denoise2,
                sigmas=custom_s2,
                callback=self._make_step_callback(unique_id, "stage2", _create_native_previewer(model2)),
                node_id=unique_id)

            _update_status(unique_id, "🗑️ 卸载采样模型...")
            _full_unload_models()

            _update_status(unique_id, "🎆 终极 VAE 解码...")
            final_image = _safe_vae_decode(vae2, latent_out_final["samples"], unique_id)

            fi_h, fi_w = self._get_pixel_size(final_image)
            sf_h, sf_w = self._get_pixel_size(stage1_final_image)
            if fi_w != sf_w or fi_h != sf_h:
                _update_status(unique_id, f"🔧 二阶段尺寸校正: {fi_w}x{fi_h} → {sf_w}x{sf_h}")
                final_image = self._pixel_upscale(final_image, sf_w, sf_h, upscale_method)

            _update_status(unique_id, "🎨 颜色匹配...")
            final_flat = self._flatten_video(final_image)
            ref_flat = self._flatten_video(color_ref_image)
            matched_flat = _color_match(final_flat, ref_flat)
            if final_image.ndim == 5:
                final_image = matched_flat.reshape(final_image.shape)
            else:
                final_image = matched_flat
            del final_flat, ref_flat, matched_flat

            fi_h, fi_w = self._get_pixel_size(final_image)
            PromptServer.instance.send_sync("laoli_vs_preview", {
                "node_id": unique_id, "stage": "stage2",
                "image": _tensor_to_base64(final_image), "final": True,
                "pixel_h": int(fi_h), "pixel_w": int(fi_w),
                "lat_h": int(latent_out_final["samples"].shape[-2]),
                "lat_w": int(latent_out_final["samples"].shape[-1])
            })

            p1 = self._flatten_video(preview_1)
            f_img = self._flatten_video(final_image)

            _full_unload_models()
            _update_status(unique_id, "✅ 全部完毕！")
            return (p1, f_img, f_img, latent_out_final)

        except comfy.model_management.InterruptProcessingException:
            _update_status(unique_id, "❌ 用户中断")
            raise
        except Exception as e:
            _update_status(unique_id, "❌ 执行崩溃！查看后台日志")
            traceback.print_exc()
            raise e

    def _make_step_callback(self, node_id, stage, previewer=None):
        def callback(step, x0, x, total_steps):
            try:
                if x0 is None: return
                x0_cpu = x0[:, :, 0, :, :].detach().cpu() if x0.ndim == 5 else x0.detach().cpu()
                
                preview_b64 = None
                
                if previewer:
                    try:
                        if hasattr(previewer, 'decode_latent_to_preview_image'):
                            res = previewer.decode_latent_to_preview_image("JPEG", x0_cpu)
                            if isinstance(res, tuple) and len(res) >= 2:
                                preview_b64 = base64.b64encode(res[1]).decode('utf-8')
                        elif hasattr(previewer, 'decode_latent_to_preview'):
                            pil_img = previewer.decode_latent_to_preview(x0_cpu)
                            if pil_img:
                                buf = BytesIO()
                                pil_img.save(buf, format='JPEG', quality=85)
                                preview_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    except Exception as e:
                        pass
                
                if not preview_b64:
                    preview_b64 = _latent_to_preview_base64_fallback(x0_cpu, max_size=384)
                    
                if preview_b64:
                    PromptServer.instance.send_sync("laoli_vs_preview", {
                        "node_id": node_id, "stage": stage,
                        "image": preview_b64,
                        "step": step, "total": total_steps, "final": False,
                        "lat_h": int(x0.shape[-2]), "lat_w": int(x0.shape[-1])
                    })
                del x0_cpu
            except: pass
        return callback


NODE_CLASS_MAPPINGS = {"LaoliVisualSampler": LaoliVisualSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"LaoliVisualSampler": "老李-可视化采样器"}