import torch
import numpy as np
import difflib
import re

# --- 节点 1：音频自动切分 ---
class LaoLiAudioSplitter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "音频": ("AUDIO",),
                "静音阈值_分贝": ("INT", {"default": -40, "min": -100, "max": 0, "step": 1}),
                "最短静音识别_秒": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "单段最大长度_分钟": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 10.0, "step": 0.5}),
                "衔接重叠长度_秒": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.5}),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("音频列表_对接Batch节点", "段落总数")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "split_logic"
    CATEGORY = "老李-工具箱/音频处理"

    def split_logic(self, 音频, 静音阈值_分贝, 最短静音识别_秒, 单段最大长度_分钟, 衔接重叠长度_秒):
        waveform = 音频["waveform"]
        sample_rate = 音频["sample_rate"]
        threshold = 10 ** (静音阈值_分贝 / 20)
        mono = torch.mean(waveform, dim=1).squeeze(0).numpy()
        window_size = int(最短静音识别_秒 * sample_rate)
        max_samples = int(单段最大长度_分钟 * 60 * sample_rate)
        overlap_samples = int(衔接重叠长度_秒 * sample_rate)
        segments = []
        start_idx = 0
        last_silence_end = -1
        i = window_size
        while i < len(mono):
            current_len = i - start_idx
            window = mono[i - window_size : i]
            rms = np.sqrt(np.mean(window**2))
            is_silence = rms < threshold
            if is_silence: last_silence_end = i
            if current_len >= max_samples:
                if last_silence_end > start_idx:
                    segments.append(waveform[:, :, start_idx:last_silence_end])
                    start_idx = last_silence_end
                else:
                    cut_point = i
                    segments.append(waveform[:, :, start_idx:cut_point])
                    start_idx = cut_point - overlap_samples
                i = start_idx + window_size
                last_silence_end = -1
                continue
            if is_silence and current_len > (sample_rate * 10):
                segments.append(waveform[:, :, start_idx:i])
                start_idx = i
                last_silence_end = -1
                i += window_size
            else:
                i += int(sample_rate * 0.5)
        if start_idx < len(mono):
            segments.append(waveform[:, :, start_idx:])
        audio_list = []
        for seg in segments:
            audio_list.append({"waveform": seg, "sample_rate": sample_rate})
        return (audio_list, len(audio_list))

# --- 节点 2：文本合并去重 (新增：自动清洗 ASR 标签) ---
class LaoLiTextJoiner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "文本列表": ("STRING", {"forceInput": True}),
                "智能去重开关": (["开启", "关闭"], {"default": "开启"}),
                "清洗ASR标签": (["开启", "关闭"], {"default": "开启"}),
                "去重敏感度": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("合并后的文本",)
    FUNCTION = "join_text"
    CATEGORY = "老李-工具箱/文本处理"

    def join_text(self, 文本列表, 智能去重开关, 清洗ASR标签, 去重敏感度):
        开关 = 智能去重开关[0] if isinstance(智能去重开关, list) else 智能去重开关
        清洗 = 清洗ASR标签[0] if isinstance(清洗ASR标签, list) else 清洗ASR标签
        敏感度 = 去重敏感度[0] if isinstance(去重敏感度, list) else 去重敏感度
        
        # 1. 展平并初步清洗标签
        all_texts = []
        # 标签匹配正则表达式，例如匹配 [0] (Chinese): 或 [12] (English):
        tag_pattern = re.compile(r'\[\d+\]\s*\(\w+\):\s*')

        for item in 文本列表:
            # 转换成字符串
            t_str = str(item)
            # 如果开启了清洗，则删掉模型自带的标签
            if 清洗 == "开启":
                t_str = tag_pattern.sub('', t_str)
            
            if t_str.strip():
                all_texts.append(t_str.strip())
        
        if not all_texts: return ("",)
        if len(all_texts) == 1: return (all_texts[0],)

        # 2. 智能去重合并逻辑
        if 开关 == "关闭":
            return ("\n".join(all_texts),)

        combined_text = all_texts[0]
        for i in range(1, len(all_texts)):
            str1 = combined_text
            str2 = all_texts[i]
            
            # 取末尾和开头进行模糊匹配
            check_len = 50 
            s1_tail = str1[-check_len:]
            s2_head = str2[:check_len]
            
            matcher = difflib.SequenceMatcher(None, s1_tail, s2_head)
            match = matcher.find_longest_match(0, len(s1_tail), 0, len(s2_head))
            
            # 如果匹配长度超过阈值，则衔接合并
            if match.size > (min(len(s1_tail), len(s2_head)) * 敏感度):
                overlap_end_in_s2 = match.b + match.size
                combined_text += str2[overlap_end_in_s2:]
            else:
                # 尝试通过标点符号或空格衔接
                if combined_text[-1] in "，。！？,.!?":
                    combined_text += str2
                else:
                    combined_text += " " + str2
                
        return (combined_text.strip(),)

NODE_CLASS_MAPPINGS = {
    "LaoLiAudioSplitter": LaoLiAudioSplitter,
    "LaoLiTextJoiner": LaoLiTextJoiner
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LaoLiAudioSplitter": "老李-音频自动切分",
    "LaoLiTextJoiner": "老李-文本合并去重"
}