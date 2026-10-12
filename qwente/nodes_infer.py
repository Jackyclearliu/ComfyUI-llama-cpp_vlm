import re

import numpy as np

import comfy.model_management as mm

from ..nodes import image2base64, scale_image
from .common import QWEN_TE_STORAGE

THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)

IMAGE_INPUTS = ["图片"] + [f"图片{i}" for i in range(2, 9)]


class QwenTE_ImageInfer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen模型": ("QWENLLAMA",),
                "输入模式": (["文本", "图片"], {"default": "文本"}),
                "提示词": ("STRING", {"default": "", "multiline": True}),
                "系统提示词": ("STRING", {"default": "", "multiline": True}),
                "最多帧数": ("INT", {"default": 24, "min": 1, "max": 1024, "step": 1,
                                     "tooltip": "多图/视频帧超过该数量时均匀抽帧"}),
                "最大边长": ("INT", {"default": 1024, "min": 128, "max": 16384, "step": 64,
                                     "tooltip": "输入图片等比缩放到该边长以内，控制视觉 token 占用"}),
                "最大生成token": ("INT", {"default": 4096, "min": 0, "max": 65536, "step": 1}),
                "温度": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 1000, "step": 1}),
                "重复惩罚": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "频率惩罚": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "存在惩罚": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True}),
                "输出think块": ("BOOLEAN", {"default": False}),
                "生成后自动卸载模型": ("BOOLEAN", {"default": False,
                                                 "tooltip": "与其他工作流串联使用时开启，防止显存占用"}),
            },
            "optional": {name: ("IMAGE",) for name in IMAGE_INPUTS},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("文本",)
    FUNCTION = "process"
    CATEGORY = "QwenTE"

    def process(self, qwen模型, 输入模式, 提示词, 系统提示词, 最多帧数, 最大边长, 最大生成token,
                温度, top_p, top_k, 重复惩罚, 频率惩罚, 存在惩罚, seed, 输出think块,
                生成后自动卸载模型, **kwargs):
        QWEN_TE_STORAGE.ensure_loaded(qwen模型)

        frames = []
        if 输入模式 == "图片":
            for name in IMAGE_INPUTS:
                img = kwargs.get(name)
                if img is None:
                    continue
                if img.ndim == 3:
                    frames.append(img)
                elif img.ndim == 4:
                    frames.extend(img[i] for i in range(img.shape[0]))
            if not frames:
                raise ValueError("输入模式为「图片」但未连接任何图片输入")
            if len(frames) > 最多帧数:
                indices = np.linspace(0, len(frames) - 1, 最多帧数, dtype=int)
                frames = [frames[i] for i in indices]
            if QWEN_TE_STORAGE.chat_handler is None:
                raise ValueError("当前模型未加载 mmproj 视觉模块，无法处理图片输入")

        user_content = []
        if 提示词.strip():
            user_content.append({"type": "text", "text": 提示词})
        for frame in frames:
            data = image2base64(scale_image(frame, 最大边长))
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}})

        if not user_content:
            raise ValueError("提示词为空且没有图片输入，没有可推理的内容")

        messages = []
        if 系统提示词.strip():
            messages.append({"role": "system", "content": 系统提示词})
        messages.append({"role": "user", "content": user_content if frames else 提示词})

        output = QWEN_TE_STORAGE.llm.create_chat_completion(
            messages=messages,
            max_tokens=最大生成token if 最大生成token > 0 else -1,
            temperature=温度,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=重复惩罚,
            frequency_penalty=频率惩罚,
            present_penalty=存在惩罚,
            seed=seed,
        )
        text = output["choices"][0]["message"]["content"] or ""
        text = text.removeprefix(": ").lstrip()
        if not 输出think块:
            text = THINK_RE.sub("", text).strip()

        if 生成后自动卸载模型:
            QWEN_TE_STORAGE.clean()
        else:
            QWEN_TE_STORAGE.reset_context()

        return (text,)


NODE_CLASS_MAPPINGS = {
    "QwenTE_ImageInfer": QwenTE_ImageInfer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenTE_ImageInfer": "QwenTE 图像推理",
}
