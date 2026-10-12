import folder_paths

from .common import QWEN_TE_STORAGE, SERIES_HANDLERS, KV_TYPE_MAP, REASONING_EFFORTS


def _model_lists():
    all_llms = folder_paths.get_filename_list("LLM")
    model_list = [f for f in all_llms if "mmproj" not in f.lower()]
    mmproj_list = ["None"] + [f for f in all_llms if "mmproj" in f.lower()]
    return model_list, mmproj_list


class QwenTE_ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        model_list, mmproj_list = _model_lists()
        return {"required": {
            "模型系列": (list(SERIES_HANDLERS.keys()), {"default": "Qwen3.5-VL",
                                                        "tooltip": "Qwen3.5-VL / Qwen3.8-VL，决定聊天模板与视觉处理方式"}),
            "主模型": (model_list, {"tooltip": "models/LLM 下的 GGUF 主模型（支持子目录）"}),
            "视觉投影mmproj": (mmproj_list, {"default": "None",
                                            "tooltip": "与主模型配套的 mmproj 视觉投影权重；处理图片/视频时必选"}),
            "启用思考": ("BOOLEAN", {"default": False,
                                     "tooltip": "开启后模型先输出 <think> 思考过程再给出回答"}),
            "保留历史think": ("BOOLEAN", {"default": False,
                                          "tooltip": "多轮对话时是否在历史中保留思考内容（占用上下文，默认裁剪）"}),
            "上下文长度": ("INT", {"default": 8192, "min": 1024, "max": 262144, "step": 128,
                                   "tooltip": "n_ctx；多轮对话的上下文预算由此减去输出预留"}),
            "GPU层数": ("INT", {"default": -1, "min": -1, "max": 1000, "step": 1,
                                "tooltip": "-1 = 全部层加载到 GPU；显存不足时调小"}),
            "KV缓存K类型": (list(KV_TYPE_MAP.keys()), {"default": "默认(F16)",
                                                      "tooltip": "KV 缓存量化类型，显存紧张时可降为 Q8_0"}),
            "KV缓存V类型": (list(KV_TYPE_MAP.keys()), {"default": "默认(F16)",
                                                      "tooltip": "KV 缓存量化类型，显存紧张时可降为 Q8_0"}),
            "MoE专家上CPU": ("BOOLEAN", {"default": False,
                                         "tooltip": "将 MoE 专家权重放到 CPU 内存，显著降低显存占用（速度变慢）"}),
            "前N层专家上CPU": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1,
                                       "tooltip": "仅前 N 层专家上 CPU；0 = 全部专家层"}),
            "Qwen3.8推理强度": (REASONING_EFFORTS, {"default": "high",
                                                    "tooltip": "Qwen3.8 推理强度档位；当前 llama-cpp-python 尚无对应参数，预留透传"}),
        }}

    RETURN_TYPES = ("QWENLLAMA",)
    RETURN_NAMES = ("qwen模型",)
    FUNCTION = "loadmodel"
    CATEGORY = "QwenTE"
    DESCRIPTION = (
        "加载 Qwen3.5/3.8 GGUF 主模型与 mmproj 视觉投影（基于 llama.cpp）。\n"
        "- 模型文件放在 ComfyUI/models/LLM 目录（支持子目录）。\n"
        "- 同参数重复执行命中缓存，不会重复加载。\n"
        "- 输出连接到 QwenTE_ImageInfer / QwenTE_MultiTurnChat 的「qwen模型」输入。\n"
        "- 27B Q5_K_XL 参考：全层 GPU + 8192 上下文 + F16 KV 约占用 20GB 显存。"
    )

    def loadmodel(self, 模型系列, 主模型, 视觉投影mmproj, 启用思考, 保留历史think, 上下文长度,
                  GPU层数, KV缓存K类型, KV缓存V类型, MoE专家上CPU, 前N层专家上CPU, **kwargs):
        # “Qwen3.8推理强度”含句点，不是合法 Python 标识符，只能经 kwargs 传入
        config = {
            "series": 模型系列,
            "model": 主模型,
            "mmproj": 视觉投影mmproj,
            "enable_thinking": 启用思考,
            "keep_think_in_history": 保留历史think,
            "n_ctx": 上下文长度,
            "n_gpu_layers": GPU层数,
            "type_k": KV缓存K类型,
            "type_v": KV缓存V类型,
            "moe_cpu": MoE专家上CPU,
            "moe_cpu_layers": 前N层专家上CPU,
            "reasoning_effort": kwargs.get("Qwen3.8推理强度", "high"),
        }
        QWEN_TE_STORAGE.ensure_loaded(config)
        return (config,)


NODE_CLASS_MAPPINGS = {
    "QwenTE_ModelLoader": QwenTE_ModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenTE_ModelLoader": "QwenTE 模型加载器",
}
