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
            "模型系列": (list(SERIES_HANDLERS.keys()), {"default": "Qwen3.5-VL"}),
            "主模型": (model_list,),
            "视觉投影mmproj": (mmproj_list, {"default": "None"}),
            "启用思考": ("BOOLEAN", {"default": False}),
            "保留历史think": ("BOOLEAN", {"default": False}),
            "上下文长度": ("INT", {"default": 8192, "min": 1024, "max": 262144, "step": 128}),
            "GPU层数": ("INT", {"default": -1, "min": -1, "max": 1000, "step": 1,
                                "tooltip": "-1 = 全部层加载到 GPU"}),
            "KV缓存K类型": (list(KV_TYPE_MAP.keys()), {"default": "默认(F16)"}),
            "KV缓存V类型": (list(KV_TYPE_MAP.keys()), {"default": "默认(F16)"}),
            "MoE专家上CPU": ("BOOLEAN", {"default": False}),
            "前N层专家上CPU": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1,
                                       "tooltip": "0 = 全部 MoE 专家层放在 CPU"}),
            "Qwen3.8推理强度": (REASONING_EFFORTS, {"default": "high"}),
        }}

    RETURN_TYPES = ("QWENLLAMA",)
    RETURN_NAMES = ("qwen模型",)
    FUNCTION = "loadmodel"
    CATEGORY = "QwenTE"

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
