import gc
import inspect
import os

import folder_paths
import comfy.model_management as mm

from llama_cpp import Llama
from llama_cpp._ggml import GGMLType

try:
    from llama_cpp.llama_chat_format import Qwen35ChatHandler
except ImportError:
    Qwen35ChatHandler = None

# 模型系列 -> chat handler 基础名
SERIES_HANDLERS = {
    "Qwen3.5-VL": "Qwen3.5",
    "Qwen3.8-VL": "Qwen3.8",
}

KV_TYPE_MAP = {
    "默认(F16)": GGMLType.GGML_TYPE_F16,
    "F32": GGMLType.GGML_TYPE_F32,
    "BF16": GGMLType.GGML_TYPE_BF16,
    "Q8_0": GGMLType.GGML_TYPE_Q8_0,
    "Q4_0": GGMLType.GGML_TYPE_Q4_0,
    "Q4_1": GGMLType.GGML_TYPE_Q4_1,
    "Q5_0": GGMLType.GGML_TYPE_Q5_0,
    "Q5_1": GGMLType.GGML_TYPE_Q5_1,
    "IQ4_NL": GGMLType.GGML_TYPE_IQ4_NL,
}

REASONING_EFFORTS = ["low", "medium", "high", "xhigh"]


class QWEN_TE_STORAGE:
    llm = None
    chat_handler = None
    current_config = None

    @classmethod
    def clean(cls):
        try:
            if cls.llm:
                cls.llm.close()
        except Exception:
            pass
        try:
            if cls.chat_handler and hasattr(cls.chat_handler, "_exit_stack"):
                cls.chat_handler._exit_stack.close()
        except Exception:
            pass
        cls.llm = None
        cls.chat_handler = None
        cls.current_config = None
        gc.collect()
        mm.soft_empty_cache()

    @classmethod
    def reset_context(cls):
        """生成结束后清理 KV 缓存，避免影响下一次无关推理。"""
        if cls.llm is None:
            return
        try:
            cls.llm.n_tokens = 0
            cls.llm._ctx.memory_clear(True)
            if getattr(cls.llm, "is_hybrid", False) and getattr(cls.llm, "_hybrid_cache_mgr", None) is not None:
                cls.llm._hybrid_cache_mgr.clear()
        except Exception:
            pass

    @classmethod
    def load_model(cls, config):
        if Qwen35ChatHandler is None:
            raise RuntimeError(
                "Qwen35ChatHandler 不可用，请从 "
                "https://github.com/JamePeng/llama-cpp-python/releases 更新 llama-cpp-python"
            )
        cls.clean()
        cls.current_config = config.copy()

        model_path = folder_paths.get_full_path("LLM", config["model"])
        if model_path is None:
            raise FileNotFoundError(f"找不到主模型: {config['model']}")
        mmproj_path = None
        if config["mmproj"] and config["mmproj"] != "None":
            mmproj_path = folder_paths.get_full_path("LLM", config["mmproj"])
            if mmproj_path is None:
                raise FileNotFoundError(f"找不到 mmproj: {config['mmproj']}")

        enable_thinking = config["enable_thinking"]
        if mmproj_path:
            handler_kwargs = {
                "clip_model_path": mmproj_path,
                "verbose": False,
                "enable_thinking": enable_thinking,
                "preserve_thinking": config["keep_think_in_history"],
            }
            handler_params = inspect.signature(Qwen35ChatHandler.__init__).parameters
            # Qwen3.8 推理强度：仅在 handler 支持时传入
            effort = config.get("reasoning_effort")
            if config["series"] == "Qwen3.8-VL" and effort:
                for key in ("reasoning_effort", "effort"):
                    if key in handler_params:
                        handler_kwargs[key] = effort
                        break
                else:
                    print(f"[QwenTE] 当前 llama-cpp-python 的 Qwen35ChatHandler 不支持推理强度参数，已忽略: {effort}")
            cls.chat_handler = Qwen35ChatHandler(**handler_kwargs)
        else:
            cls.chat_handler = None
            if enable_thinking:
                print("[QwenTE] 未加载 mmproj，纯文本模式下思考开关由聊天模板控制")

        llama_kwargs = {
            "model_path": model_path,
            "chat_handler": cls.chat_handler,
            "n_gpu_layers": config["n_gpu_layers"],
            "n_ctx": config["n_ctx"],
            "type_k": KV_TYPE_MAP[config["type_k"]],
            "type_v": KV_TYPE_MAP[config["type_v"]],
            "verbose": False,
        }
        llama_params = inspect.signature(Llama.__init__).parameters
        if config["moe_cpu"]:
            n_cpu_moe = config["moe_cpu_layers"] if config["moe_cpu_layers"] > 0 else 9999
            if "n_cpu_moe" in llama_params:
                llama_kwargs["n_cpu_moe"] = n_cpu_moe
            else:
                print("[QwenTE] 当前 llama-cpp-python 不支持 n_cpu_moe 参数，MoE 专家上 CPU 已忽略")

        print(f"[QwenTE] Loading model: {config['model']}")
        print(f"[QwenTE] n_gpu_layers={llama_kwargs['n_gpu_layers']}, n_ctx={llama_kwargs['n_ctx']}, "
              f"type_k={llama_kwargs['type_k']}, type_v={llama_kwargs['type_v']}")
        cls.llm = Llama(**llama_kwargs)

    @classmethod
    def ensure_loaded(cls, config):
        if cls.llm is None or cls.current_config != config:
            cls.load_model(config)
