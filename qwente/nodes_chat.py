import json
import os
import random
import re
import time

import numpy as np
from PIL import Image

import folder_paths

from ..nodes import image2base64, parse_json
from .common import QWEN_TE_STORAGE

THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)

OUTPUT_FORMAT_INSTRUCTION = (
    "\n\n【输出格式要求】你的每次回复必须是一个 JSON 对象（不要用 markdown 代码块包裹），格式：\n"
    '{"reply": "给用户看的回复正文", "state": {流程状态对象，原样保留并更新其中的字段}, "options": ["2-4个供用户快捷选择的下一步选项"]}\n'
    "reply 中不要使用 JSON 转义以外的特殊符号；state 必须完整包含输入流程状态的所有字段；"
    "若当前没有适合用户快捷选择的分支，options 输出空数组。"
)


def _parse_json_safe(text, default):
    if text is None:
        return default
    text = text.strip()
    if not text:
        return default
    try:
        return parse_json(text)
    except Exception:
        return default


def _extract_envelope(text):
    """从模型输出中解析 reply/state/options 信封，失败时整体作为 reply。"""
    candidate = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", candidate, re.DOTALL)
    if m:
        candidate = m.group(1)
    else:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end > start:
            candidate = candidate[start:end + 1]
    try:
        data = json.loads(candidate)
    except Exception:
        return {"reply": text.strip(), "state": None, "options": None}
    if not isinstance(data, dict) or "reply" not in data:
        return {"reply": text.strip(), "state": None, "options": None}
    return {
        "reply": str(data.get("reply", "")).strip(),
        "state": data.get("state") if isinstance(data.get("state"), dict) else None,
        "options": data.get("options") if isinstance(data.get("options"), list) else None,
    }


def _token_count(llm, text):
    try:
        return len(llm.tokenize(text.encode("utf-8"), add_bos=False))
    except Exception:
        return max(1, len(text) // 2)


class QwenTE_ChatSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "系统提示词": ("STRING", {"default": "", "multiline": True}),
            "最大历史轮数": ("INT", {"default": 100, "min": 1, "max": 1000, "step": 1,
                                     "tooltip": "超出后最早的用户/助手轮次会被裁剪"}),
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
            "最大边长": ("INT", {"default": 1024, "min": 128, "max": 16384, "step": 64,
                                 "tooltip": "当前图片JSON 中图片的缩放边长"}),
        }}

    RETURN_TYPES = ("QWEN_TE_CHAT_SETTINGS",)
    RETURN_NAMES = ("增强设置",)
    FUNCTION = "pack"
    CATEGORY = "QwenTE"

    def pack(self, 系统提示词, 最大历史轮数, 最大生成token, 温度, top_p, top_k,
             重复惩罚, 频率惩罚, 存在惩罚, seed, 输出think块, 最大边长):
        return ({
            "system_prompt": 系统提示词,
            "max_rounds": 最大历史轮数,
            "max_tokens": 最大生成token,
            "temperature": 温度,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": 重复惩罚,
            "frequency_penalty": 频率惩罚,
            "present_penalty": 存在惩罚,
            "seed": seed,
            "show_think": 输出think块,
            "max_size": 最大边长,
        },)


class QwenTE_MultiTurnChat:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen模型": ("QWENLLAMA",),
                "增强设置": ("QWEN_TE_CHAT_SETTINGS",),
                "用户消息": ("STRING", {"default": "", "multiline": True}),
                "对话历史JSON": ("STRING", {"default": "", "multiline": True,
                                            "tooltip": "将上一轮的对话历史JSON输出连回/粘贴回此处"}),
                "请求ID": ("STRING", {"default": "",
                                      "tooltip": "会话标识，留空自动生成；同一 ID 视为同一会话"}),
                "当前图片JSON": ("STRING", {"default": "[]", "multiline": True,
                                            "tooltip": 'JSON 数组，元素为图片路径（相对 ComfyUI input 目录或绝对路径），如 ["test.png"]'}),
                "流程状态JSON": ("STRING", {"default": "", "multiline": True,
                                            "tooltip": "将上一轮的流程状态JSON输出连回/粘贴回此处"}),
                "选项JSON": ("STRING", {"default": "[]", "multiline": True,
                                        "tooltip": "上一轮的选项回显；选择选项请直接把选项文本填入用户消息"}),
            },
            "optional": {
                "skill加载器": ("QWEN_TE_SKILL",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("回复", "对话历史JSON", "流程状态JSON", "选项JSON", "请求ID", "上下文状态JSON")
    FUNCTION = "process"
    CATEGORY = "QwenTE"

    def _load_images(self, 当前图片JSON, max_size):
        paths = _parse_json_safe(当前图片JSON, [])
        if not isinstance(paths, list):
            return [], 0
        frames = []
        token_est = 0
        for p in paths:
            if not isinstance(p, str) or not p.strip():
                continue
            p = p.strip()
            full = p if os.path.isabs(p) else os.path.join(folder_paths.get_input_directory(), p)
            if not full or not os.path.isfile(full):
                print(f"[QwenTE] 当前图片JSON 中的文件不存在，已跳过: {p}")
                continue
            img = Image.open(full).convert("RGB")
            w, h = img.size
            scale = min(max_size / max(w, h), 1.0)
            if scale < 1.0:
                img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
            token_est += int((img.size[0] / 28) * (img.size[1] / 28))
            frames.append(image2base64(np.array(img)))
        return frames, token_est

    def process(self, qwen模型, 增强设置, 用户消息, 对话历史JSON, 请求ID, 当前图片JSON,
                流程状态JSON, 选项JSON, skill加载器=None):
        QWEN_TE_STORAGE.ensure_loaded(qwen模型)
        llm = QWEN_TE_STORAGE.llm
        cfg = QWEN_TE_STORAGE.current_config

        request_id = 请求ID.strip() or f"{int(time.time() * 1000)}-{''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=10))}"

        history = _parse_json_safe(对话历史JSON, [])
        if not isinstance(history, list):
            history = []
        history = [m for m in history if isinstance(m, dict) and m.get("role") in ("user", "assistant")]

        state = _parse_json_safe(流程状态JSON, {})
        if not isinstance(state, dict):
            state = {}
        if skill加载器 is not None:
            state["skill"] = skill加载器["id"]
            state["skill_name"] = skill加载器["name"]
        state.setdefault("stage", "")
        state.setdefault("loaded_references", [])
        state.setdefault("final_result", "")

        frames, image_token_est = self._load_images(当前图片JSON, 增强设置["max_size"])

        system_parts = []
        if 增强设置["system_prompt"].strip():
            system_parts.append(增强设置["system_prompt"].strip())
        if skill加载器 is not None:
            system_parts.append(f"# 当前加载的技能：{skill加载器['name']}\n{skill加载器['content']}")
        system_parts.append(OUTPUT_FORMAT_INSTRUCTION)
        system_prompt = "\n\n".join(system_parts)

        # 组装消息（历史中的 think 块按加载器配置决定是否保留）
        messages = [{"role": "system", "content": system_prompt}]
        for m in history:
            content = str(m.get("content", ""))
            if m["role"] == "assistant" and not cfg.get("keep_think_in_history"):
                content = THINK_RE.sub("", content).strip()
            messages.append({"role": m["role"], "content": content})

        if frames:
            user_content = [{"type": "text", "text": 用户消息}] if 用户消息.strip() else []
            for data in frames:
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}})
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": 用户消息})

        # token 预算与历史裁剪
        context_limit = cfg["n_ctx"]
        output_reserve = max(增强设置["max_tokens"], 256)
        prompt_budget = context_limit - output_reserve

        def tokens_of(msgs):
            total = image_token_est
            for m in msgs:
                c = m["content"]
                if isinstance(c, str):
                    total += _token_count(llm, c) + 4
                else:
                    total += sum(_token_count(llm, i.get("text", "")) for i in c if isinstance(i, dict) and i.get("type") == "text") + 4
            return total

        used_tokens = tokens_of(messages)
        trimmed = 0
        # messages[0] 是 system；按“轮”（user+assistant 对）从最早开始裁剪
        while used_tokens > prompt_budget and len(messages) > 3:
            del messages[1:3]
            trimmed += 1
            used_tokens = tokens_of(messages)
        max_rounds = 增强设置["max_rounds"]
        while len(messages) > 1 + max_rounds * 2 + 1:  # system + 历史轮 + 本轮 user
            del messages[1:3]
            trimmed += 1
        used_tokens = tokens_of(messages)

        if used_tokens > prompt_budget:
            raise ValueError(
                f"系统提示词/skill 与当前消息已占用 {used_tokens} token，超出预算 {prompt_budget}"
                f"（上下文 {context_limit} - 输出预留 {output_reserve}），请增大上下文长度或减少输入"
            )

        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=增强设置["max_tokens"] if 增强设置["max_tokens"] > 0 else -1,
            temperature=增强设置["temperature"],
            top_p=增强设置["top_p"],
            top_k=增强设置["top_k"],
            repeat_penalty=增强设置["repeat_penalty"],
            frequency_penalty=增强设置["frequency_penalty"],
            present_penalty=增强设置["present_penalty"],
            seed=增强设置["seed"],
        )
        raw_text = (output["choices"][0]["message"]["content"] or "").removeprefix(": ").strip()

        envelope = _extract_envelope(raw_text)
        reply = envelope["reply"]
        if not 增强设置["show_think"]:
            reply = THINK_RE.sub("", reply).strip()

        if envelope["state"]:
            state.update(envelope["state"])
        options = envelope["options"]
        if options is None:
            prev = _parse_json_safe(选项JSON, [])
            options = prev if isinstance(prev, list) else []
        options = [str(o) for o in options][:8]

        now = int(time.time() * 1000)
        history.append({"role": "user", "content": 用户消息, "created_at": now,
                        "token_count": _token_count(llm, 用户消息)})
        history_for_store = raw_text if cfg.get("keep_think_in_history") else THINK_RE.sub("", raw_text).strip()
        history.append({"role": "assistant", "content": history_for_store, "created_at": now,
                        "token_count": _token_count(llm, history_for_store)})

        current_rounds = sum(1 for m in history if m["role"] == "user")
        context_state = {
            "used_tokens": used_tokens,
            "prompt_budget": prompt_budget,
            "context_limit": context_limit,
            "output_reserve": output_reserve,
            "remaining_tokens": prompt_budget - used_tokens,
            "percent": round(used_tokens / max(prompt_budget, 1) * 100, 1),
            "trimmed_messages": trimmed,
            "current_rounds": current_rounds,
            "max_rounds": max_rounds,
            "estimated": False,
        }

        QWEN_TE_STORAGE.reset_context()

        display = reply + f"\n\n[上下文 {context_state['percent']}% | 轮次 {current_rounds}/{max_rounds} | 裁剪 {trimmed}]"
        if options:
            display += "\n选项: " + " / ".join(options)

        return {
            "ui": {"text": [display]},
            "result": (
                reply,
                json.dumps(history, ensure_ascii=False),
                json.dumps(state, ensure_ascii=False),
                json.dumps(options, ensure_ascii=False),
                request_id,
                json.dumps(context_state, ensure_ascii=False),
            ),
        }


NODE_CLASS_MAPPINGS = {
    "QwenTE_ChatSettings": QwenTE_ChatSettings,
    "QwenTE_MultiTurnChat": QwenTE_MultiTurnChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenTE_ChatSettings": "QwenTE 对话设置",
    "QwenTE_MultiTurnChat": "QwenTE 多轮对话",
}
