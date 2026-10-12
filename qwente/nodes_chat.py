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

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}

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
    """从模型输出中解析 reply/state/options 信封。

    依次尝试：完整 JSON → 截断 JSON 的 reply 字段正则提取 → 整体作为纯文本回复。
    """
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
        if isinstance(data, dict) and "reply" in data:
            return {
                "reply": str(data.get("reply", "")).strip(),
                "state": data.get("state") if isinstance(data.get("state"), dict) else None,
                "options": data.get("options") if isinstance(data.get("options"), list) else None,
            }
    except Exception:
        pass

    # JSON 被 max_tokens 截断：尽力提取 reply 字段
    m = re.search(r'"reply"\s*:\s*"((?:[^"\\]|\\.)*)', text)
    if m:
        fragment = m.group(1)
        reply = None
        while fragment:
            try:
                reply = json.loads(f'"{fragment}"')
                break
            except Exception:
                fragment = fragment[:-1]
        if reply:
            om = re.search(r'"options"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            options = None
            if om:
                options = re.findall(r'"((?:[^"\\]|\\.)*)"', om.group(1))
            return {"reply": reply.strip(), "state": None, "options": options}

    return {"reply": text.strip(), "state": None, "options": None}


def _token_count(llm, text):
    try:
        return len(llm.tokenize(text.encode("utf-8"), add_bos=False))
    except Exception:
        return max(1, len(text) // 2)


class QwenTE_ChatSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "系统提示词": ("STRING", {"default": "", "multiline": True,
                                      "tooltip": "附加的系统提示词，会拼接在 skill 内容之前"}),
            "最大历史轮数": ("INT", {"default": 100, "min": 1, "max": 1000, "step": 1,
                                     "tooltip": "超出后最早的用户/助手轮次会被裁剪"}),
            "最大生成token": ("INT", {"default": 4096, "min": 0, "max": 65536, "step": 1,
                                      "tooltip": "同时作为上下文预算的输出预留量"}),
            "温度": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
            "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
            "top_k": ("INT", {"default": 20, "min": 0, "max": 1000, "step": 1}),
            "重复惩罚": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "频率惩罚": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "存在惩罚": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                             "control_after_generate": True}),
            "输出think块": ("BOOLEAN", {"default": False,
                                        "tooltip": "开启后回复中保留 <think> 思考过程（前端折叠显示）"}),
            "最大边长": ("INT", {"default": 1024, "min": 128, "max": 16384, "step": 64,
                                 "tooltip": "上传图片等比缩放到该边长以内，控制视觉 token 占用"}),
        }}

    RETURN_TYPES = ("QWEN_TE_CHAT_SETTINGS",)
    RETURN_NAMES = ("增强设置",)
    FUNCTION = "pack"
    CATEGORY = "QwenTE"
    DESCRIPTION = ("多轮对话参数包：系统提示词、历史轮数、采样参数与图片预处理设置。\n"
                   "输出连接 QwenTE_MultiTurnChat 的「增强设置」输入。")

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
                # 以下控件由节点前端聊天面板自动管理，请勿手工编辑
                "用户消息": ("STRING", {"default": "", "multiline": True}),
                "对话历史JSON": ("STRING", {"default": "", "multiline": True}),
                "请求ID": ("STRING", {"default": "",
                                      "tooltip": "会话标识，留空自动生成；同一 ID 视为同一会话"}),
                "当前媒体JSON": ("STRING", {"default": "[]", "multiline": True,
                                            "tooltip": 'JSON 数组，本轮要发送的图片/视频路径（相对 input 目录或绝对路径）'}),
                "流程状态JSON": ("STRING", {"default": "", "multiline": True}),
                "选项JSON": ("STRING", {"default": "[]", "multiline": True}),
            },
            "optional": {
                "skill加载器": ("QWEN_TE_SKILL",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("对话历史JSON",)
    FUNCTION = "process"
    CATEGORY = "QwenTE"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "多轮对话节点（前端聊天面板）。\n"
        "- 在节点面板的输入框中直接对话：Enter 发送，Shift+Enter 换行；右侧按钮可上传图片（≤8MB）和视频（≤300MB）。\n"
        "- 斜杠命令：/new 开启新会话（清空历史并重置请求ID）、/clear 清空面板显示、/unload 推理完成后卸载模型、/help 查看帮助。\n"
        "- 模型回复以 reply/state/options JSON 信封约定解析：options 会渲染为可点击的快捷选项，state 在轮次间自动透传。\n"
        "- 连接 QwenTE_SkillLoader 可注入技能（skills/ 目录下的 SKILL.md）。\n"
        "- 空消息且无附件时自动跳过推理，整体运行工作流不会产生空白对话。\n"
        "- 输出的对话历史JSON会自动回存到节点内部，也可外接显示节点存档。"
    )

    def _load_media(self, 当前媒体JSON, max_size):
        """返回 (content_items, token_estimate, attachment_names)。"""
        paths = _parse_json_safe(当前媒体JSON, [])
        if not isinstance(paths, list):
            return [], 0, []
        items = []
        token_est = 0
        names = []
        for p in paths:
            if not isinstance(p, str) or not p.strip():
                continue
            p = p.strip()
            full = p if os.path.isabs(p) else os.path.join(folder_paths.get_input_directory(), p)
            if not os.path.isfile(full):
                print(f"[QwenTE] 当前媒体JSON 中的文件不存在，已跳过: {p}")
                continue
            ext = os.path.splitext(full)[1].lower()
            names.append(os.path.basename(full))
            if ext in VIDEO_EXTS:
                # mtmd 原生视频解码（需要 ffmpeg 在 PATH 中）
                items.append({"type": "video_url", "video_url": {"url": os.path.abspath(full)}})
                token_est += int((max_size / 28) ** 2) * 8  # 按 8 帧估算
            else:
                img = Image.open(full).convert("RGB")
                w, h = img.size
                scale = min(max_size / max(w, h), 1.0)
                if scale < 1.0:
                    img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
                token_est += int((img.size[0] / 28) * (img.size[1] / 28))
                items.append({"type": "image_url",
                              "image_url": {"url": f"data:image/jpeg;base64,{image2base64(np.array(img))}"}})
        return items, token_est, names

    def process(self, qwen模型, 增强设置, 用户消息, 对话历史JSON, 请求ID, 当前媒体JSON,
                流程状态JSON, 选项JSON, skill加载器=None):
        # 空消息且未附媒体：跳过推理（整体运行工作流时不应产生空白对话轮次）
        media_paths = _parse_json_safe(当前媒体JSON, [])
        if not 用户消息.strip() and not (isinstance(media_paths, list) and media_paths):
            history = _parse_json_safe(对话历史JSON, [])
            if not isinstance(history, list):
                history = []
            state = _parse_json_safe(流程状态JSON, {})
            options = _parse_json_safe(选项JSON, [])
            payload = {"reply": "", "history": history,
                       "state": state if isinstance(state, dict) else {},
                       "options": options if isinstance(options, list) else [],
                       "request_id": 请求ID.strip(), "context_state": None, "attachments": [],
                       "note": "未输入消息，已跳过推理（在面板输入内容后发送）"}
            return {"ui": {"qwente_chat": [payload]},
                    "result": (json.dumps(history, ensure_ascii=False),)}

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

        # 服务端斜杠命令：/unload 卸载模型后直接返回，不做推理
        if 用户消息.strip() == "/unload":
            QWEN_TE_STORAGE.clean()
            reply = "✅ 模型已卸载，显存已释放。"
            now = int(time.time() * 1000)
            history.append({"role": "user", "content": "/unload", "created_at": now, "token_count": 0})
            history.append({"role": "assistant", "content": reply, "created_at": now, "token_count": 0})
            context_state = {"used_tokens": 0, "prompt_budget": 0, "context_limit": cfg["n_ctx"],
                             "output_reserve": 0, "remaining_tokens": 0, "percent": 0,
                             "trimmed_messages": 0, "current_rounds": 0, "max_rounds": 0, "estimated": False}
            payload = {"reply": reply, "history": history, "state": state, "options": [],
                       "request_id": request_id, "context_state": context_state, "attachments": []}
            return {"ui": {"qwente_chat": [payload]},
                    "result": (json.dumps(history, ensure_ascii=False),)}

        media_items, media_token_est, attachment_names = self._load_media(当前媒体JSON, 增强设置["max_size"])
        if media_items and QWEN_TE_STORAGE.chat_handler is None:
            raise ValueError("当前模型未加载 mmproj 视觉模块，无法处理图片/视频输入")

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

        if media_items:
            user_content = [{"type": "text", "text": 用户消息}] if 用户消息.strip() else []
            user_content.extend(media_items)
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": 用户消息})

        # token 预算与历史裁剪
        context_limit = cfg["n_ctx"]
        output_reserve = max(增强设置["max_tokens"], 256)
        prompt_budget = context_limit - output_reserve

        def tokens_of(msgs):
            total = media_token_est
            for m in msgs:
                c = m["content"]
                if isinstance(c, str):
                    total += _token_count(llm, c) + 4
                else:
                    total += sum(_token_count(llm, i.get("text", "")) for i in c
                                 if isinstance(i, dict) and i.get("type") == "text") + 4
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
        base_reply = THINK_RE.sub("", envelope["reply"]).strip()
        think_match = THINK_RE.search(raw_text)
        think_block = (think_match.group(0) + "\n") if think_match else ""
        # 展示回复：按设置决定是否带 think 块（前端折叠显示）
        reply = (think_block + base_reply) if 增强设置["show_think"] else base_reply

        if envelope["state"]:
            state.update(envelope["state"])
        options = envelope["options"]
        if options is None:
            prev = _parse_json_safe(选项JSON, [])
            options = prev if isinstance(prev, list) else []
        options = [str(o) for o in options][:8]

        now = int(time.time() * 1000)
        user_entry = {"role": "user", "content": 用户消息, "created_at": now,
                      "token_count": _token_count(llm, 用户消息)}
        if attachment_names:
            user_entry["attachments"] = attachment_names
        history.append(user_entry)
        # 历史中存解析后的干净回复（而非原始 JSON 信封），think 按加载器配置保留
        history_content = (think_block + base_reply) if cfg.get("keep_think_in_history") else base_reply
        history.append({"role": "assistant", "content": history_content, "created_at": now,
                        "token_count": _token_count(llm, history_content)})

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

        payload = {"reply": reply, "history": history, "state": state, "options": options,
                   "request_id": request_id, "context_state": context_state,
                   "attachments": attachment_names}
        return {
            "ui": {"qwente_chat": [payload]},
            "result": (json.dumps(history, ensure_ascii=False),),
        }


NODE_CLASS_MAPPINGS = {
    "QwenTE_ChatSettings": QwenTE_ChatSettings,
    "QwenTE_MultiTurnChat": QwenTE_MultiTurnChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenTE_ChatSettings": "QwenTE 对话设置",
    "QwenTE_MultiTurnChat": "QwenTE 多轮对话",
}
