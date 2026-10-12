import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const MAX_IMAGE_BYTES = 8 * 1024 * 1024;
const MAX_VIDEO_BYTES = 300 * 1024 * 1024;
const VIDEO_EXTS = ["mp4", "webm", "mov", "mkv", "avi", "m4v"];

const CSS = `
.qwente-chat { display:flex; flex-direction:column; gap:6px; width:100%; height:100%;
  background:#1e1e24; border-radius:8px; padding:8px; box-sizing:border-box;
  font-family:system-ui,"Segoe UI",sans-serif; font-size:13px; color:#e6e6ea; }
.qwente-chat * { box-sizing:border-box; }
.qwente-head { display:flex; justify-content:space-between; align-items:center; flex:none; }
.qwente-title { font-weight:600; color:#9ab; }
.qwente-badge { font-size:11px; padding:2px 8px; border-radius:10px; background:#2a2a33; color:#8fd; }
.qwente-badge.warn { background:#4d3b1f; color:#fc6; }
.qwente-badge.danger { background:#4d1f1f; color:#f88; }
.qwente-msgs { flex:1 1 auto; min-height:120px; overflow-y:auto; display:flex; flex-direction:column;
  gap:6px; padding:4px; background:#17171c; border-radius:6px; }
.qwente-msg { max-width:85%; padding:7px 10px; border-radius:10px; white-space:pre-wrap;
  word-break:break-word; line-height:1.5; position:relative; }
.qwente-msg.user { align-self:flex-end; background:#2b5278; border-bottom-right-radius:2px; }
.qwente-msg.asst { align-self:flex-start; background:#2a2a33; border-bottom-left-radius:2px; }
.qwente-msg.sys { align-self:center; background:transparent; color:#889; font-size:12px; padding:2px; }
.qwente-copy { position:absolute; top:3px; right:4px; opacity:0; border:none; background:rgba(0,0,0,.35);
  color:#dde; border-radius:4px; font-size:11px; cursor:pointer; padding:1px 5px; transition:opacity .15s; }
.qwente-msg:hover .qwente-copy { opacity:1; }
.qwente-msg.user .qwente-copy { right:auto; left:4px; }
.qwente-attach { display:inline-block; font-size:11px; background:#3a3a46; border-radius:4px;
  padding:1px 6px; margin:2px 4px 0 0; color:#bcd; }
.qwente-think { margin-top:4px; font-size:12px; color:#99a; }
.qwente-think summary { cursor:pointer; color:#7a8; }
.qwente-think div { border-left:2px solid #445; padding-left:6px; margin-top:2px; white-space:pre-wrap; }
.qwente-opts { display:flex; flex-wrap:wrap; gap:4px; flex:none; }
.qwente-chip { font-size:12px; padding:3px 10px; border-radius:12px; background:#24344a;
  color:#9cf; cursor:pointer; border:1px solid #35a; }
.qwente-chip:hover { background:#2f4462; }
.qwente-inputrow { display:flex; gap:6px; align-items:flex-end; flex:none; }
.qwente-input { flex:1; resize:none; min-height:34px; max-height:120px; background:#121216;
  color:#e6e6ea; border:1px solid #383844; border-radius:6px; padding:6px 8px; font:inherit; }
.qwente-btn { flex:none; width:34px; height:34px; border-radius:6px; border:1px solid #383844;
  background:#26262e; color:#cdd; cursor:pointer; font-size:15px; line-height:1; }
.qwente-btn:hover { background:#32323c; }
.qwente-btn.send { background:#2b5278; border-color:#35a; }
.qwente-btn:disabled { opacity:.45; cursor:default; }
.qwente-foot { display:flex; gap:6px; align-items:center; flex:none; }
.qwente-reqid { flex:1; background:#121216; color:#9ab; border:1px solid #383844;
  border-radius:6px; padding:4px 8px; font-size:11px; }
.qwente-foot .qwente-btn { width:auto; height:26px; padding:0 10px; font-size:12px; }
.qwente-pending { align-self:flex-start; color:#8ab; font-size:12px; padding:4px 10px; }
`;

function escapeHtml(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// 兼容旧历史：assistant 消息若是 JSON 信封，提取 reply 纯文本
function displayText(m) {
    const c = String(m.content ?? "");
    if (m.role !== "assistant" || !c.trimStart().startsWith("{")) return c;
    try {
        const d = JSON.parse(c);
        if (d && typeof d.reply === "string") return d.reply;
    } catch { /* 继续尝试截断 JSON */ }
    const mm = c.match(/"reply"\s*:\s*"((?:[^"\\]|\\.)*)/);
    if (mm) {
        try { return JSON.parse(`"${mm[1]}"`); } catch { return mm[1]; }
    }
    return c;
}

function renderContent(text) {
    // think 块折叠显示
    const m = String(text).match(/<think>([\s\S]*?)(?:<\/think>|$)/);
    let body = escapeHtml(String(text).replace(/<think>[\s\S]*?(?:<\/think>|$)/, "").trim());
    if (m) {
        body += `<details class="qwente-think"><summary>思考过程</summary><div>${escapeHtml(m[1].trim())}</div></details>`;
    }
    return body;
}

function hideWidget(node, widget) {
    if (!widget || widget.hidden) return;
    widget.hidden = true;
    widget.computeSize = () => [0, -4];
    if (widget.inputEl) widget.inputEl.style.display = "none";
}

app.registerExtension({
    name: "qwente.chat",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "QwenTE_MultiTurnChat") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            setupChatPanel(this);
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const payload = message?.qwente_chat?.[0];
            if (payload && this._qwente) {
                applyPayload(this, payload);
            }
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            // 工作流加载后，从持久化的对话历史JSON 重建气泡
            const node = this;
            setTimeout(() => rebuildFromWidgets(node), 0);
        };
    },
});

function getW(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function setW(node, name, value) {
    const w = getW(node, name);
    if (w) w.value = value;
}

function getWVal(node, name) {
    return getW(node, name)?.value ?? "";
}

function applyPayload(node, payload) {
    const st = node._qwente;
    st.pending = false;
    st.history = Array.isArray(payload.history) ? payload.history : [];
    setW(node, "对话历史JSON", JSON.stringify(payload.history ?? []));
    setW(node, "流程状态JSON", JSON.stringify(payload.state ?? {}));
    setW(node, "选项JSON", JSON.stringify(payload.options ?? []));
    if (payload.request_id) {
        setW(node, "请求ID", payload.request_id);
        if (st.reqInput && st.reqInput.value !== payload.request_id) st.reqInput.value = payload.request_id;
    }
    setW(node, "用户消息", "");
    setW(node, "当前媒体JSON", "[]");
    st.attachments = [];
    renderAll(node);
    renderBadge(node, payload.context_state);
    renderOptions(node, payload.options ?? []);
    scrollBottom(node);
}

function rebuildFromWidgets(node) {
    if (!node._qwente) return;
    try {
        const hist = JSON.parse(getWVal(node, "对话历史JSON") || "[]");
        node._qwente.history = Array.isArray(hist) ? hist : [];
    } catch { node._qwente.history = []; }
    try {
        const opts = JSON.parse(getWVal(node, "选项JSON") || "[]");
        renderOptions(node, Array.isArray(opts) ? opts : []);
    } catch { /* ignore */ }
    const reqid = getWVal(node, "请求ID");
    if (node._qwente.reqInput && node._qwente.reqInput.value !== reqid) {
        node._qwente.reqInput.value = reqid;
    }
    renderAll(node);
    scrollBottom(node);
}

function renderBadge(node, ctx) {
    const badge = node._qwente?.badge;
    if (!badge) return;
    if (!ctx) {
        try { ctx = JSON.parse(getWVal(node, "上下文状态JSON") || "null"); } catch { ctx = null; }
    }
    if (!ctx || !ctx.context_limit) { badge.textContent = "上下文 --"; return; }
    badge.textContent = `上下文 ${ctx.used_tokens}/${ctx.context_limit} (${ctx.percent}%)`;
    badge.className = "qwente-badge" + (ctx.percent >= 85 ? " danger" : ctx.percent >= 60 ? " warn" : "");
    if (ctx.trimmed_messages > 0) badge.title = `已裁剪 ${ctx.trimmed_messages} 轮最早历史`;
    else badge.title = "";
}

function renderAll(node) {
    const box = node._qwente?.msgs;
    if (!box) return;
    box.innerHTML = "";
    for (const m of node._qwente.history) {
        const div = document.createElement("div");
        div.className = "qwente-msg " + (m.role === "user" ? "user" : "asst");
        const text = displayText(m);
        let html = renderContent(text);
        if (Array.isArray(m.attachments) && m.attachments.length) {
            html = m.attachments.map((a) => `<span class="qwente-attach">📎 ${escapeHtml(a)}</span>`).join("") + html;
        }
        const body = document.createElement("div");
        body.className = "qwente-msgbody";
        body.innerHTML = html || "<i>(空)</i>";
        div.appendChild(body);
        // 气泡复制按钮
        const copyBtn = document.createElement("button");
        copyBtn.className = "qwente-copy";
        copyBtn.textContent = "📋";
        copyBtn.title = "复制内容";
        copyBtn.onclick = async (e) => {
            e.stopPropagation();
            try {
                await navigator.clipboard.writeText(text);
                copyBtn.textContent = "✓";
                setTimeout(() => { copyBtn.textContent = "📋"; }, 1200);
            } catch {
                copyBtn.textContent = "✗";
                setTimeout(() => { copyBtn.textContent = "📋"; }, 1200);
            }
        };
        div.appendChild(copyBtn);
        box.appendChild(div);
    }
    if (node._qwente.pending) {
        const p = document.createElement("div");
        p.className = "qwente-pending";
        p.textContent = "⏳ 生成中…";
        box.appendChild(p);
    }
}

function renderOptions(node, options) {
    const row = node._qwente?.optsRow;
    if (!row) return;
    row.innerHTML = "";
    for (const opt of options) {
        const chip = document.createElement("span");
        chip.className = "qwente-chip";
        chip.textContent = opt;
        chip.onclick = () => { node._qwente.input.value = opt; sendMessage(node); };
        row.appendChild(chip);
    }
    row.style.display = options.length ? "flex" : "none";
}

function scrollBottom(node) {
    const box = node._qwente?.msgs;
    if (box) box.scrollTop = box.scrollHeight;
}

function sysMsg(node, text) {
    const box = node._qwente.msgs;
    const div = document.createElement("div");
    div.className = "qwente-msg sys";
    div.textContent = text;
    box.appendChild(div);
    scrollBottom(node);
}

async function uploadFile(file) {
    const form = new FormData();
    form.append("image", file);
    form.append("type", "input");
    form.append("subfolder", "qwente_chat");
    form.append("overwrite", "true");
    const resp = await api.fetchApi("/upload/image", { method: "POST", body: form });
    if (!resp.ok) throw new Error("上传失败: HTTP " + resp.status);
    const data = await resp.json();
    return (data.subfolder ? data.subfolder + "/" : "") + data.name;
}

async function sendMessage(node) {
    const st = node._qwente;
    const text = st.input.value.trim();
    if (st.pending) return;

    // 斜杠命令（前端侧）
    if (text.startsWith("/") && !st.attachments.length) {
        st.input.value = "";
        if (text === "/new") {
            setW(node, "对话历史JSON", "");
            setW(node, "流程状态JSON", "");
            setW(node, "选项JSON", "[]");
            setW(node, "请求ID", "");
            st.reqInput.value = "";
            st.history = [];
            renderAll(node); renderOptions(node, []); renderBadge(node, null);
            sysMsg(node, "已开始新会话（历史与流程状态已清空）");
        } else if (text === "/clear") {
            sysMsg(node, "输入框已清空");
        } else if (text === "/help") {
            sysMsg(node, "/new 新会话 | /clear 清空输入框 | /unload 卸载模型释放显存 | /help 帮助");
        } else if (text === "/unload") {
            setW(node, "用户消息", "/unload");
            st.pending = true; renderAll(node);
            app.queuePrompt(0, 1);
        } else {
            sysMsg(node, `未知命令: ${text}（输入 /help 查看可用命令）`);
        }
        return;
    }

    if (!text && !st.attachments.length) return;

    setW(node, "用户消息", text);
    setW(node, "当前媒体JSON", JSON.stringify(st.attachments.map((a) => a.path)));
    st.input.value = "";
    // 乐观回显用户消息，后端返回后以完整历史覆盖
    st.history = [...st.history, {
        role: "user", content: text,
        attachments: st.attachments.map((a) => a.name),
    }];
    st.pending = true;
    renderAll(node);
    renderOptions(node, []);
    scrollBottom(node);
    app.queuePrompt(0, 1);
}

function setupChatPanel(node) {
    // 隐藏由前端托管的原生控件
    for (const name of ["用户消息", "对话历史JSON", "当前媒体JSON", "流程状态JSON", "选项JSON", "请求ID"]) {
        hideWidget(node, getW(node, name));
    }

    const root = document.createElement("div");
    root.className = "qwente-chat";
    root.innerHTML = `
      <div class="qwente-head">
        <span class="qwente-title">QwenTE 对话</span>
        <span class="qwente-badge">上下文 --</span>
      </div>
      <div class="qwente-msgs"></div>
      <div class="qwente-opts"></div>
      <div class="qwente-inputrow">
        <textarea class="qwente-input" rows="1" placeholder="输入消息，Enter 发送，Shift+Enter 换行；/help 查看命令"></textarea>
        <button class="qwente-btn" data-act="img" title="上传图片（≤8MB）">🖼</button>
        <button class="qwente-btn" data-act="video" title="上传视频（≤300MB）">🎬</button>
        <button class="qwente-btn send" data-act="send" title="发送">➤</button>
      </div>
      <div class="qwente-foot">
        <input class="qwente-reqid" placeholder="请求ID（留空自动生成，同一 ID 延续同一会话）" />
        <button class="qwente-btn" data-act="regen" title="重新生成上一轮回复">↻ 重试</button>
        <button class="qwente-btn" data-act="export" title="导出对话历史 JSON">⤓ 导出</button>
      </div>`;

    node._qwente = {
        root,
        badge: root.querySelector(".qwente-badge"),
        msgs: root.querySelector(".qwente-msgs"),
        optsRow: root.querySelector(".qwente-opts"),
        input: root.querySelector(".qwente-input"),
        reqInput: root.querySelector(".qwente-reqid"),
        history: [],
        attachments: [],
        pending: false,
    };

    const st = node._qwente;
    st.input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage(node);
        }
        e.stopPropagation();
    });
    st.input.addEventListener("keyup", (e) => e.stopPropagation());
    st.reqInput.addEventListener("change", () => setW(node, "请求ID", st.reqInput.value.trim()));
    st.reqInput.addEventListener("keydown", (e) => e.stopPropagation());

    root.querySelector('[data-act="send"]').onclick = () => sendMessage(node);
    root.querySelector('[data-act="regen"]').onclick = () => regenerate(node);
    root.querySelector('[data-act="export"]').onclick = () => exportHistory(node);
    root.querySelector('[data-act="img"]').onclick = () => pickFile(node, false);
    root.querySelector('[data-act="video"]').onclick = () => pickFile(node, true);

    // 阻止面板内的拖拽/按键被画布劫持
    for (const ev of ["keydown", "keyup", "pointerdown", "pointerup", "wheel"]) {
        root.addEventListener(ev, (e) => e.stopPropagation(), { passive: false });
    }

    const dw = node.addDOMWidget("chat_panel", "qwente_chat_panel", root, {
        serializeValue: () => undefined,
        hideOnZoom: false,
    });
    dw.computeSize = (width) => [width, 480];
    node.setSize([Math.max(node.size[0], 480), Math.max(node.size[1], 620)]);
    setTimeout(() => rebuildFromWidgets(node), 0);
}

function pickFile(node, isVideo) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = isVideo ? "video/*" : "image/*";
    input.onchange = async () => {
        const file = input.files?.[0];
        if (!file) return;
        const limit = isVideo ? MAX_VIDEO_BYTES : MAX_IMAGE_BYTES;
        if (file.size > limit) {
            sysMsg(node, `文件过大：${(file.size / 1048576).toFixed(1)}MB，限制 ${isVideo ? "300" : "8"}MB`);
            return;
        }
        const ext = (file.name.split(".").pop() || "").toLowerCase();
        if (isVideo && !VIDEO_EXTS.includes(ext)) {
            sysMsg(node, `不支持的视频格式: ${ext}`);
            return;
        }
        try {
            sysMsg(node, `正在上传 ${file.name} …`);
            const path = await uploadFile(file);
            node._qwente.attachments.push({ name: file.name, path });
            sysMsg(node, `已附加 ${isVideo ? "视频" : "图片"}：${file.name}（发送时随消息提交）`);
        } catch (e) {
            sysMsg(node, `上传失败：${e.message}`);
        }
    };
    input.click();
}

function regenerate(node) {
    const st = node._qwente;
    if (st.pending) return;
    let hist = [];
    try { hist = JSON.parse(getWVal(node, "对话历史JSON") || "[]"); } catch { /* ignore */ }
    if (!Array.isArray(hist) || hist.length < 2) {
        sysMsg(node, "没有可重新生成的上一轮对话");
        return;
    }
    const lastUser = [...hist].reverse().find((m) => m.role === "user");
    // 移除最后一轮（user + assistant）
    while (hist.length && hist[hist.length - 1].role !== "user") hist.pop();
    if (hist.length) hist.pop();
    setW(node, "对话历史JSON", JSON.stringify(hist));
    st.history = hist;
    st.input.value = lastUser?.content ?? "";
    renderAll(node);
    sysMsg(node, "已回退最后一轮，点击发送或编辑后重新生成");
}

function exportHistory(node) {
    const data = getWVal(node, "对话历史JSON") || "[]";
    const blob = new Blob([data], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `qwente_chat_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
}

// 执行失败/中断时解除 pending 状态并提示
api.addEventListener("execution_error", (e) => {
    const node = app.graph.getNodeById(e.detail?.node_id);
    if (node?._qwente) {
        node._qwente.pending = false;
        renderAll(node);
        sysMsg(node, "❌ 执行出错：" + (e.detail?.exception_message || "未知错误"));
    }
});
api.addEventListener("execution_interrupted", () => {
    for (const node of app.graph._nodes ?? []) {
        if (node._qwente?.pending) {
            node._qwente.pending = false;
            renderAll(node);
            sysMsg(node, "⛔ 已中断");
        }
    }
});

// 注入样式
const style = document.createElement("style");
style.textContent = CSS;
document.head.appendChild(style);
