import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "qwente.chat",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "QwenTE_MultiTurnChat") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            // 只读显示部件放在最前，与示例工作流的 widgets_values 顺序对齐
            const w = this.addWidget("text", "对话显示", "", () => {}, { multiline: true });
            if (w.inputEl) w.inputEl.readOnly = true;
            this.widgets.unshift(this.widgets.pop());
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const w = this.widgets?.find((x) => x.name === "对话显示");
            if (w && message?.text?.[0] != null) {
                w.value = message.text[0];
            }
        };
    },
});
