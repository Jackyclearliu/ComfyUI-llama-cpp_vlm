import os

SKILLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "skills")


def _parse_skill_md(path):
    """解析 SKILL.md：--- frontmatter（name/description）--- + 正文。不引入 yaml 依赖。"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    meta = {}
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            for line in text[3:end].strip().splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    meta[k.strip()] = v.strip().strip('"').strip("'")
            body = text[end + 4:].strip()
    return meta, body


def scan_skills():
    """扫描 skills/ 目录，返回 {显示名: skill dict}。显示名格式：名称 [id]"""
    skills = {}
    if not os.path.isdir(SKILLS_DIR):
        return skills
    for entry in sorted(os.listdir(SKILLS_DIR)):
        skill_md = os.path.join(SKILLS_DIR, entry, "SKILL.md")
        if not os.path.isfile(skill_md):
            continue
        try:
            meta, body = _parse_skill_md(skill_md)
        except Exception as e:
            print(f"[QwenTE] 解析 skill 失败，已跳过 {entry}: {e}")
            continue
        name = meta.get("name", entry)
        skills[f"{name} [{entry}]"] = {
            "id": entry,
            "name": name,
            "description": meta.get("description", ""),
            "content": body,
        }
    return skills


class QwenTE_SkillLoader:
    @classmethod
    def INPUT_TYPES(s):
        skills = scan_skills()
        options = list(skills.keys()) or ["无可用skill"]
        return {"required": {"skill": (options,)}}

    RETURN_TYPES = ("QWEN_TE_SKILL",)
    RETURN_NAMES = ("skill加载器",)
    FUNCTION = "load"
    CATEGORY = "QwenTE"
    DESCRIPTION = (
        "从节点包 skills/ 目录加载技能（每个技能一个子目录，内含 SKILL.md：\n"
        "--- frontmatter（name/description）--- + 正文提示词指令）。\n"
        "输出连接 QwenTE_MultiTurnChat 的「skill加载器」输入；\n"
        "skills 目录内容变化后重启或刷新即可出现在下拉列表。"
    )

    @classmethod
    def VALIDATE_INPUTS(cls, skill, **kwargs):
        # skills 目录内容可能在服务启动后变化，放宽 combo 校验
        if skill not in scan_skills():
            return f"skill 不存在: {skill}"
        return True

    def load(self, skill):
        skills = scan_skills()
        if skill not in skills:
            raise ValueError(f"skill 不存在: {skill}（skills 目录: {SKILLS_DIR}）")
        return (skills[skill],)


NODE_CLASS_MAPPINGS = {
    "QwenTE_SkillLoader": QwenTE_SkillLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenTE_SkillLoader": "QwenTE Skill加载器",
}
