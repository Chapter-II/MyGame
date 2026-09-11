#!/usr/bin/env python3
"""Generate one-page project summary DOCX for Commander Tactical Arena."""
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

OUT = Path(r"D:\hello\项目汇报摘要.docx")
FONT = "Microsoft YaHei"


def set_run(run, size=9.5, bold=False, color=(0x1A, 0x1F, 0x2E)):
    run.font.name = FONT
    run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(*color)


def add_para(doc, text, size=9.5, bold=False, color=(0x1A, 0x1F, 0x2E), space_after=2, space_before=0):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.line_spacing = 1.05
    run = p.add_run(text)
    set_run(run, size=size, bold=bold, color=color)
    return p


def add_heading(doc, text):
    return add_para(doc, text, size=10, bold=True, color=(0x1F, 0x3A, 0x8A), space_after=2, space_before=4)


def add_bullets(doc, items):
    for item in items:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.line_spacing = 1.02
        p.paragraph_format.left_indent = Cm(0.3)
        run = p.add_run("• " + item)
        set_run(run, size=9)


def main():
    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(1.0)
    section.bottom_margin = Cm(0.9)
    section.left_margin = Cm(1.3)
    section.right_margin = Cm(1.3)

    add_para(doc, "指挥官战术对抗 · 一页式项目汇报摘要", size=14, bold=True, space_after=1)
    add_para(
        doc,
        "版本 1.0.0（开发中） | Windows / Linux | Python + Pygame-ce | 双人团队 | 2026-09-09",
        size=8,
        color=(0x5B, 0x6B, 0x82),
        space_after=4,
    )

    add_heading(doc, "一、项目背景")
    add_para(
        doc,
        "面向中文策略玩家的单人指挥官战术游戏：玩家以将领身份，用鼠标/键盘/文字/可选语音统筹 500×500 独立士兵，"
        "在永久探索战争迷雾与可变地形中完成侦察、编组、征兵、工程与作战，约 10 分钟一局，击杀敌方将领决胜。"
        "V1.0 强调离线可完整游玩——无网或无大模型时仍可依靠本地规则解析与传统输入完成对局。",
        size=9,
        space_after=2,
    )

    add_heading(doc, "二、关键进展")
    add_bullets(
        doc,
        [
            "核心闭环：配置 → 兵种分化 → 迷雾侦察 → 村庄征兵 → 工程交战 → 结算 / 存档回放。",
            "战场地图：连续曲流（河宽 4–12 格）、草场/森林/沼泽/道路、六座对称村庄；种子复现、180° 公平。",
            "兵种工程：将领+护卫、步兵/侦察/工兵/刺客/预备；架桥、造船、开路、防御塔（工兵驻塔转弓箭手）。",
            "指挥系统：中文规则解析、复合指令、建议补全、歧义候选；可选语音与 DeepSeek；断网暂停与离线降级。",
            "敌方 AI：战略/战术双层、三档难度、仅读阵营可见信息；自动作战可开关，手动命令可覆盖。",
            "体验与基建：像素战场缩放、永久迷雾、双尺度单位、指挥台 HUD、特效结算；测试 + Zstd 存档 + 双平台打包。",
        ]
    )

    add_heading(doc, "三、主要风险与应对")
    add_bullets(
        doc,
        [
            "千人性能：性能原型优先，空间划分/流场/分频更新；默认规模目标 ≥30 FPS，含长压 benchmark。",
            "窄口/桥面拥堵：弹性碰撞、局部绕行、桥栏通行区约束，避免桥上交战被推入水面。",
            "AI 延迟断网：异步请求、自动暂停；R 重试一次 / O 切离线 / Q 紧急存档退出。",
            "语义误读与公平：结构化校验+可见反馈+歧义候选；AI/UI 强制 ObservationSnapshot，未探索区不泄密。",
            "范围与发布：按阶段验收；素材许可证台账；语音依赖不进基础发行包。",
        ]
    )

    add_heading(doc, "四、下一步计划")
    add_bullets(
        doc,
        [
            "阶段 8 体验完善：教程（选兵→工程→决胜）、平衡试玩、操作反馈与分辨率适配。",
            "阶段 9 V1.0 验收：Win/Linux 全功能、500v500 长压、信息隔离审计、随机图公平、存回放一致、许可证核查。",
            "交付：可运行构建、发布说明与素材署名；经济/联机/多语言仅预留架构，不纳入 V1.0。",
            "协作：双人主责模块 + 交叉评审；公共协议（Command/Snapshot/Event/Save）变更需双方确认。",
        ]
    )

    add_para(
        doc,
        "当前状态：核心对战与指挥闭环已可完整游玩，重心转向体验完善、跨平台验收与 V1.0 发布准备。",
        size=9.5,
        bold=True,
        color=(0x1F, 0x3A, 0x8A),
        space_before=4,
        space_after=0,
    )

    doc.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
