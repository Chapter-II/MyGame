#!/usr/bin/env python3
"""Generate a shareable multi-page project report DOCX (then convert to PDF)."""
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

OUT = Path(r"D:\hello\项目报告.pdf")
DOCX = Path(r"D:\hello\项目报告.docx")
FONT = "Microsoft YaHei"


def set_run(run, size=11, bold=False, color=(0x1A, 0x1F, 0x2E)):
    run.font.name = FONT
    run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(*color)


def para(doc, text, size=11, bold=False, color=(0x1A, 0x1F, 0x2E), after=6, before=0, indent=True, align=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.line_spacing = 1.25
    if indent:
        p.paragraph_format.first_line_indent = Cm(0.74)
    if align is not None:
        p.alignment = align
    run = p.add_run(text)
    set_run(run, size=size, bold=bold, color=color)
    return p


def h1(doc, text):
    return para(doc, text, size=14, bold=True, color=(0x1F, 0x3A, 0x8A), before=14, after=6, indent=False)


def h2(doc, text):
    return para(doc, text, size=12, bold=True, color=(0x2D, 0x4A, 0x6E), before=10, after=4, indent=False)


def bullets(doc, items):
    for item in items:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.2
        p.paragraph_format.left_indent = Cm(0.4)
        run = p.add_run("• " + item)
        set_run(run, size=10.5)


def main():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.0)
    sec.bottom_margin = Cm(2.0)
    sec.left_margin = Cm(2.2)
    sec.right_margin = Cm(2.2)

    # Cover-like header
    t = para(doc, "指挥官战术对抗", size=22, bold=True, indent=False, after=4)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s = para(doc, "基于自然语言指挥的千人战术对抗游戏 · 项目报告", size=13, color=(0x4A, 0x5A, 0x72), indent=False, after=2)
    s.alignment = WD_ALIGN_PARAGRAPH.CENTER
    m = para(
        doc,
        "技术栈：Python 3.12 + Pygame-ce　|　平台：Windows / Linux　|　版本：1.0.0（开发中）",
        size=10, color=(0x5B, 0x6B, 0x82), indent=False, after=12,
    )
    m.alignment = WD_ALIGN_PARAGRAPH.CENTER

    h1(doc, "一、项目背景与定位")
    para(
        doc,
        "本项目是一款俯视角、即时制的单人指挥官战术游戏。玩家扮演战场将领，通过鼠标、键盘、文字以及可选的语音下达宏观军令，"
        "在永久探索战争迷雾与可变地形中，指挥默认 500 对 500 名独立士兵完成侦察、编组、征兵、工程与作战，"
        "最终以击杀敌方将领决出胜负。目标单局节奏约 10 分钟。",
    )
    para(
        doc,
        "产品定位可概括为：可验证的自然语言指挥 + 一千名独立士兵 + 公平战争迷雾下的单人战术对抗。"
        "与传统 RTS 相比，本作降低逐兵微操负担；与抽象兵团模拟相比，每名士兵仍是真实存在、可独立移动与战斗的单位。",
    )
    para(
        doc,
        "V1.0 强调离线可玩：在没有网络或大模型服务时，仍可依靠本地中文规则解析与传统输入完成整局游戏。"
        "在线 DeepSeek 军令解析、本地 Whisper 语音与在线 ASR 均为可选增强能力。",
    )

    h1(doc, "二、核心功能")
    h2(doc, "2.1 战场与战争迷雾")
    bullets(doc, [
        "种子化地图：中央曲流河宽 4–12 格连续变化；草场、森林、沼泽、道路与六座对称村庄。",
        "180° 旋转对称保证双方路线与资源公平；种子可复现便于分享与调试。",
        "永久探索迷雾：部队经过如橡皮擦揭开地图；敌军进入已探索区即显示；未探索区不泄露真实地形。",
        "陆军不能无桥强渡；桥梁跨度与工程量按落点河宽计算，通行严格限制在桥面。",
    ])
    h2(doc, "2.2 兵种与工程")
    bullets(doc, [
        "将领 + 4 护卫；步兵、侦察兵、工兵、刺客、预备兵。",
        "村庄征兵与兵种训练；编组命名与底栏快捷选兵。",
        "架桥、造船、砍树开路、建造防御塔；工兵可自动驻塔转为守塔弓箭手。",
    ])
    h2(doc, "2.3 指挥输入")
    bullets(doc, [
        "中文规则命令解析（离线可用），支持复合指令切分与上下文建议补全。",
        "歧义最多提供 3 个候选解释；执行状态与失败原因始终可见。",
        "可选：DeepSeek 在线解析、本地 Whisper（按住 V）、OpenAI 兼容在线 ASR。",
        "断网路径：自动暂停 → R 重试 / O 离线 / Q 紧急存档退出。",
    ])
    h2(doc, "2.4 敌方 AI 与公平性")
    bullets(doc, [
        "战略层（约 2 秒）与战术层（约 0.25 秒）分离；难度三档。",
        "侦察、分兵、工程、护卫、刺杀、撤退等战术行为。",
        "AI 与界面仅消费 ObservationSnapshot，从数据边界保证公平，而非依赖 AI 自觉。",
    ])

    h1(doc, "三、系统架构")
    para(doc, "系统采用七层架构，输入与 AI 不得直接修改单位状态，只能提交结构化命令：")
    bullets(doc, [
        "表现层：Pygame 渲染、HUD、回放界面",
        "输入层：语音、文字、鼠标、键盘",
        "命令层：Command 协议、校验、队列与反馈",
        "AI 层：本地战略/战术 AI、大模型适配器",
        "感知层：战争迷雾与 ObservationSnapshot",
        "模拟层：固定时间步、单位/战斗/工程",
        "持久化：存档、回放、设置与统计",
    ])
    para(
        doc,
        "四条关键公共协议：Command（想做什么）、ObservationSnapshot（依法能看见什么）、"
        "GameEvent（已发生什么）、SaveSnapshot（如何完整恢复）。协议变更需双方评审。",
    )

    h1(doc, "四、技术难点与解决")
    h2(doc, "4.1 千人同屏性能")
    para(
        doc,
        "1000 个单位同时寻路与碰撞极易导致帧率不足。解决方案：固定时间步，渲染与模拟解耦；"
        "空间哈希做邻居查询；群体移动使用共享流场；远离交战区降低更新频率；地形预合成缓存，缩放时仅重绘可见区域。"
        "默认规模以 30 FPS 为门槛，并提供 soak 长压脚本。",
    )
    h2(doc, "4.2 桥面交战错位")
    para(
        doc,
        "单位在桥上交战时可能被普通碰撞挤入开放水面。解决：逐帧校验渡河合法性；"
        "桥梁通行区改为与贴图一致的定向矩形，单位中心与碰撞半径必须落在桥栏以内。",
    )
    h2(doc, "4.3 自动作战僵持")
    para(
        doc,
        "桥未完工时 AI 一度改派工兵造船，导致双方长期隔河对峙。解决：调整工程优先级，优先完成架桥，未完成前不切换造船。",
    )
    h2(doc, "4.4 语音与在线服务")
    para(
        doc,
        "语音采集与转写为可选路径：按住 V 或底栏按钮录音，静音与过短录音直接忽略；"
        "本地 faster-whisper 与在线 OpenAI 兼容 ASR 可切换，失败自动回落；"
        "DeepSeek 文字军令通过 local.env 注入密钥，启动时自动加载，不进入版本库。",
    )

    h1(doc, "五、工程与质量")
    bullets(doc, [
        "测试：单元 / 集成 / 性能 pytest；静态检查 ruff + mypy strict。",
        "存档：MessagePack + Zstandard，带版本魔数与原子替换。",
        "发行：PyInstaller 规格与 GitHub Actions Windows / Linux 打包冒烟。",
        "配置：兵种、地形、AI 参数数据驱动（TOML）；素材程序化可再生成。",
        "协作：双人主责模块 + 交叉评审 + 阶段集成版本。",
    ])

    h1(doc, "六、当前状态与后续计划")
    h2(doc, "6.1 已完成")
    bullets(doc, [
        "完整对局闭环：配置 → 对战 → 将领死亡结算 → 存档/回放。",
        "自然语言指挥链路与离线兜底；本地 AI 与信息隔离。",
        "工程设施、永久迷雾、像素战场与指挥台 HUD。",
        "语音输入模块完善；DeepSeek / 在线 ASR 可选接入。",
    ])
    h2(doc, "6.2 不足与后续")
    bullets(doc, [
        "新手教程与数值平衡仍需大量试玩。",
        "默认地图美术完成度与多机型性能复测。",
        "语音在无麦克风环境的体验与模型下载引导可再打磨。",
        "经济系统、真人联机、战役剧情明确不纳入本期范围。",
    ])

    h1(doc, "七、附录：如何运行")
    para(doc, "开发环境：", indent=False, bold=True)
    bullets(doc, [
        "python 3.12 创建虚拟环境后：pip install -e \".[dev]\"（可选语音：.[voice]）",
        "启动：.venv\\Scripts\\python -m mygame 或使用 启动游戏.bat",
        "在线 DeepSeek：在项目根目录 local.env 中配置 DEEPSEEK_API_KEY（勿提交仓库）",
        "在线语音 ASR：配置 SPEECH_API_KEY / SPEECH_API_URL（OpenAI 兼容）",
        "验证：pytest、ruff check、mypy src/mygame",
    ])
    para(
        doc,
        "本报告用于课程/项目对外说明，内容依据仓库规划文档、产品说明与工作日志整理。具体数值与平衡以可运行版本为准。",
        size=10, color=(0x5B, 0x6B, 0x82), indent=False, before=12,
    )

    doc.save(DOCX)
    print(f"wrote {DOCX}")


if __name__ == "__main__":
    main()
