#!/usr/bin/env python3
"""Build course defense PPT for Commander Tactical Arena."""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(r"D:\hello")
OUT = ROOT / "答辩PPT.pptx"
ASSET_DIR = ROOT / "assets" / "generated"

BG = RGBColor(0x0B, 0x12, 0x20)
PANEL = RGBColor(0x15, 0x20, 0x33)
PANEL2 = RGBColor(0x1B, 0x2A, 0x42)
INK = RGBColor(0xE8, 0xEE, 0xF7)
MUTED = RGBColor(0x8B, 0x9B, 0xB4)
COBALT = RGBColor(0x3B, 0x82, 0xF6)
AMBER = RGBColor(0xD9, 0x77, 0x06)
CORAL = RGBColor(0xE8, 0x5D, 0x4C)
TEAL = RGBColor(0x2D, 0xD4, 0xBF)
LINE = RGBColor(0x2A, 0x3A, 0x55)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT = "Microsoft YaHei"
W, H = Inches(13.333), Inches(7.5)


def set_cjk(run, font_name=FONT):
    run.font.name = font_name
    rPr = run._r.get_or_add_rPr()
    successors = {
        "a:ea": ("a:cs", "a:sym", "a:hlinkClick", "a:hlinkMouseOver", "a:rtl", "a:extLst"),
        "a:cs": ("a:sym", "a:hlinkClick", "a:hlinkMouseOver", "a:rtl", "a:extLst"),
    }
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.insert_element_before(el, *successors[tag])
        el.set("typeface", font_name)


def style_run(run, size=16, bold=False, color=INK):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    set_cjk(run)


def add_text(slide, text, x, y, w, h, size=16, bold=False, color=INK, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    style_run(run, size=size, bold=bold, color=color)
    return box


def add_rect(slide, x, y, w, h, fill=PANEL, line=None, radius=False):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(shape_type, x, y, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(1)
    if radius:
        try:
            shp.adjustments[0] = 0.08
        except Exception:
            pass
    return shp


def add_footer(slide, page, total=13, label="小组作业答辩 · 指挥官战术对抗"):
    add_rect(slide, Inches(0.55), Inches(7.05), Inches(12.2), Pt(1), fill=LINE)
    add_text(slide, label, Inches(0.55), Inches(7.12), Inches(6), Inches(0.3), size=10, color=MUTED)
    add_text(slide, f"{page:02d} / {total:02d}", Inches(11.5), Inches(7.12), Inches(1.25), Inches(0.3), size=10, color=MUTED, align=PP_ALIGN.RIGHT)


def set_bg(slide, color=BG):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def card(slide, x, y, w, h, title, body, accent=COBALT, body_size=13):
    add_rect(slide, x, y, w, h, fill=PANEL, line=LINE, radius=True)
    add_rect(slide, x, y, Inches(0.08), h, fill=accent)
    add_text(slide, title, x + Inches(0.25), y + Inches(0.16), w - Inches(0.4), Inches(0.35), size=14, bold=True, color=INK)
    add_text(slide, body, x + Inches(0.25), y + Inches(0.55), w - Inches(0.4), h - Inches(0.7), size=body_size, color=MUTED)


def bullets(slide, items, x, y, w, h, size=14, color=INK):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(8)
        run = p.add_run()
        run.text = "▸  " + item
        style_run(run, size=size, color=color)
    return box


def cover_img():
    for name in ("village.png", "tower.png", "terrain_atlas.png"):
        p = ASSET_DIR / name
        if p.exists():
            return p
    return None


def crop_cover(src: Path, dst: Path, target=(1280, 720), blend=0.58) -> Path:
    im = Image.open(src).convert("RGB")
    tw, th = target
    sw, sh = im.size
    scale = max(tw / sw, th / sh)
    nw, nh = int(sw * scale), int(sh * scale)
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    left, top = (nw - tw) // 2, (nh - th) // 2
    im = im.crop((left, top, left + tw, top + th))
    dark = Image.new("RGB", im.size, (8, 14, 24))
    im = Image.blend(im, dark, blend)
    im.save(dst, quality=88)
    return dst


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def build():
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    blank = prs.slide_layouts[6]
    total = 13

    # 1 Cover
    s = prs.slides.add_slide(blank)
    set_bg(s)
    img = cover_img()
    tmp = ROOT / "assets" / "_defense_cover.jpg"
    if img:
        try:
            crop_cover(img, tmp, blend=0.62)
            s.shapes.add_picture(str(tmp), 0, 0, width=W, height=H)
        except Exception:
            pass
    add_rect(s, 0, 0, Inches(7.6), H, fill=RGBColor(0x0B, 0x12, 0x20))
    add_rect(s, Inches(7.6), 0, Inches(0.12), H, fill=COBALT)
    add_text(s, "COURSE PROJECT DEFENSE", Inches(0.7), Inches(1.35), Inches(5.5), Inches(0.35), size=12, bold=True, color=COBALT)
    add_text(s, "指挥官战术对抗", Inches(0.7), Inches(1.8), Inches(6.4), Inches(0.85), size=40, bold=True)
    add_text(s, "基于自然语言指挥的千人战术对抗游戏", Inches(0.7), Inches(2.75), Inches(6.4), Inches(0.45), size=16, color=MUTED)
    add_text(s, "Python + Pygame  ·  Windows / Linux  ·  单人对战 AI", Inches(0.7), Inches(3.35), Inches(6.4), Inches(0.35), size=13, color=TEAL)

    add_rect(s, Inches(0.7), Inches(4.15), Inches(6.3), Inches(1.55), fill=PANEL, line=LINE, radius=True)
    add_text(s, "小组成员", Inches(0.95), Inches(4.35), Inches(5.5), Inches(0.3), size=12, color=MUTED)
    add_text(s, "组员一 ｜ 模拟、性能与工程实现", Inches(0.95), Inches(4.75), Inches(5.6), Inches(0.35), size=14, bold=True)
    add_text(s, "组员二 ｜ 交互、地图与智能系统", Inches(0.95), Inches(5.2), Inches(5.6), Inches(0.35), size=14, bold=True)
    add_text(s, "2026 · 小组作业答辩", Inches(0.7), Inches(6.1), Inches(6.2), Inches(0.3), size=12, color=MUTED)
    notes(s, "P1 0:00-0:30：各位老师好，我们小组的项目是《指挥官战术对抗》。约10分钟介绍背景、设计、功能、难点与成果。")

    # 2 Agenda
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_text(s, "汇报提纲", Inches(0.55), Inches(0.45), Inches(12), Inches(0.5), size=30, bold=True)
    items = [
        ("01", "选题背景与目标", "为什么做、做成什么"),
        ("02", "系统设计与架构", "模块分层与关键协议"),
        ("03", "核心功能展示", "战场 / 指挥 / AI / 工程"),
        ("04", "技术难点与方案", "性能、公平性、离线可用"),
        ("05", "分工、成果与总结", "我们做了什么、还差什么"),
    ]
    y = Inches(1.35)
    for num, title, sub in items:
        add_rect(s, Inches(0.55), y, Inches(12.2), Inches(0.95), fill=PANEL, line=LINE, radius=True)
        add_text(s, num, Inches(0.85), y + Inches(0.22), Inches(0.8), Inches(0.5), size=22, bold=True, color=COBALT)
        add_text(s, title, Inches(1.8), y + Inches(0.18), Inches(3.5), Inches(0.4), size=17, bold=True)
        add_text(s, sub, Inches(5.5), y + Inches(0.25), Inches(6.8), Inches(0.4), size=14, color=MUTED)
        y += Inches(1.08)
    add_footer(s, 2, total)
    notes(s, "P2 0:30-0:50：五部分提纲——背景目标、架构、功能、难点、分工成果。手势轻点，不要逐条念。")

    # 3 Background
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_text(s, "选题背景：RTS 太“点”，大战略太“远”", Inches(0.55), Inches(0.4), Inches(12), Inches(0.5), size=28, bold=True)
    add_text(s, "我们想做一款“看得见一千名士兵、却只下宏观军令”的指挥官体验。", Inches(0.55), Inches(1.0), Inches(12), Inches(0.4), size=14, color=MUTED)

    cards = [
        ("痛点", "传统 RTS 微操负担重；\n抽象兵团数字缺乏战场临场感。", CORAL),
        ("机会", "自然语言/语音可降低操作门槛；\n固定时间步模拟可支撑千人规模。", COBALT),
        ("玩家价值", "当将领下军令，而不是当鼠标点小兵；\n信息战、工程与地形成为胜负关键。", TEAL),
    ]
    x = Inches(0.55)
    for title, body, acc in cards:
        card(s, x, Inches(1.55), Inches(3.95), Inches(1.2), title, body, accent=acc, body_size=13)
        x += Inches(4.15)

    stats = [
        ("500 vs 500", "默认独立士兵规模"),
        ("约 10 分钟", "单局节奏目标"),
        ("离线可玩", "无网/无 LLM 仍可完整对局"),
        ("双平台", "Windows / Linux"),
    ]
    x = Inches(0.55)
    for label, sub in stats:
        add_rect(s, x, Inches(3.05), Inches(2.95), Inches(1.35), fill=PANEL2, line=LINE, radius=True)
        add_text(s, label, x + Inches(0.2), Inches(3.25), Inches(2.5), Inches(0.45), size=20, bold=True, color=COBALT)
        add_text(s, sub, x + Inches(0.2), Inches(3.8), Inches(2.5), Inches(0.4), size=12, color=MUTED)
        x += Inches(3.1)

    add_rect(s, Inches(0.55), Inches(4.7), Inches(12.2), Inches(1.95), fill=PANEL, line=LINE, radius=True)
    add_text(s, "项目定位一句话", Inches(0.85), Inches(4.95), Inches(2.2), Inches(0.3), size=13, bold=True, color=AMBER)
    add_text(s, "可验证的自然语言指挥 + 一千名独立士兵 + 公平战争迷雾下的单人战术对抗。", Inches(0.85), Inches(5.4), Inches(11.5), Inches(0.4), size=16, color=INK)
    add_text(s, "技术路线：Python + Pygame-ce，固定时间步模拟，可选 DeepSeek / 本地语音。", Inches(0.85), Inches(5.95), Inches(11.5), Inches(0.35), size=13, color=MUTED)
    add_footer(s, 3, total)
    notes(s, "P3 0:50-1:50：痛点/机会→定位一句话。下军令不是点小兵；500v500、约10分钟、离线可玩。")

    # 4 Goals
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_text(s, "项目目标：V1.0 必须完成一局完整对战", Inches(0.55), Inches(0.4), Inches(12), Inches(0.5), size=28, bold=True)
    left = [
        "500×500 独立单位，固定时间步模拟",
        "将领、护卫与步兵/侦察/工兵/刺客/预备兵",
        "永久探索战争迷雾 + 双方信息隔离",
        "河流硬阻挡，架桥/造船/建塔/开路",
        "文字/鼠标/键盘指挥，语音与大模型可选",
        "本地 AI、存档回放、结算统计、双平台运行",
    ]
    right = [
        "不做：经济系统、长线养成、真人联机",
        "不做：战役剧情、多语言、复杂科技树",
        "约束：默认规模最低 30 FPS",
        "约束：断网可离线完成整局",
        "约束：AI 只能读阵营可见快照",
        "约束：第三方素材许可证可追溯",
    ]
    add_rect(s, Inches(0.55), Inches(1.25), Inches(6.0), Inches(5.3), fill=PANEL, line=LINE, radius=True)
    add_text(s, "必须做到", Inches(0.85), Inches(1.45), Inches(3), Inches(0.4), size=16, bold=True, color=TEAL)
    bullets(s, left, Inches(0.85), Inches(2.0), Inches(5.4), Inches(4.2), size=14)
    add_rect(s, Inches(6.8), Inches(1.25), Inches(5.95), Inches(5.3), fill=PANEL, line=LINE, radius=True)
    add_text(s, "明确不做 / 约束", Inches(7.1), Inches(1.45), Inches(3), Inches(0.4), size=16, bold=True, color=AMBER)
    bullets(s, right, Inches(7.1), Inches(2.0), Inches(5.3), Inches(4.2), size=14)
    add_footer(s, 4, total)
    notes(s, "P4 1:50-2:50：必须做到 vs 不做/约束。范围克制是工程判断，不是能力不足。")

    # 5 Architecture
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_text(s, "系统架构：七层分层，命令单向驱动模拟", Inches(0.55), Inches(0.4), Inches(12), Inches(0.5), size=28, bold=True)
    add_text(s, "输入与 AI 不能直接改单位，只能提交结构化命令；渲染只读状态。", Inches(0.55), Inches(1.0), Inches(12), Inches(0.4), size=14, color=MUTED)

    layers = [
        ("表现层", "Pygame 渲染 / HUD / 回放", COBALT),
        ("输入层", "语音 · 文字 · 鼠标 · 键盘", COBALT),
        ("命令层", "Command · 校验 · 队列 · 反馈", TEAL),
        ("AI 层", "战略 / 战术 · 大模型适配", TEAL),
        ("感知层", "战争迷雾 · ObservationSnapshot", AMBER),
        ("模拟层", "固定时间步 · 单位 / 战斗 / 工程", CORAL),
        ("持久化", "存档 · 回放 · 设置 · 统计", MUTED),
    ]
    y = Inches(1.45)
    for name, desc, color in layers:
        add_rect(s, Inches(0.55), y, Inches(7.4), Inches(0.7), fill=PANEL, line=LINE, radius=True)
        add_rect(s, Inches(0.55), y, Inches(0.1), Inches(0.7), fill=color)
        add_text(s, name, Inches(0.85), y + Inches(0.16), Inches(1.4), Inches(0.4), size=14, bold=True, color=color)
        add_text(s, desc, Inches(2.4), y + Inches(0.16), Inches(5.2), Inches(0.4), size=13, color=MUTED)
        y += Inches(0.78)

    card(s, Inches(8.2), Inches(1.45), Inches(4.55), Inches(2.5), "四条关键协议", "Command：想做什么\nObservationSnapshot：依法能看见什么\nGameEvent：已发生什么\nSaveSnapshot：如何完整恢复", accent=COBALT, body_size=13)
    card(s, Inches(8.2), Inches(4.15), Inches(4.55), Inches(2.45), "依赖方向（防作弊）", "模拟层不依赖 UI / 网络 / 麦克风\nAI 只读阵营快照\n回放只记结构化命令\n公共协议变更需双方评审", accent=AMBER, body_size=13)
    add_footer(s, 5, total)
    notes(s, "P5 2:50-4:10：七层+四协议。命令单向驱动；公平靠 ObservationSnapshot 数据边界。")

    # 6 Core feature - battlefield
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_text(s, "核心功能 ①：可验证公平的战场与迷雾", Inches(0.55), Inches(0.4), Inches(12), Inches(0.5), size=28, bold=True)
    feats = [
        ("地图", "连续曲流河宽 4–12 格；草场/森林/沼泽/道路；六座对称村庄；种子可复现、180° 公平。", COBALT),
        ("迷雾", "永久探索：部队经过像橡皮擦；敌军进入已探索区即显示；未探索区不泄露真实地形。", TEAL),
        ("工程", "架桥按河宽算工程量；造船渡河；砍树开路；建塔后工兵自动驻塔转弓箭手。", AMBER),
        ("表现", "像素战场最近邻缩放；远景战术符号 / 近景人物贴图；中世纪指挥台 HUD。", CORAL),
    ]
    y = Inches(1.25)
    for title, body, acc in feats:
        add_rect(s, Inches(0.55), y, Inches(12.2), Inches(1.2), fill=PANEL, line=LINE, radius=True)
        add_rect(s, Inches(0.55), y, Inches(0.1), Inches(1.2), fill=acc)
        add_text(s, title, Inches(0.9), y + Inches(0.35), Inches(1.4), Inches(0.4), size=16, bold=True, color=acc)
        add_text(s, body, Inches(2.5), y + Inches(0.3), Inches(9.9), Inches(0.7), size=14, color=INK)
        y += Inches(1.35)
    add_footer(s, 6, total)
    notes(s, "P6 4:10-5:10：地图/迷雾/工程/表现四条。强调未探索区不泄露真实地形。")

    # 7 Command system (highlight)
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_text(s, "核心功能 ②：自然语言指挥（项目最大亮点）", Inches(0.55), Inches(0.4), Inches(12), Inches(0.5), size=28, bold=True)
    add_text(s, "语音/文字 → 意图解析 → 结构化命令 → 合法性校验 → 执行与可视反馈", Inches(0.55), Inches(1.0), Inches(12), Inches(0.4), size=14, color=MUTED)

    steps = [
        ("1 输入", "语音 / 文字 / 鼠标键盘"),
        ("2 解析", "本地规则 / 可选 DeepSeek"),
        ("3 校验", "可见信息与合法性"),
        ("4 执行", "模拟层任务分派"),
        ("5 反馈", "成功/失败原因可见"),
    ]
    x = Inches(0.55)
    for title, body in steps:
        add_rect(s, x, Inches(1.55), Inches(2.3), Inches(1.35), fill=PANEL, line=LINE, radius=True)
        add_text(s, title, x + Inches(0.15), Inches(1.7), Inches(2.0), Inches(0.35), size=14, bold=True, color=COBALT)
        add_text(s, body, x + Inches(0.15), Inches(2.15), Inches(2.0), Inches(0.5), size=12, color=MUTED)
        if x < Inches(10):
            add_text(s, "→", x + Inches(2.25), Inches(1.95), Inches(0.35), Inches(0.4), size=18, bold=True, color=TEAL)
        x += Inches(2.55)

    add_rect(s, Inches(0.55), Inches(3.2), Inches(7.5), Inches(3.4), fill=PANEL, line=LINE, radius=True)
    add_text(s, "示例军令", Inches(0.85), Inches(3.4), Inches(2), Inches(0.35), size=15, bold=True, color=TEAL)
    ex = [
        "第一侦察队前往北部中央",
        "20名初始兵分化为工兵",
        "工兵队在中央架桥",
        "第1组集火敌方将领",
        "第6组进入防御塔",
    ]
    bullets(s, ex, Inches(0.85), Inches(3.9), Inches(6.9), Inches(2.5), size=14)

    card(s, Inches(8.25), Inches(3.2), Inches(4.5), Inches(3.4), "工程细节", "· 中文 IME 候选窗定位\n· 复合指令自动切分\n· 上下文命令建议补全\n· 歧义最多 3 个候选\n· 断网：R 重试 / O 离线 / Q 紧急存档\n· 指令书说明 AI 能力边界", accent=AMBER, body_size=13)
    add_footer(s, 7, total)
    notes(s, "P7 5:10-6:20 最大亮点：输入→解析→校验→执行→反馈。断网可降级离线。语速稍慢。")

    # 8 AI & fairness
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_text(s, "核心功能 ③：本地 AI 与信息隔离", Inches(0.55), Inches(0.4), Inches(12), Inches(0.5), size=28, bold=True)

    card(s, Inches(0.55), Inches(1.35), Inches(4.0), Inches(2.7), "双层 AI", "战略层 ~2s：目标、分兵、工程、征召\n战术层 ~0.25s：接敌、走位、护卫、撤退\n难度三档；自动作战可开关，手动随时覆盖", accent=COBALT, body_size=13)
    card(s, Inches(4.75), Inches(1.35), Inches(4.0), Inches(2.7), "信息隔离怎么做", "完整战场 → 阵营过滤 → Snapshot → AI/UI\n未探索区不显示真实地形\n寻路缓存只含己方已知桥\n统计与音效不做侧信道泄密", accent=TEAL, body_size=13)
    card(s, Inches(8.95), Inches(1.35), Inches(3.8), Inches(2.7), "战术行为", "侦察开图绕翼 · 低血后撤\n刺客集火将领 · 护卫拦截\n将领阵亡后进入拼命模式\n工兵按局势：桥 → 船 → 塔", accent=AMBER, body_size=13)
    add_rect(s, Inches(0.55), Inches(4.35), Inches(12.2), Inches(2.25), fill=PANEL2, line=LINE, radius=True)
    add_text(s, "公平性设计要点", Inches(0.85), Inches(4.55), Inches(3), Inches(0.35), size=14, bold=True, color=TEAL)
    bullets(s, [
        "AI 不读完整战场对象、未发现坐标或对方秘密命令",
        "界面不通过计数、音效泄露迷雾外实时敌情",
        "公平靠数据边界，而不是让 AI“假装看不见”",
    ], Inches(0.85), Inches(5.05), Inches(11.5), Inches(1.3), size=14)
    add_footer(s, 8, total)
    notes(s, "P8 6:20-7:10：双层AI+隔离+战术行为。收束句：公平靠接口边界，不靠AI假装看不见。")

    # 9 Challenges
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_text(s, "技术难点：三个我们真正踩过的坑", Inches(0.55), Inches(0.4), Inches(12), Inches(0.5), size=28, bold=True)
    rows = [
        ("千人性能", "1000 单位寻路/碰撞导致卡顿", "固定时间步 + 空间哈希 + 流场共享路径 + 分频更新 + 地形缓存"),
        ("桥面拥堵错位", "桥上交战被碰撞推入开放水面", "逐帧渡河合法性校验 + 定向矩形桥面通行区约束"),
        ("自动作战僵持", "桥未完工时改派造船，双方隔河挂机", "AI 工程优先级：优先架桥，未完成不切换造船"),
        ("高分屏模糊", "文字与像素战场放大后发虚", "SCALED + 最近邻战场缩放 + 文字按最终分辨率重绘"),
        ("断网打断体验", "大模型/语音失败导致无法继续", "本地规则解析器兜底 + 暂停/重试/离线/紧急存档"),
    ]
    y = Inches(1.2)
    # header
    add_rect(s, Inches(0.55), y, Inches(2.6), Inches(0.5), fill=COBALT)
    add_rect(s, Inches(3.15), y, Inches(4.2), Inches(0.5), fill=COBALT)
    add_rect(s, Inches(7.35), y, Inches(5.4), Inches(0.5), fill=COBALT)
    add_text(s, "难点", Inches(0.7), y + Inches(0.1), Inches(2.2), Inches(0.35), size=13, bold=True, color=WHITE)
    add_text(s, "现象", Inches(3.3), y + Inches(0.1), Inches(3.8), Inches(0.35), size=13, bold=True, color=WHITE)
    add_text(s, "解决方案", Inches(7.5), y + Inches(0.1), Inches(5), Inches(0.35), size=13, bold=True, color=WHITE)
    y += Inches(0.5)
    for i, (a, b, c) in enumerate(rows):
        bg = PANEL if i % 2 == 0 else PANEL2
        add_rect(s, Inches(0.55), y, Inches(2.6), Inches(0.95), fill=bg, line=LINE)
        add_rect(s, Inches(3.15), y, Inches(4.2), Inches(0.95), fill=bg, line=LINE)
        add_rect(s, Inches(7.35), y, Inches(5.4), Inches(0.95), fill=bg, line=LINE)
        add_text(s, a, Inches(0.7), y + Inches(0.28), Inches(2.3), Inches(0.5), size=13, bold=True, color=CORAL)
        add_text(s, b, Inches(3.3), y + Inches(0.22), Inches(3.9), Inches(0.6), size=12, color=MUTED)
        add_text(s, c, Inches(7.5), y + Inches(0.18), Inches(5.1), Inches(0.7), size=12, color=INK)
        y += Inches(0.95)
    add_footer(s, 9, total)
    notes(s, "P9 7:10-8:20 重点页：深挖千人性能+桥面错位/自动作战僵持，其余一句带过。")

    # 10 Teamwork
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_text(s, "团队协作：主责模块 + 交叉评审 + 阶段集成", Inches(0.55), Inches(0.4), Inches(12), Inches(0.5), size=28, bold=True)
    card(s, Inches(0.55), Inches(1.25), Inches(6.0), Inches(2.85), "组员一｜模拟与性能", "· 固定时间步、实体模型与配置\n· 移动 / 流场寻路 / 弹性碰撞\n· 战斗结算、兵种与工程设施\n· 命令执行器与模拟合法性\n· 存档回放底层与性能基准\n· 测试：确定性、碰撞、500v500 长压", accent=COBALT, body_size=13)
    card(s, Inches(6.8), Inches(1.25), Inches(5.95), Inches(2.85), "组员二｜交互与智能", "· Pygame 外壳、场景与摄像机\n· HUD、菜单、缩略图与教程入口\n· 迷雾感知、命令解析与指挥面板\n· 本地战略 AI 与难度配置\n· 语音/大模型适配与离线降级\n· 测试：命令解析、信息泄漏、地图公平", accent=TEAL, body_size=13)
    add_rect(s, Inches(0.55), Inches(4.4), Inches(12.2), Inches(2.2), fill=PANEL2, line=LINE, radius=True)
    add_text(s, "协作机制", Inches(0.85), Inches(4.6), Inches(1.5), Inches(0.35), size=14, bold=True, color=COBALT)
    bullets(s, [
        "主责模块清晰，避免两人同时改核心文件",
        "公共协议 / 存档格式 / 主循环变更需双方评审",
        "每个内部阶段都产出可运行集成版本",
        "共同负责：玩法规则、公平性审计、跨平台发布",
    ], Inches(0.85), Inches(5.05), Inches(11.5), Inches(1.4), size=14)
    add_footer(s, 10, total)
    notes(s, "P10 8:20-8:50：主责+交叉评审。成员A/B换成真名。不要展开每个文件。")

    # 11 Results
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_text(s, "成果与验证：可完整游玩，且有自动化检查", Inches(0.55), Inches(0.4), Inches(12), Inches(0.5), size=28, bold=True)
    stats = [
        ("完整对局", "配置 → 对战 → 结算可闭环", TEAL),
        ("测试", "单元 / 集成 / 性能 pytest", COBALT),
        ("静态检查", "ruff + mypy strict", AMBER),
        ("长压", "benchmark_soak 长时间模拟", CORAL),
    ]
    x = Inches(0.55)
    for title, body, acc in stats:
        add_rect(s, x, Inches(1.25), Inches(2.95), Inches(1.5), fill=PANEL, line=LINE, radius=True)
        add_text(s, title, x + Inches(0.2), Inches(1.45), Inches(2.5), Inches(0.4), size=16, bold=True, color=acc)
        add_text(s, body, x + Inches(0.2), Inches(1.95), Inches(2.5), Inches(0.55), size=12, color=MUTED)
        x += Inches(3.1)

    add_rect(s, Inches(0.55), Inches(3.05), Inches(12.2), Inches(3.5), fill=PANEL, line=LINE, radius=True)
    add_text(s, "可演示路径（建议现场跑一条）", Inches(0.85), Inches(3.25), Inches(5), Inches(0.35), size=15, bold=True, color=COBALT)
    demo = [
        "1. 进入默认对称地图，框选初始部队，分化为步兵/工兵/侦察",
        "2. 文字军令：“工兵队在中央架桥”，观察跨度预览与施工进度",
        "3. 右键村庄征兵，底栏训练兵种；编组后下达攻击移动",
        "4. 开启自动作战，再用鼠标手动覆盖，验证人机指令优先级",
        "5. F5 存档 / F9 载入；将领阵亡进入结算统计",
    ]
    bullets(s, demo, Inches(0.85), Inches(3.8), Inches(11.5), Inches(2.5), size=14)
    add_footer(s, 11, total)
    notes(s, "P11 8:50-9:30：闭环+测试+双平台。一句话点演示路径：架桥或自然语言指令。")

    # 12 Summary
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_text(s, "总结：做了什么，还差什么", Inches(0.55), Inches(0.4), Inches(12), Inches(0.5), size=28, bold=True)
    card(s, Inches(0.55), Inches(1.25), Inches(6.0), Inches(2.7), "我们完成的", "· 千人级独立模拟与对战闭环\n· 自然语言指挥 + 离线兜底\n· 永久迷雾与信息隔离 AI\n· 桥/船/塔工程 · 存档回放结算\n· 双平台打包 · 测试与静态检查", accent=TEAL, body_size=14)
    card(s, Inches(6.8), Inches(1.25), Inches(5.95), Inches(2.7), "不足与后续", "· 教程与数值平衡仍需试玩\n· 语音在无麦克风环境验证不足\n· 默认地图美术与多机型性能复测\n· 经济/联机/剧情明确不进本期", accent=AMBER, body_size=14)
    add_rect(s, Inches(0.55), Inches(4.25), Inches(12.2), Inches(2.35), fill=PANEL2, line=LINE, radius=True)
    add_text(s, "一句话总结", Inches(0.85), Inches(4.5), Inches(1.8), Inches(0.35), size=13, bold=True, color=COBALT)
    add_text(s, "我们交付的不是“按钮很多的 RTS”，而是一套可验证、可离线、可复盘的指挥官式对战原型。", Inches(0.85), Inches(4.95), Inches(11.5), Inches(0.5), size=16, color=INK)
    add_text(s, "下一步优先：教程打磨 → 平衡试玩 → 多机型性能复测 → 发布验收。", Inches(0.85), Inches(5.6), Inches(11.5), Inches(0.35), size=14, color=MUTED)
    add_text(s, "感谢各位老师指导。", Inches(0.85), Inches(6.1), Inches(11.5), Inches(0.35), size=14, bold=True, color=TEAL)
    add_footer(s, 12, total)
    notes(s, "P12 9:30-9:50：诚实说不足，再收一句话总结。不要展开新话题。")

    # 13 Thanks
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_text(s, "感谢聆听", Inches(0.55), Inches(2.4), Inches(12), Inches(0.7), size=40, bold=True)
    add_text(s, "指挥官战术对抗 · 小组作业答辩", Inches(0.55), Inches(3.3), Inches(12), Inches(0.45), size=18, color=MUTED)
    add_text(s, "欢迎老师提问 · 演示 / 架构 / 公平性 / 性能均可展开", Inches(0.55), Inches(4.0), Inches(12), Inches(0.4), size=14, color=TEAL)
    add_footer(s, 13, total)
    notes(s, "P13 9:50-10:00：停顿半秒后致谢，进入问答。可先演示再提问。")

    prs.save(OUT)
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
