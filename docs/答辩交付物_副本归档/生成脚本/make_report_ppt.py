#!/usr/bin/env python3
"""Build project status PPT for Commander Tactical Arena."""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(r"D:\hello")
OUT = ROOT / "项目汇报.pptx"
ASSET_DIR = ROOT / "assets" / "generated"

# Palette
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


def add_text(slide, text, x, y, w, h, size=16, bold=False, color=INK, align=PP_ALIGN.LEFT, valign_center=False):
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


def add_footer(slide, page, total=9):
    add_rect(slide, Inches(0.55), Inches(7.05), Inches(12.2), Pt(1), fill=LINE)
    add_text(slide, "指挥官战术对抗 · 项目汇报", Inches(0.55), Inches(7.12), Inches(5), Inches(0.3), size=10, color=MUTED)
    add_text(slide, f"{page:02d} / {total:02d}", Inches(11.5), Inches(7.12), Inches(1.25), Inches(0.3), size=10, color=MUTED, align=PP_ALIGN.RIGHT)


def set_bg(slide, color=BG):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def card(slide, x, y, w, h, title, body, accent=COBALT):
    add_rect(slide, x, y, w, h, fill=PANEL, line=LINE, radius=True)
    add_rect(slide, x, y, Inches(0.08), h, fill=accent)
    add_text(slide, title, x + Inches(0.28), y + Inches(0.18), w - Inches(0.4), Inches(0.35), size=15, bold=True, color=INK)
    add_text(slide, body, x + Inches(0.28), y + Inches(0.58), w - Inches(0.45), h - Inches(0.7), size=12, color=MUTED)


def bullet_block(slide, items, x, y, w, h, size=14, color=INK, accent=None):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(8)
        run = p.add_run()
        run.text = "▸  " + item
        style_run(run, size=size, color=color)
    return box


def cover_image_path():
    v = ASSET_DIR / "village.png"
    if v.exists():
        return v
    t = ASSET_DIR / "terrain_atlas.png"
    if t.exists():
        return t
    menu = ROOT / "main_menu.png"
    if menu.exists() and menu.stat().st_size > 2000:
        return menu
    return None


def crop_cover(src: Path, dst: Path, target=(1280, 720)) -> Path:
    im = Image.open(src).convert("RGB")
    tw, th = target
    # cover-fit
    sw, sh = im.size
    scale = max(tw / sw, th / sh)
    nw, nh = int(sw * scale), int(sh * scale)
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    im = im.crop((left, top, left + tw, top + th))
    # darken for text overlay
    dark = Image.new("RGB", im.size, (8, 14, 24))
    im = Image.blend(im, dark, 0.55)
    im.save(dst, quality=88)
    return dst


def build():
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    blank = prs.slide_layouts[6]
    total = 9

    # --- Slide 1 Cover ---
    s = prs.slides.add_slide(blank)
    set_bg(s)
    img = cover_image_path()
    tmp = ROOT / "assets" / "_report_cover.jpg"
    if img:
        try:
            crop_cover(img, tmp)
            s.shapes.add_picture(str(tmp), 0, 0, width=W, height=H)
        except Exception:
            pass
    # left gradient panel
    add_rect(s, 0, 0, Inches(7.4), H, fill=RGBColor(0x0B, 0x12, 0x20))
    add_rect(s, Inches(7.4), 0, Inches(0.15), H, fill=COBALT)
    add_text(s, "PROJECT STATUS REPORT", Inches(0.7), Inches(1.6), Inches(5.5), Inches(0.35), size=12, bold=True, color=COBALT)
    add_text(s, "指挥官战术对抗", Inches(0.7), Inches(2.05), Inches(6.2), Inches(0.8), size=40, bold=True, color=INK)
    add_text(s, "俯视角 · 千人独立模拟 · 自然语言指挥的单人战术游戏", Inches(0.7), Inches(2.95), Inches(6.2), Inches(0.5), size=16, color=MUTED)
    add_text(s, "背景 · 关键进展 · 风险 · 下一步计划", Inches(0.7), Inches(3.55), Inches(6), Inches(0.4), size=14, color=TEAL)
    # meta chips
    chips = [
        ("版本", "1.0.0 开发中"),
        ("平台", "Win / Linux"),
        ("团队", "2 人"),
        ("日期", "2026-09-09"),
    ]
    cx = Inches(0.7)
    for label, val in chips:
        add_rect(s, cx, Inches(4.5), Inches(1.45), Inches(0.85), fill=PANEL, line=LINE, radius=True)
        add_text(s, label, cx + Inches(0.15), Inches(4.62), Inches(1.15), Inches(0.28), size=11, color=MUTED)
        add_text(s, val, cx + Inches(0.15), Inches(4.92), Inches(1.2), Inches(0.3), size=12, bold=True, color=INK)
        cx += Inches(1.55)
    add_text(s, "来源：myplan.md / PRODUCT.md / CHANGELOG.md / team-collaboration-plan.md", Inches(0.7), Inches(6.7), Inches(6.5), Inches(0.3), size=10, color=MUTED)

    # --- Slide 2 Background ---
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_text(s, "项目背景：一局 10 分钟的千人战场指挥", Inches(0.55), Inches(0.4), Inches(12), Inches(0.5), size=28, bold=True)
    add_text(s, "让玩家以将领身份统筹真实独立的士兵，而不是操作抽象兵团数字。", Inches(0.55), Inches(1.0), Inches(12), Inches(0.4), size=14, color=MUTED)

    stats = [
        ("500 × 2", "默认双方独立单位", COBALT),
        ("~10 min", "目标对局节奏", TEAL),
        ("30+ FPS", "默认规模帧率门槛", AMBER),
        ("离线可玩", "无网 / 无 LLM 完整对局", CORAL),
    ]
    x = Inches(0.55)
    for label, sub, accent in stats:
        add_rect(s, x, Inches(1.55), Inches(2.95), Inches(1.35), fill=PANEL, line=LINE, radius=True)
        add_text(s, label, x + Inches(0.2), Inches(1.7), Inches(2.5), Inches(0.55), size=26, bold=True, color=accent)
        add_text(s, sub, x + Inches(0.2), Inches(2.3), Inches(2.5), Inches(0.4), size=12, color=MUTED)
        x += Inches(3.1)

    card(s, Inches(0.55), Inches(3.2), Inches(3.95), Inches(1.85), "核心体验", "宏观指挥：士兵接敌自动战斗\n自然语言：语音/文字 → 结构化军令\n信息战争：双方独立永久探索迷雾\n地形工程：架桥 · 造船 · 建塔 · 开路", accent=COBALT)
    card(s, Inches(4.7), Inches(3.2), Inches(3.95), Inches(1.85), "产品定位", "可验证的自然语言指挥\n一千名独立士兵公平交战\n失败可恢复：歧义、断网、缺设备均有下一步\n品牌气质：冷峻、克制、可信的夜间军令台", accent=TEAL)
    card(s, Inches(8.85), Inches(3.2), Inches(3.95), Inches(1.85), "V1.0 边界内", "单人 vs 本地 AI\n完整兵种 / 村庄 / 工程\n设置、教程、统计、结算、存档回放\nWin + Linux 窗口化与全屏", accent=AMBER)
    add_text(s, "定位来源：PRODUCT.md / myplan.md", Inches(0.55), Inches(5.3), Inches(8), Inches(0.3), size=11, color=MUTED)
    add_footer(s, 2, total)

    # --- Slide 3 Positioning ---
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_text(s, "产品主张：可验证的指挥，而不是按钮堆砌", Inches(0.55), Inches(0.4), Inches(12), Inches(0.5), size=28, bold=True)
    add_text(s, "每条命令都必须能看见理解结果、执行状态或失败原因。", Inches(0.55), Inches(1.0), Inches(12), Inches(0.4), size=14, color=MUTED)

    principles = [
        ("01", "战况优先", "地图、可见敌情与当前任务永远拥有最高视觉优先级。"),
        ("02", "命令可追溯", "识别文本 → 结构化结果 → 合法性 → 执行状态连续反馈。"),
        ("03", "公平可证明", "界面、音效、统计与 AI 只消费阵营可见信息。"),
        ("04", "熟悉而高效", "策略玩家熟悉的框选、编组、攻击移动与摄像机。"),
        ("05", "失败可恢复", "无效命令、断网、设备缺失、存档不兼容都给出下一步。"),
    ]
    y = Inches(1.55)
    for num, title, body in principles:
        add_rect(s, Inches(0.55), y, Inches(12.2), Inches(0.95), fill=PANEL, line=LINE, radius=True)
        add_text(s, num, Inches(0.75), y + Inches(0.22), Inches(0.7), Inches(0.5), size=20, bold=True, color=COBALT)
        add_text(s, title, Inches(1.55), y + Inches(0.15), Inches(2.4), Inches(0.4), size=16, bold=True)
        add_text(s, body, Inches(4.0), y + Inches(0.22), Inches(8.4), Inches(0.5), size=13, color=MUTED)
        y += Inches(1.05)
    add_footer(s, 3, total)

    # --- Slide 4 Timeline ---
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_text(s, "关键进展：三日连打从机制到体验的补全", Inches(0.55), Inches(0.4), Inches(12), Inches(0.5), size=28, bold=True)
    add_text(s, "CHANGELOG 工作日志显示：09-07 AI/命令 → 09-08 机制与显示 → 09-09 单位、征兵与自动作战。", Inches(0.55), Inches(1.0), Inches(12), Inches(0.4), size=14, color=MUTED)

    # timeline line
    add_rect(s, Inches(0.9), Inches(2.55), Inches(11.5), Pt(3), fill=COBALT)
    points = [
        (0.9, "09-07", "AI 增强 · IME · 建议器\n复合指令 · 高 DPI · 50 测试", COBALT),
        (4.7, "09-08", "永久迷雾 · 河流硬阻挡\n地形菜单 · HUD · 性能缓存", TEAL),
        (8.5, "09-09", "曲流地图 · 征兵训练\n自动作战 · 桥梁工程量", AMBER),
    ]
    for x, day, body, color in points:
        cx = Inches(x + 0.35)
        circle = s.shapes.add_shape(MSO_SHAPE.OVAL, cx, Inches(2.4), Inches(0.35), Inches(0.35))
        circle.fill.solid()
        circle.fill.fore_color.rgb = color
        circle.line.fill.background()
        add_rect(s, Inches(x), Inches(3.0), Inches(3.5), Inches(1.95), fill=PANEL, line=LINE, radius=True)
        add_text(s, day, Inches(x + 0.25), Inches(3.15), Inches(3), Inches(0.4), size=18, bold=True, color=color)
        add_text(s, body, Inches(x + 0.25), Inches(3.65), Inches(3), Inches(1.1), size=13, color=INK)

    add_rect(s, Inches(0.55), Inches(5.25), Inches(12.2), Inches(1.45), fill=PANEL2, line=LINE, radius=True)
    add_text(s, "阶段对照", Inches(0.8), Inches(5.4), Inches(1.5), Inches(0.35), size=14, bold=True, color=COBALT)
    add_text(
        s,
        "阶段 0–5 核心能力基本落地（性能原型、对战闭环、迷雾地形、兵种工程、指挥系统、本地 AI）；\n阶段 6–7 语音/大模型与存回放已有可选与可用路径；阶段 8 体验完善进行中。",
        Inches(2.3), Inches(5.4), Inches(10.1), Inches(1.1), size=13, color=MUTED,
    )
    add_footer(s, 4, total)

    # --- Slide 5 Delivered capabilities ---
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_text(s, "已交付能力：一局可完整游玩的功能矩阵", Inches(0.55), Inches(0.4), Inches(12), Inches(0.5), size=28, bold=True)

    rows = [
        ("战场与地图", "连续曲流、六村庄、种子复现、180° 对称、永久探索迷雾、缩略图导航", COBALT),
        ("兵种与编制", "将领+4 护卫；步兵/侦察/工兵/刺客/预备兵；编组命名；鼠标底栏快捷选兵", TEAL),
        ("工程设施", "架桥（按河宽算工程量）、造船登岸、砍树开路、防御塔驻军转弓箭手", AMBER),
        ("指挥输入", "中文规则命令、复合指令、建议补全、歧义候选；可选语音与在线大模型解析", CORAL),
        ("敌方 AI", "战略/战术分层、三档难度、自动作战、信息隔离、将领阵亡适应策略", COBALT),
        ("系统与发行", "存档/回放、设置、统计结算、PyInstaller、CI/Release 双平台打包", TEAL),
    ]
    y = Inches(1.2)
    for title, body, color in rows:
        add_rect(s, Inches(0.55), y, Inches(12.2), Inches(0.85), fill=PANEL, line=LINE, radius=True)
        add_rect(s, Inches(0.55), y, Inches(0.1), Inches(0.85), fill=color)
        add_text(s, title, Inches(0.85), y + Inches(0.22), Inches(2.2), Inches(0.4), size=15, bold=True, color=INK)
        add_text(s, body, Inches(3.2), y + Inches(0.22), Inches(9.2), Inches(0.5), size=13, color=MUTED)
        y += Inches(0.95)
    add_footer(s, 5, total)

    # --- Slide 6 Tech & quality ---
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_text(s, "技术与质量：分层架构 + 可验证的公平与性能", Inches(0.55), Inches(0.4), Inches(12), Inches(0.5), size=28, bold=True)

    layers = [
        ("表现层", "Pygame 渲染 / HUD / 回放 UI"),
        ("输入层", "语音 · 文字 · 鼠标 · 键盘"),
        ("命令层", "结构化命令 · 校验 · 队列"),
        ("AI 层", "本地战略/战术 · 大模型适配"),
        ("感知层", "战争迷雾 · 观察快照"),
        ("模拟层", "固定时间步 · 单位/战斗/工程"),
        ("持久化", "存档 · 回放 · 设置 · 统计"),
    ]
    y = Inches(1.2)
    for i, (name, desc) in enumerate(layers):
        indent = Inches(0.55 + i * 0.08)
        add_rect(s, indent, y, Inches(5.6), Inches(0.7), fill=PANEL if i % 2 == 0 else PANEL2, line=LINE, radius=True)
        add_text(s, name, indent + Inches(0.2), y + Inches(0.18), Inches(1.3), Inches(0.35), size=14, bold=True, color=COBALT if i < 3 else TEAL)
        add_text(s, desc, indent + Inches(1.55), y + Inches(0.18), Inches(3.8), Inches(0.35), size=12, color=MUTED)
        y += Inches(0.78)

    # right quality metrics
    add_rect(s, Inches(6.6), Inches(1.2), Inches(6.15), Inches(5.5), fill=PANEL, line=LINE, radius=True)
    add_text(s, "质量与工程信号", Inches(6.9), Inches(1.4), Inches(5.5), Inches(0.4), size=16, bold=True)
    bullets = [
        "依赖方向强制：模拟层不依赖 UI/网络/麦克风",
        "AI 与界面只读 ObservationSnapshot，防信息泄漏",
        "pytest 单元/集成/性能 + ruff + mypy strict",
        "benchmark_soak 长压；空间哈希与流场寻路",
        "MessagePack + Zstd 原子存档，带版本魔数",
        "语音与 DeepSeek 为可选能力，基础包可离线运行",
        "像素资产程序化可再生成，缺失自动回退占位",
    ]
    bullet_block(s, bullets, Inches(6.9), Inches(2.0), Inches(5.5), Inches(4.4), size=13, color=INK)
    add_footer(s, 6, total)

    # --- Slide 7 Risks ---
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_text(s, "风险与应对：把不确定性压进可控路径", Inches(0.55), Inches(0.4), Inches(12), Inches(0.5), size=28, bold=True)

    headers = ["风险", "影响", "应对"]
    col_w = [Inches(3.4), Inches(2.8), Inches(6.0)]
    x0 = Inches(0.55)
    y0 = Inches(1.2)
    # header
    x = x0
    for htxt, w in zip(headers, col_w):
        add_rect(s, x, y0, w, Inches(0.5), fill=COBALT)
        add_text(s, htxt, x + Inches(0.15), y0 + Inches(0.1), w - Inches(0.2), Inches(0.35), size=13, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
        x += w

    risks = [
        ("千人寻路与碰撞", "帧率不足 / 卡顿", "性能原型优先；空间划分、共享路径、分频更新"),
        ("窄口与桥面拥堵", "单位卡死", "弹性碰撞、局部绕行、桥栏通行区约束"),
        ("AI 服务延迟断网", "命令无法执行", "异步请求、自动暂停、离线切换、紧急存档"),
        ("自然语言理解错误", "玩家失去控制感", "结构化校验、可见反馈、歧义候选、重新下令"),
        ("AI 读取隐藏状态", "破坏公平性", "强制观察快照、接口隔离、泄漏测试"),
        ("回放与实战不一致", "存回放失效", "固定时间步、受控随机、事件日志、检查点"),
        ("V1.0 范围偏大", "周期与联调风险", "阶段交付与完成标准；范围外能力仅预留架构"),
    ]
    y = y0 + Inches(0.5)
    for i, row in enumerate(risks):
        bg = PANEL if i % 2 == 0 else PANEL2
        x = x0
        for j, (cell, w) in enumerate(zip(row, col_w)):
            add_rect(s, x, y, w, Inches(0.7), fill=bg, line=LINE)
            color = CORAL if j == 0 else (AMBER if j == 1 else MUTED)
            add_text(s, cell, x + Inches(0.15), y + Inches(0.15), w - Inches(0.25), Inches(0.5), size=12, bold=(j == 0), color=color if j < 2 else INK)
            x += w
        y += Inches(0.7)
    add_footer(s, 7, total)

    # --- Slide 8 Next steps ---
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_text(s, "下一步：从“能玩”走向“可发布验收”", Inches(0.55), Inches(0.4), Inches(12), Inches(0.5), size=28, bold=True)
    add_text(s, "工作重心进入阶段 8 体验完善与阶段 9 V1.0 验收。", Inches(0.55), Inches(1.0), Inches(12), Inches(0.4), size=14, color=MUTED)

    card(s, Inches(0.55), Inches(1.55), Inches(3.95), Inches(3.0), "阶段 8 · 体验完善", "教程覆盖：选兵 → 编队 → 侦察 → 征兵 → 工程 → 决胜\n\n平衡试玩与数值调参\n单位辨识度与操作反馈\n窗口化 / 全屏 / 分辨率适配\n素材、字体、音效与许可证台账\n\n完成标准：新玩家无需外部文档即可完成一局", accent=COBALT)
    card(s, Inches(4.7), Inches(1.55), Inches(3.95), Inches(3.0), "阶段 9 · V1.0 验收", "Windows / Linux 完整功能测试\n默认 500v500 长期性能测试\nAI 信息隔离与界面侧信道检查\n随机地图公平性与种子复现\n存档 / 回放一致性与版本兼容\n阻断修复 + 可运行构建 + 发布说明\n\n完成标准：满足 V1.0 验收清单全部条件", accent=TEAL)
    card(s, Inches(8.85), Inches(1.55), Inches(3.95), Inches(3.0), "明确不做的（预留）", "经济 / 资源系统\n长线养成与科技树\n真人网络对战\n战役剧情与多语言\n更多兵种与建筑\n\n架构预留接口，V1.0 不实现。\n发布节奏：先验收可玩构建，再对外发布。", accent=AMBER)
    add_rect(s, Inches(0.55), Inches(4.85), Inches(12.2), Inches(1.85), fill=PANEL2, line=LINE, radius=True)
    add_text(s, "近期动作建议", Inches(0.8), Inches(5.05), Inches(2), Inches(0.3), size=13, bold=True, color=COBALT)
    add_text(s, "① 补齐新手教程与指令书闭环　② 完成 500v500 长压与帧率记录　③ 跨平台冒烟 + 存回放一致回归　④ 输出发布说明与素材署名", Inches(0.8), Inches(5.45), Inches(11.5), Inches(0.5), size=13, color=MUTED)
    add_text(s, "协作提示：公共协议（Command / ObservationSnapshot / GameEvent / SaveSnapshot）变更需双方评审后合入。", Inches(0.8), Inches(6.05), Inches(11.5), Inches(0.4), size=12, color=MUTED)
    add_footer(s, 8, total)

    # --- Slide 9 Closing ---
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_text(s, "当前结论", Inches(0.55), Inches(1.4), Inches(12), Inches(0.4), size=14, bold=True, color=COBALT)
    add_text(
        s,
        "核心对战与指挥闭环已可完整游玩，\n双人分层协作与测试基建已就位，\n下一里程碑是体验打磨与跨平台 V1.0 验收。",
        Inches(0.55), Inches(1.95), Inches(8.5), Inches(2.2), size=28, bold=True, color=INK,
    )

    metrics = [
        ("完整对局", "可玩"),
        ("本地 AI", "公平隔离"),
        ("离线输入", "可完成整局"),
        ("双平台", "打包链路已建"),
    ]
    x = Inches(0.55)
    for label, val in metrics:
        add_rect(s, x, Inches(4.7), Inches(2.95), Inches(1.4), fill=PANEL, line=LINE, radius=True)
        add_text(s, label, x + Inches(0.2), Inches(4.9), Inches(2.5), Inches(0.35), size=12, color=MUTED)
        add_text(s, val, x + Inches(0.2), Inches(5.3), Inches(2.5), Inches(0.45), size=20, bold=True, color=TEAL)
        x += Inches(3.1)

    add_text(s, "谢谢 · 欢迎就范围、平衡与验收标准提出问题", Inches(0.55), Inches(6.4), Inches(10), Inches(0.4), size=14, color=MUTED)
    add_footer(s, 9, total)

    prs.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
