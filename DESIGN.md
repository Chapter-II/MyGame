---
name: Commander Tactical Arena
description: A restrained midnight command table for readable thousand-unit warfare.
colors:
  night-command: "oklch(0.100 0.000 0)"
  command-surface: "oklch(0.165 0.018 260)"
  cobalt-signal: "oklch(0.620 0.160 260)"
  cold-ink: "oklch(0.930 0.010 260)"
  quiet-ink: "oklch(0.720 0.025 260)"
  warning-amber: "oklch(0.780 0.150 75)"
  ally-cyan: "oklch(0.760 0.130 220)"
  enemy-coral: "oklch(0.680 0.190 28)"
rounded:
  control: 6
  panel: 10
spacing:
  xs: 4
  sm: 8
  md: 12
  lg: 20
---

# Design System: Commander Tactical Arena

## 1. Overview

**Creative North Star: "午夜军令台"**

玩家在普通室内光线下长时间观察一块冷色战术沙盘。近黑底色压低环境噪声，钴蓝只标记可以操作或已选中的内容，琥珀只承担风险和等待，阵营颜色留给战场实体。

界面采用固定工具栏与可折叠信息轨道，不用漂浮玻璃卡片侵占战场。信息密度可以高，但层级必须稳定；动效只解释状态变化。

**Key Characteristics:**

- 战场占据主要面积，工具区贴边停靠。
- 高对比中文文字与清晰的 4/8 像素节奏。
- 图形、颜色和文字共同表达阵营与状态。
- 平面色阶区分层级，仅模态窗口获得短小阴影。

## 2. Colors

近黑中性色构成夜间指挥台，冷钴是稀缺的操作信号，琥珀与珊瑚承担风险而不装饰页面。

### Primary

- **钴蓝军令** (`oklch(0.620 0.160 260)`): 主要操作、键盘焦点、当前选择。
- **友军青** (`oklch(0.760 0.130 220)`): 友军轮廓、可控范围和己方迷雾视野。

### Secondary

- **警戒琥珀** (`oklch(0.780 0.150 75)`): 等待、耐力不足、工程警告。
- **敌情珊瑚** (`oklch(0.680 0.190 28)`): 当前可见敌军和危险事件。

### Neutral

- **夜幕底色** (`oklch(0.100 0.000 0)`): 窗口和未探索区域。
- **军令台面** (`oklch(0.165 0.018 260)`): 工具栏与面板。
- **冷白正文** (`oklch(0.930 0.010 260)`): 正文和关键数据。
- **静默文字** (`oklch(0.720 0.025 260)`): 次要说明，不能用于关键状态。

**The Sparse Signal Rule.** 钴蓝和琥珀只用于状态与操作，每个非战场界面不超过约 10% 的可见面积。

## 3. Typography

**Display Font:** Noto Sans CJK SC（Microsoft YaHei、WenQuanYi、系统无衬线回退）
**Body Font:** 同一字体家族
**Label/Mono Font:** 系统等宽字体，仅用于种子、帧号和调试数据

**Character:** 单一中文无衬线字体提供可信且紧凑的工具感。标题靠字号与字重区分，不使用装饰字体。

### Hierarchy

- **Display** (700, 32px, 1.2): 主菜单标题。
- **Headline** (700, 24px, 1.25): 场景与结算标题。
- **Title** (600, 18px, 1.3): 面板分区。
- **Body** (400, 15px, 1.5): 指令、教程和反馈；长文限制约 70 个中文字符宽。
- **Label** (600, 13px, 0): 快捷键、数值和状态。

**The Operational Type Rule.** 按钮、标签和数据不使用展示字体，也不使用全大写英文制造层级。

## 4. Elevation

默认无阴影，通过夜幕、台面和悬浮层三档色阶建立结构。仅暂停、断网和确认窗口使用 0 4px 8px 的紧凑阴影，遮罩负责聚焦而非装饰。

**The Flat Command Table Rule.** 静止面板保持平面；悬浮、焦点和按下状态通过颜色、轮廓及 150–200ms 过渡表达。

## 5. Components

### Buttons

- **Shape:** 6px 圆角，最小高度 36px。
- **Primary:** 钴蓝底、冷白文字、水平 16px 内距。
- **Hover / Focus:** 提高亮度；焦点使用 2px 友军青轮廓。
- **Secondary / Ghost:** 台面色或透明底，完整 1px 边框，不使用侧边色条。

### Chips

- **Style:** 4px 圆角、紧凑标签，用图标或形状补充颜色含义。
- **State:** 选中时填充，未选中时只有完整轮廓。

### Cards / Containers

- **Corner Style:** 10px 封顶；战斗 HUD 优先使用连续停靠面板而非卡片网格。
- **Background:** 军令台面或更深一档色阶。
- **Shadow Strategy:** 常态无阴影。
- **Border:** 仅在层级不清时使用完整 1px 边框。
- **Internal Padding:** 12px 或 20px。

### Inputs / Fields

- **Style:** 6px 圆角、深色实体背景、1px 中性描边。
- **Focus:** 钴蓝描边与清晰插入光标。
- **Error / Disabled:** 珊瑚色文字说明原因；禁用状态仍保持可读。

### Navigation

顶部状态栏和右侧指挥轨道保持固定。当前页以钴蓝文字和完整底色表示；面板折叠后保留图标、文字提示和键盘访问。

### Command Feed

每条指令按时间排列，显示来源、解析文本、目标、状态图标和中文原因码。等待、成功、失败不能只靠颜色区分。

## 6. Do's and Don'ts

### Do:

- **Do** 让战场始终占据最大连续区域。
- **Do** 对所有语音、网络和命令状态提供文本反馈。
- **Do** 使用 4/8 像素节奏、150–200ms 状态过渡和减弱动态替代。
- **Do** 用轮廓形状与兵种字形补充友军青和敌情珊瑚。

### Don't:

- **Don't** 用花哨装饰、玻璃拟态、过量粒子或无意义动效遮挡战场。
- **Don't** 用大面积高饱和色装饰非交互区域。
- **Don't** 把复杂状态隐藏在只有颜色才能理解的编码中。
- **Don't** 在卡片上使用大于 1px 的侧边彩色条、渐变文字或 16px 以上模糊阴影。
