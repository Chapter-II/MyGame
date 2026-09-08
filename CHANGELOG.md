# 工作日志

## 2026-09-07 会话总结

本次会话对指挥官战术对抗游戏进行了全面改进，涵盖中文输入支持、AI 智能增强、新命令系统、UI/UX 优化和渲染性能提升。

---

### 一、中文 IME 输入支持

**提交**: `d5263af`

- 添加 `pygame.TEXTINPUT` / `pygame.TEXTEDITING` 事件处理，支持中文输入法
- 进入/退出输入模式时调用 `start_text_input()` / `stop_text_input()`
- 添加 `_update_ime_rect()` 方法，将画布坐标转换为屏幕坐标定位 IME 候选窗口
- 输入框实时显示 IME 组合文本（未提交的拼音等）

### 二、AI 智能增强（P0 核心）

**提交**: `e757bc3`

#### 战术层 (`ai/local.py`)
- 将 AI 拆分为战略层（每~2秒）和战术层（每~0.25秒）
- **侦察行为**: 侦察兵走宽侧翼路线揭示战争迷雾
- **残血撤退**: 低 HP 单位自动向远离敌人的方向后撤
- **刺杀突袭**: 刺客发现敌方将领时集火攻击
- **战术命令**: AI 在接近敌人时使用冲锋/冲刺
- **将领保护**: 护卫拦截靠近将领的敌人

#### 工程多样性
- AI 现在会根据局势建造：桥→船→防御塔→道路
- 征召新兵后智能分配：侦察兵、工兵、刺客、步兵

#### 难度设置
- 新增简单/普通/困难三档难度配置（`balance_v1.toml`）
- 设置界面可切换难度，不同难度调整：决策间隔、攻击性、撤退阈值、侦察兵数量
- 困难模式启用 AI 冲锋/鼓舞能力

### 三、战斗统计结算画面

**提交**: `e757bc3`

- 增强世界统计数据：击杀数、输出伤害、按兵种分类的损失明细
- 结算画面显示：对局时长、双方损失/击杀/伤害对比、各兵种损失明细表
- 存档格式向后兼容新增的统计字段

### 四、指令书完善

**提交**: `e757bc3`

- F1 指令书全面改版，列出所有文字命令分类：
  - 移动/攻击/搜索、编组/命名、兵种分化
  - 设施命令（架桥/造船/开路/建塔）
  - 战术命令（全速/冲锋/突击）
  - 载具命令（登船/开船/下船/入塔/出塔）
  - 兵种职责说明、键盘快捷键

### 五、命令建议系统

**提交**: `948ac7b`

- 新增 `CommandSuggester` 类，基于部分输入提供上下文感知的命令建议
- 输入"侦察兵"→ 建议"侦察兵搜索东部"等
- 输入"工兵"→ 建议"工兵在中央架桥"等
- 输入"保护"→ 建议"保护将领"
- 空输入时显示默认建议（根据是否有可见敌人动态调整）
- 模板自动填充方向、村庄、将领等上下文信息

### 六、保护将领命令

**提交**: `948ac7b`

- 新增命令：输入"保护将领"全员以将领为中心围成一圈保护
- 解析器识别"保护"关键词，选择所有单位
- 世界模拟层检测保护意图（守卫目标靠近将领），生成圆形阵型
- 圆形半径根据人数自适应计算

### 七、撤退命令

**提交**: `622bcbe`

- 新增命令：输入"撤退/撤退/后撤/逃跑"
- 智能撤退方向：有可见敌人时向远离敌人的方向撤退 300 像素
- 无敌人时默认撤回己方基地侧

### 八、兵种颜色区分

**提交**: `948ac7b`

- 步兵：红色 `(220, 60, 60)`
- 侦察兵：蓝色 `(70, 130, 230)`
- 工兵：黄色 `(230, 200, 50)`
- 刺客：紫色 `(170, 80, 220)`
- 将领：保持金色/橙色
- 敌方所有单位：外圈黑色光晕

### 九、P1 体验优化

**提交**: `3ba45c0`, `285c330`

| 项目 | 改动 |
|------|------|
| Windows DPI 感知 | `SetProcessDpiAwareness(1)` 修复高 DPI 模糊 |
| 语音录制失焦 | `WINDOWFOCUSLOST` 事件自动停止录制 |
| reduced_motion | 启用时禁止单位位置插值动画 |
| 菜单键盘导航 | Tab/Up/Down 切换焦点，Enter/Space 确认，焦点环高亮 |
| 地形渲染缓存 | numpy 缓冲区 + `make_surface`，仅相机移动或迷雾更新时重建 |
| 小地图缓存 | 地形 Surface 按迷雾更新周期缓存，每帧只绘制单位点 |

---

### 统计

- **提交数**: 7 次有效提交
- **修改文件**: `ai/local.py`, `app/game.py`, `commands/parser.py`, `simulation/world.py`, `config/data/balance_v1.toml`
- **新增代码**: 约 1200+ 行
- **测试状态**: 全部 38 个测试通过
- **待完成**: P1 #12 手工制作默认地图、P2/P3 改进项

---

## 2026-09-08 P2/P3 改进项 + 功能增强

本次会话完成了 P2（代码质量）和 P3（锦上添花）的所有可行项，并新增三项用户可见功能。

---

### 一、P2 代码质量改进

#### 1. 静默异常改为日志记录 (#17)
- `game.py` 中5处 `except: pass` 改为 `logger.debug/warning`
- 新增 `import logging` 和 `logger = logging.getLogger(__name__)`
- DPI 设置、音效加载、按钮焦点、回放保存、设置保存均不再静默吞错

#### 2. RuleCommandParser 世界尺寸同步 (#18)
- `new_battle()` 和 `load_battle()` 中同步 `rule_parser` 及 `suggester` 的 `world_width/world_height`
- 修复地图尺寸变化时指令解析器坐标计算错误的隐患

#### 3. `_resolve_combat` 拆分重构 (#15)
- 原198行方法拆为9个子方法：
  - `_filter_attackers()` — 过滤船上单位和无桥渡河单位
  - `_build_combat_buckets()` — 空间哈希分桶
  - `_compute_combat_visibility()` — 迷雾可见性计算
  - `_select_targets()` — 目标选择主循环
  - `_try_attack_facility()` — 设施攻击处理
  - `_find_nearest_target()` — 最近敌人搜索
  - `_resolve_attack()` — 攻击判定与伤害
  - `_apply_facility_damage()` — 设施伤害结算
  - `_apply_unit_damage()` — 单位伤害结算与死亡

#### 4. AI 将领死后适应
- `_has_commander()` 检测己方将领存活
- 将领死亡后 aggression +0.3（拼命模式）
- 跳过保护将领指令
- 跳过兵种转化（直接投入战斗）

### 二、P3 代码改进

#### 5. 删除未使用枚举值 (#27)
- 移除 `CommandStatus.CANCELLED` 和 `COMPLETED`（从未被引用）

#### 6. 缓存鼠标位置 (#26)
- `_draw()` 入口缓存一次 `self._mouse_pos`
- 6处 draw 方法改用缓存值，每帧减少5次 `pygame.mouse.get_pos()` 调用

#### 7. 魔法数字提取为常量 (#19)
- 新建 `constants.py`（51行），提取30+个命名常量
- `world.py` 中替换：BRIDGE_RADIUS、FACILITY_INTERACTION_RADIUS、BUILD_PROXIMITY、GUARD_INTERCEPT_DISTANCE、COMBAT_CELL_SIZE、MELEE_RANGE_THRESHOLD、MAX_MELEE_ENGAGEMENTS、CHASE_DISTANCE、GOLDEN_ANGLE、FORMATION_SPACING、ARRIVAL_DISTANCE、COLLISION_*、COMPOSITION_* 等
- `game.py` 中替换：UNIT_CLICK_RADIUS_SQ、ENEMY_CLICK_RADIUS_SQ

#### 8. 河流过河预警 (#23)
- 移动命令被接受后，调用 `_crossing_loss_estimate()` 估算 HP 损失
- 损失 >10 时显示黄色警告消息

### 三、新增功能

#### 9. 文字指令切分（复合指令）
- `_submit_text()` 检测连接词（然后/接着/并且/同时/再/并）并自动拆分
- 每段独立解析执行，全部成功后提示"已执行 N 条指令"
- 扩展移动关键词：前进/推进/进军

#### 10. 预选项系统接入
- `CommandSuggester` 已实现但从未接入，本次完成集成：
  - 输入文字时实时调用 `suggester.suggest()` 刷新建议
  - 输入框上方渲染建议下拉列表
  - Tab 补全 / ↑↓ 选择 / Enter 执行
  - 打开输入框时自动显示默认建议

#### 11. 高 DPI 清晰渲染
- 使用 `pygame.SCALED` 标志，游戏以屏幕原生分辨率渲染
- 移除 `smoothscale` 双线性插值放大，文字和线条不再模糊
- 不支持 SCALED 时自动回退到原方案
- 全屏切换和窗口缩放均适配

### 四、游戏平衡

#### 12. 全局速度 ×0.5
- 所有兵种 speed 减半：步兵 48→24、侦察兵 78→39、工兵 45→22.5 等
- 船只速度 55→27.5

### 五、测试

- 新增 `tests/unit/test_ai.py`（4个 AI 将领死亡测试）
- `test_commands.py` 扩展7个测试（parser 尺寸同步、前进/进军识别、建议器、复合指令切分）
- `test_world.py` 调整步数以适配新速度
- **最终状态**: 50 passed, 1 skipped

### 统计

- **修改文件**: 8个修改 + 2个新建
- **净改动**: +447 / -184 行
- **新建**: `src/mygame/constants.py`, `tests/unit/test_ai.py`
