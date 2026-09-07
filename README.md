# 指挥官战术对抗

俯视角、即时制、千人独立模拟的单人指挥官战术游戏。完整需求见 [myplan.md](myplan.md)。

## 开发运行

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev,voice]'
.venv/bin/mygame
```

没有语音依赖或 DeepSeek 密钥时，鼠标、键盘和本地中文规则命令仍可完成对局。可选的大模型解析只读取环境变量：

```bash
export DEEPSEEK_API_KEY='...'
export DEEPSEEK_MODEL='deepseek-v4-flash'
```

## 操作

- WASD / 中键拖动：摄像机；滚轮：缩放
- 左键 / 框选：选择；右键：移动；A + 右键：攻击移动
- Ctrl+1–9：编组；1–9：选择编组；Shift：追加命令
- 方向键：直接移动将领；V：按住说话；Enter：文字命令
- Space：暂停；- / +：速度；F1：指令书；F5：保存；F9：载入
- 回放中 F2 / F3 / F4：玩家 / 敌方 / 全局视角；左右方向键跳转 5 秒

离线标准文字命令示例：

- `第一侦察队前往北部中央`
- `20名初始兵分化为工兵`
- `工兵队在中央架桥`
- `第1组守住最近村庄`
- `第一战团集火敌方将领`
- `20名步兵编为第6组 命名为河西守军`
- `第6组进入防御塔`、`第6组离开防御塔`
- `第1组登船`、`船只前往西岸`、`第1组下船`

在线解析失败时战局自动暂停：`R` 最多重试一次，`O` 切换离线，`Q` 创建紧急存档并退出。语音模型首次下载必须在游戏提示后按 `Y` 明确确认。

## 验证

```bash
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/mypy src/mygame
.venv/bin/python -m mygame --headless --ticks 400
.venv/bin/python scripts/benchmark_soak.py --ticks 24000
```

程序化资源可重复生成：

```bash
.venv/bin/python scripts/generate_assets.py
```

存档、回放、设置和轮换日志分别写入 `platformdirs` 返回的用户数据/日志目录。存档与回放使用带版本魔数的 MessagePack + Zstandard 容器、原子替换和单份备份。

## 原生发行包

Windows 与 Linux 必须分别在对应系统构建：

```bash
.venv/bin/pyinstaller --noconfirm CommanderTacticalArena.spec
dist/CommanderTacticalArena/CommanderTacticalArena --headless --ticks 10
```

`.github/workflows/release.yml` 会在 Ubuntu 22.04 x64 与 Windows x64 上分别打包、冒烟测试并上传目录式发行包。语音依赖及模型保持可选，不进入基础发行包；源码安装 `.[voice]` 后可启用本地转写。
