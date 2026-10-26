# TTS Hub · 统一语音合成调度服务

从 NarrifyAudio 拆出的统一 TTS 调度服务：一套 API、音色库、任务队列和显存分批，调度多个引擎（目前两个），通过 HTTP API（以及 MCP）供 agent 自主调用制作配音：

| 引擎 | 模型 | 许可 | 擅长 |
|---|---|---|---|
| `breeze`（默认） | [Breeze TTS 2](https://github.com/breezeblue-ai/breeze-tts) | 权重与输出**仅限非商业** | 中英文；克隆音色也能用 instruction 控制语气；声音事件 `[笑]` |
| `qwen3` | [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 1.7B（CustomVoice / Base / VoiceDesign） | **Apache-2.0，可商用** | 10 种语言；9 个预置音色（含北京话、四川话）支持 instruction |

- **音色库**：8 个内置音色 + 9 个 Qwen3 预置音色；可用参考音频克隆，或用一句文字描述「设计」新音色并固化，保证同一角色每句音色一致。克隆/设计出的音色两个引擎通用
- **单段合成** `/v1/tts`：长文本自动按句切分、批量合成、拼接
- **多角色配音** `/v1/dub`：一份剧本 → 合并音频 + 逐行音频 + SRT 字幕 + 时间轴
- **批量条目** `/v1/batch`：上千条互相独立的台词一次提交，每条一个文件 + `manifest.json`
- **按显存自动分批**：实测拟合的显存模型算出每批最多放几行，OOM 自动拆批（见「批处理与显存上限」）
- **语气控制**：每句可带 instruction（情绪/语速/音量），文本内可插 `[笑]` `[叹气]` 等声音事件
- **单 GPU 任务队列**：并发请求排队不报错；同步等待或异步轮询
- **OpenAI 兼容** `/v1/audio/speech`；**MCP 服务器** `mcp_server.py`

## 安装

需要 Windows + NVIDIA GPU（≥12 GB 显存）、git、Python 启动器 `py`。

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

脚本会：创建 `.venv`（Python 3.14，uv）→ 装依赖 → 装 CUDA 版 torch 2.11 → 检出固定版本的上游推理代码到
`vendor\breeze-tts` → 校验 GPU。Qwen3-TTS 的推理代码是 pip 包 `qwen-tts`，已在依赖里。

### 模型权重（自动下载）

不需要手动下载：服务用到哪个模型，就先查 HuggingFace 缓存（默认 `%USERPROFILE%\.cache\huggingface\hub`），
没有才自动下载，之后一直复用。

| 引擎 | 权重仓库 | 大小 | 什么时候下载 | 对应变量 |
|---|---|---|---|---|
| breeze | `BreezeBlue/breeze-tts-2` | 约 7.7 GB | 第一次启动（breeze 是默认引擎时） | `BREEZE_MODEL` |
| qwen3 | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 约 4.5 GB | 第一次用 `qwen_*` 预置音色 | `QWEN_CUSTOM_MODEL` |
| qwen3 | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 约 4.5 GB | 第一次用 qwen3 合成克隆/设计出的音色 | `QWEN_BASE_MODEL` |
| qwen3 | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 约 4.5 GB | 第一次用 qwen3 设计音色 | `QWEN_DESIGN_MODEL` |

下载期间 `/health` 停在 `loading`（启动时）或请求一直等待（Qwen3 首次用到时），日志里有「权重未缓存，正在从
HuggingFace 下载」。分词器在各自仓库里，不会再拉别的仓库。

**国内网络 / 镜像**：设置 `HF_ENDPOINT` 后，服务的自动下载就走镜像。写进 `.env`（`start.ps1` 启动时读取）：

```
HF_ENDPOINT=https://hf-mirror.com
```

或只对当前终端生效：`$env:HF_ENDPOINT = "https://hf-mirror.com"; .\start.ps1`。
其他常用变量：`HF_HOME`（换缓存位置，例如放到大容量盘）、`HF_TOKEN`（需要登录时）。

**提前 / 手动下载**（想在启动前下好、机器离线，或遇到下面的符号链接问题）：用虚拟环境自带的 `hf` 命令下载到
普通目录，再把对应变量指向该目录（写进 `.env` 即可）：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"   # 可选：走镜像
.\.venv\Scripts\hf.exe download BreezeBlue/breeze-tts-2 --local-dir D:\models\breeze-tts-2
.\.venv\Scripts\hf.exe download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir D:\models\Qwen3-TTS-CustomVoice
.\.venv\Scripts\hf.exe download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir D:\models\Qwen3-TTS-Base
.\.venv\Scripts\hf.exe download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local-dir D:\models\Qwen3-TTS-VoiceDesign
```

```
BREEZE_MODEL=D:\models\breeze-tts-2
QWEN_CUSTOM_MODEL=D:\models\Qwen3-TTS-CustomVoice
QWEN_BASE_MODEL=D:\models\Qwen3-TTS-Base
QWEN_DESIGN_MODEL=D:\models\Qwen3-TTS-VoiceDesign
```

只用 breeze 时下第一个就够；Qwen3 按需要的功能下对应的仓库。去掉 `--local-dir` 则下载到 HuggingFace 缓存，
服务会自动找到，不用设变量。

> **Windows 符号链接**：HuggingFace 缓存要创建符号链接，没开「开发者模式」也不是管理员时，自动下载会报
> WinError 1314（日志有中文提示）。开启「设置 → 系统 → 开发者选项 → 开发者模式」即可；或者按上面用
> `--local-dir` 下载到普通目录（不需要符号链接），再用变量指过去。

## 启动

```powershell
.\start.ps1                                    # 仅本机：http://127.0.0.1:7861
.\start.ps1 -BindHost 0.0.0.0 -ApiKey 换成你的密钥  # 局域网可访问，需 Bearer 认证
.\start.ps1 -Engine qwen3                      # 默认引擎改为 Qwen3（请求不指定 engine 时使用）
```

约 30–60 秒后 `GET /health` 返回 `"status": "ready"`（加载默认引擎 → 首次生成内置音色 → 预热）。另一个引擎在第一次
被请求时才加载。`/health` 的 `engines` 字段列出每个引擎是否已加载。

### 显存

| 已加载 | 显存（RTX 5090 实测） |
|---|---|
| 仅 Breeze | 约 7.3 GB |
| Breeze + Qwen3 全部三个模型 | 约 19 GB |

Qwen3 的三个模型各自按需加载，只用预置音色时只占一个。显存不足 24 GB 时建议只用一个引擎（`-Engine` 选默认，
请求里不要混用）；空闲卸载会把两个引擎一起释放。
配置项见 [.env.example](.env.example)（复制为 `.env` 生效）：端口、权重路径、API key、批大小、空闲策略等。
服务级变量统一为 `TTS_*`（旧名 `BREEZE_*` 仍兼容，两者都设时以 `TTS_*` 为准）；引擎专属变量带引擎前缀：
`BREEZE_MODEL`、`BREEZE_CODE_DIR`、`BREEZE_FAST_CLONE`、`QWEN_*_MODEL`。也可直接 `python -m tts_hub --help` 用命令行参数。

### 空闲自动卸载 / 关闭

| 阶段 | 默认 | 配置 | 效果 |
|---|---|---|---|
| 无请求 → 卸载权重 | 10 分钟 | `TTS_IDLE_UNLOAD_MINUTES` / `-IdleUnloadMinutes` | 显存 9.4 GB → 2.4 GB（保留 CUDA 上下文）；下个请求自动重载，首个请求多等约 30 秒 |
| 卸载后再无请求 → 退出进程 | 30 分钟 | `TTS_IDLE_SHUTDOWN_MINUTES` / `-IdleShutdownMinutes` | 进程正常退出，显存全部归还 |

设为 `0` 关闭对应阶段（例如 `.\start.ps1 -IdleUnloadMinutes 0 -IdleShutdownMinutes 0` 常驻）。只有 API 请求计入活动，
`/health` 轮询不会让服务保持加载。也可手动 `POST /v1/unload`、`POST /v1/shutdown`。

## 调用

完整接口说明见 [AGENT_GUIDE.md](AGENT_GUIDE.md)（服务运行时也可 `GET /v1/guide` 取到，`/docs` 为交互式文档）。

```bash
curl -X POST http://127.0.0.1:7861/v1/tts -H "Content-Type: application/json" -d "{\"text\":\"你好，欢迎收听。\",\"voice\":\"zh_male_narrator\",\"format\":\"mp3\"}"
```

```python
import httpx
r = httpx.post("http://127.0.0.1:7861/v1/dub", timeout=900, json={
    "name": "ep01", "format": "mp3", "srt_speaker": True,
    "lines": [
        {"voice": "zh_male_narrator", "text": "夜色渐深，古堡里传来一阵奇怪的声响。"},
        {"voice": "zh_young_female", "speaker": "小雨", "text": "你听到了吗？", "instruction": "紧张地小声说"},
    ]}).json()
print(r["result"]["path"], r["result"]["srt"]["path"])
```

选引擎：请求或某一行加 `"engine": "qwen3"`；用 `qwen_*` 预置音色时自动走 qwen3。同一份剧本可以混用两个引擎。

```bash
curl -X POST http://127.0.0.1:7861/v1/tts -H "Content-Type: application/json" -d "{\"text\":\"你怎么能这样！\",\"voice\":\"qwen_uncle_fu\",\"instruction\":\"非常愤怒\"}"
```

## 给 agent 接入 MCP

服务先启动，再把 MCP 服务器注册给 agent（以 Claude Code 为例）：

```bash
claude mcp add tts-hub -- "<项目目录>\.venv\Scripts\python.exe" "<项目目录>\mcp_server.py"
```

其他支持 MCP 的客户端照此配置 stdio 命令即可。环境变量 `TTS_URL`、`TTS_API_KEY` 指向远程服务（旧名 `BREEZE_URL`、`BREEZE_API_KEY` 仍兼容）。
工具：`tts_health`、`tts_list_voices`、`tts_design_voice`、`tts_clone_voice`、`tts_delete_voice`、
`tts_speak`、`tts_dub`、`tts_batch`、`tts_job`。`tts_speak`/`tts_dub`/`tts_batch` 可传 `output_path`/`output_dir` 把结果存到 agent 指定位置。

不支持 MCP 的 agent 直接调 HTTP：让它先读 `GET /v1/guide`。

配音流程与判断经验另有用户级 skill `tts-dubbing`（`~/.claude/skills/tts-dubbing/`）：服务启动检查、
选角与 `casting.json` 跨集复用、剧本整理、按语速自检（`scripts/check_timeline.py`）并重做坏行、交付清单。
提到「配音」「有声书」等会自动加载，也可用 `/tts-dubbing` 显式调用。

## 性能（RTX 5090 实测）

| 场景 | 耗时 | 音频时长 |
|---|---|---|
| Breeze 单句克隆 | 3.3 s | 5.3 s |
| Breeze 剧本 5 行 | 5.4 s | 32.5 s |
| Breeze 剧本 12 行 | 5.1 s | 76.7 s |
| Qwen3 单句预置音色（已加载） | 1.7–6 s | 1.3–4.3 s |
| Qwen3 剧本 12 行预置音色 | 11.2 s | 75.9 s |
| Qwen3 首次加载一个模型 | 约 +10 s | — |

同一请求内的多行会批量合成，所以剧本一次提交比逐句调用快得多。大批量实测见下节。

## 批处理与显存上限

`/v1/dub`、`/v1/batch`、长文本 `/v1/tts` 的所有行先按「引擎 + 生成模板 + 种子」分组，组内从短到长排序，再由
[`tts_hub/batch_limits.py`](tts_hub/batch_limits.py) 的显存模型决定每批放多少行：

```
批峰值 ≈ base + base_per_frame × F + 行数 × multiplier × (row_fixed + per_frame × F)
F      = 参考音频帧 + 文本字数 + 解码上限（批内最长一行）
预算   = 空闲 + 已缓存 − 已占用（规划前实时读取，另一个已加载的引擎自动扣除）
每批行数 = 0.9 × 预算能容纳的行数，向下取偶数，上限 TTS_MAX_BATCH（默认 128）
```

- **Qwen3**：常数来自 NarrifyAudio 的拟合（RTX 5090，每行强制解到上限，残差 <0.4%），并装上同款 talker
  hidden-state 裁剪（数值等价，每帧省 ~116 KiB/行）。单元测试核对了与 Narrify 相同的批大小。
- **Breeze**：本机用 `scripts/calibrate_vram.py` 实测 30 个批次（1–96 行 × 12/48/110 字 × 纯克隆/CFG）拟合；
  CFG（设计音色、带 instruction）每行要跑 cond+uncond 两行，`multiplier=2`。模型对所有样本都不低估。
- **OOM 兜底**：仍然 OOM 时把较短的 90% 立即重跑、最长的推到下一批，之后的批次预算再乘 0.9（`/health` 里的 `oom_scale`）。
- **换显卡**：停服务后运行 `.venv\Scripts\python.exe scripts\calibrate_vram.py`，写出 `data/vram_fit.json`，启动时自动覆盖内置常数。
- 实际分批：`/health` → `engines.<名>.last_batches`。压测：服务运行时执行 `scripts\batch_stress_test.py --items 300`。

大批量实测（RTX 5090，`batch_stress_test.py --items 300`：300 条 / 6980 字，句长 3–80 字，4 个音色）：

| 引擎 | 自动分批 | 耗时 | 音频时长 | 速度 | GPU 峰值 | OOM |
|---|---|---|---|---|---|---|
| Breeze（纯克隆） | 128 + 128 + 44 | 40.5 s | 1553 s | 38× 实时 | 15.3 GiB | 0 |
| Qwen3（预置音色） | 128 + 128 + 44 | 256.4 s | 1691 s | 6.6× 实时 | 31.4 GiB¹ | 0 |

¹ 两个引擎同时驻留，含 PyTorch 缓存池。两者都触到 `TTS_MAX_BATCH=128` 的上限，显存模型给出的容量更大；
冷启动第一次（含 CUDA Graph 捕获）Breeze 约 15× 实时。逐句调用只有 1–2× 实时。

相比上游：纯克隆（无 instruction）在上游每帧调用一次 HF `generate`，约 0.4 倍实时；这里复用了深度解码器的
CUDA Graph（cond=uncond、scale=1，数学上与无引导等价），提速约 3 倍。`BREEZE_FAST_CLONE=0` 可回退上游路径。

## 目录

```
tts_hub/          服务代码：api（HTTP）、jobs（GPU 队列）、engine（Breeze 推理）、qwen_engine（Qwen3 推理）、
                  batch_limits（显存分批模型）、voices（音色库）、audio、text
mcp_server.py     MCP 桥接
scripts/smoke_test.py         Breeze 真机端到端测试（服务运行时执行）
scripts/qwen3_smoke_test.py   Qwen3 真机端到端测试
scripts/batch_stress_test.py  大批量压测：自动批大小、GPU 峰值、吞吐
scripts/calibrate_vram.py     在本机 GPU 上拟合 Breeze 显存模型（停服务后执行）
tests/            单元测试（假引擎，无需 GPU）：.venv\Scripts\python.exe -m pytest tests
vendor/breeze-tts 上游推理代码（install.ps1 检出）
data/             运行数据：voices/ 音色库、outputs/<任务id>/ 产物、jobs/ 任务记录
```

## 许可

服务代码与上游推理代码为 Apache 2.0。
- Breeze TTS 2：**权重与自托管生成的音频仅限研究与非商业用途**（BreezeBlue Research and Non-Commercial License）。
- Qwen3-TTS：权重为 Apache-2.0，可商用。商用作品请整份用 `engine: "qwen3"`，并注意内置的 `zh_*`/`en_*` 音色是用
  Breeze 设计生成的参考音频——商用时改用 `qwen_*` 预置音色，或用 `engine: "qwen3"` 重新设计/克隆的音色。

克隆他人声音须取得授权。
