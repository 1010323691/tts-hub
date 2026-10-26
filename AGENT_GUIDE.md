# TTS Hub（Breeze TTS 2 + Qwen3-TTS）· Agent 调用指南

Base URL 默认 `http://127.0.0.1:7861`。设置了 API key 时，每个请求带 `Authorization: Bearer <key>`。
所有接口收发 JSON（建音色上传文件除外）。交互式文档：`/docs`。

## 推荐流程

1. `GET /health` → `status` 为 `ready` 或 `unloaded` 可合成（`unloaded` = 空闲卸载了权重，首个请求自动重载、多等约 30 秒）；
   `loading`/`seeding`/`warming` 正在启动，约 30–60 秒。连不上 = 服务未启动或已空闲自动退出。
2. `GET /v1/voices` → 选音色。剧本里的每个角色固定用一个 `consistent: true` 的音色。
3. 缺角色就建音色：
   - 文字描述：`POST /v1/voices/design`（默认固化：生成样本作为参考音频，之后每句音色一致）
   - 参考音频：`POST /v1/voices`（multipart，音频 + 逐字文本）
4. 单段文本 `POST /v1/tts`；多角色剧本 `POST /v1/dub`；大量互相独立的条目 `POST /v1/batch`。
   多条内容一定要一次提交：服务按显存自动分批并行，比逐条调用快一个数量级（见「批处理」）。
5. 结果里 `path` 是服务端本机绝对路径，`url` 可直接下载。

## 两个引擎

| | `breeze`（默认，见 `/health` 的 `default_engine`） | `qwen3` |
|---|---|---|
| 许可 | 权重与输出**仅限非商业** | Apache-2.0，**可商用** |
| 语言 | 中、英 | 中、英、日、韩、德、法、西、意、葡、俄 |
| 克隆音色 + instruction | 支持（Voice Direction） | **不支持**，instruction 被忽略并在 `warnings` 提示 |
| 预置音色 `qwen_*` | 不可用 | 支持，且可带 instruction |
| 声音事件 `[笑]` | 支持 | 不支持，不要在 qwen3 的行里写 |

- 请求体或剧本某一行加 `"engine": "qwen3"` 选引擎；用 `qwen_*` 预置音色时自动走 qwen3，显式指定 breeze 会返回 400。
- 克隆、设计出来的音色两个引擎通用；同一份剧本可以混用两个引擎（结果每行带 `engine`）。
- `language`（仅 qwen3）：`zh`/`en`/`ja`/`ko`/`de`/`fr`/`es`/`it`/`pt`/`ru`，缺省自动识别。
- 未加载的引擎在第一次请求时加载，多等约 10–30 秒。
- **商用作品**：整份剧本用 qwen3，音色用 `qwen_*` 预置音色或用 `engine: "qwen3"` 设计/克隆的音色；
  内置 `zh_*`/`en_*` 音色的参考音频是 Breeze 生成的，不要用于商用。

## 内置音色

| id | 说明 | 引擎 |
|---|---|---|
| `zh_female_narrator` | 中文女声旁白（默认） | 两者 |
| `zh_male_narrator` | 中文男声旁白，沉稳醇厚 | 两者 |
| `zh_young_male` / `zh_young_female` | 青年男声 / 青年女声 | 两者 |
| `zh_girl` / `zh_old_man` | 小女孩 / 老爷爷 | 两者 |
| `en_female_narrator` / `en_male_narrator` | 英文女声 / 男声旁白 | 两者 |
| `qwen_vivian` / `qwen_serena` | 明亮有个性的年轻女声 / 温暖柔和的年轻女声 | qwen3 |
| `qwen_uncle_fu` | 低沉醇厚的成熟男声 | qwen3 |
| `qwen_dylan` / `qwen_eric` | 年轻男声（北京话）/ 活泼略沙哑男声（四川话） | qwen3 |
| `qwen_ryan` / `qwen_aiden` | 英语男声：有动感 / 阳光清晰 | qwen3 |
| `qwen_ono_anna` / `qwen_sohee` | 日语女声 / 韩语女声 | qwen3 |

## 文本写法

- 同一句尽量单一语言。
- 声音事件（仅 breeze）直接写进 text：中文 `[笑]` `[叹气]` `[咳嗽]` `[清嗓子]`，英文 `(laugh)` `(sigh)` `(cough)` `(clears throat)`。
- `instruction` 控制语气、情绪、语速、音量，例如 `"愤怒地大声质问，语速很快"`、`"Speak slowly with a restrained, serious tone."`。
  instruction 语言与台词一致效果最好。breeze 带 instruction 时默认 CFG=4，否则 1（`cfg_scale` 可覆盖，qwen3 忽略）。
- 长文本自动按句切分（每块 ≤120 字）后拼接，无需手动拆。

## POST /v1/tts

```json
{"text": "[笑] 欢迎来到今晚的故事时间。", "voice": "zh_young_female",
 "instruction": "", "format": "mp3", "seed": 42}
```

可选：`engine` breeze|qwen3、`language`（仅 qwen3）、`format` wav|mp3|flac|ogg（默认 wav）、`seed`（同参数同种子可复现）、`trim_silence`（默认 true）、
`normalize`（响度统一，默认 true）、`chunk_pause_ms`（切块停顿，默认 200）、`wait`（默认 true 同步）、`wait_seconds`。

响应（200）：

```json
{"id": "tts_...", "status": "done",
 "result": {"path": "D:\\...\\speech.mp3", "url": "http://.../files/tts_.../speech.mp3",
            "duration_s": 5.15, "format": "mp3", "sample_rate": 24000, "voice": "zh_young_female",
            "chunks": [{"text": "...", "start": 0.0, "end": 5.15, "capped": false}], "warnings": []}}
```

## POST /v1/dub（多角色配音）

```json
{"name": "第一集", "format": "mp3", "pause_ms": 400, "same_speaker_pause_ms": 250, "srt_speaker": true,
 "lines": [
   {"voice": "zh_male_narrator", "text": "夜色渐深，古堡里传来一阵奇怪的声响。"},
   {"voice": "zh_young_female", "speaker": "小雨", "text": "你听到了吗？", "instruction": "紧张害怕，压低声音"},
   {"voice": "villain", "speaker": "伯爵", "text": "[笑] 欢迎来到我的城堡。", "pause_after_ms": 800}
 ]}
```

每行可选：`instruction`、`speaker`（字幕角色名）、`pause_after_ms`、`seed`、`cfg_scale`、`engine`、`language`。
请求级的 `engine`/`language` 是各行的默认值。
某行效果不好时只给该行换 `seed` 再整份提交即可重做该行（同一批次的其他行也可能随之变化，需重新检查）。
停顿规则：`pause_after_ms` 优先；否则下一行同音色用 `same_speaker_pause_ms`，换音色用 `pause_ms`。

结果：`path`/`url`（合并音频）、`srt`（字幕）、`timeline_json`、`output_dir`、
`lines[]`（每行 `start`/`end` 秒、`file` 逐行音频）。

## POST /v1/batch（批量独立条目）

互相独立的语音条目（游戏/App 台词、提示音、单词朗读、同一句多个版本），每条输出一个文件，不拼接、不出字幕：

```json
{"format": "mp3", "engine": "qwen3",
 "items": [
   {"id": "npc_merchant_greet", "text": "欢迎光临，随便看看！", "voice": "qwen_uncle_fu", "instruction": "热情"},
   {"id": "npc_guard_warn", "text": "站住，前面不能过去。", "voice": "qwen_dylan"},
   {"text": "没有 id 时按序号命名。", "voice": "qwen_serena"}
 ]}
```

每条可选：`id`（文件名，缺省为序号，不能重复）、`voice`、`instruction`、`seed`、`cfg_scale`、`engine`、`language`；
请求级的 `engine`/`language`/`seed`/`format` 是默认值。最多 2000 条。
结果：`output_dir`、`manifest`（`manifest.json`）、`items[]`（每条 `id`、`file`、`duration_s`、`capped`、`engine`）、`total_duration_s`。

## 批处理

`/v1/dub`、`/v1/batch`（以及长文本的 `/v1/tts`）里的所有行，服务会按「模板/引擎 + 种子」分组，组内按长度从短到长，
用实测拟合的显存模型计算**当前空闲显存下每批最多能放多少行**，切成尽可能大的批次并行生成：

- 显存模型：`批峰值 ≈ 批开销 + 行数 × (每行固定开销 + 每帧开销 × 帧数)`，帧数按批内最长一行的「参考音频 + 文本 + 解码上限」算，
  预算取 `空闲 + 已缓存 − 已占用`，只用到预测值的 90%。Qwen3 常数来自 NarrifyAudio 的拟合，Breeze 常数在本机实测。
- 显存不足（OOM）时自动拆批：较短的 90% 立即重跑，最长的另起一批，之后的批次预算再收紧 10%。
- `/health` 的 `engines.<名>.last_batches` 显示上一个任务实际切成了几批、每批几行。

所以：**几十上百条一次交给 `/v1/dub` 或 `/v1/batch`**，不要循环调 `/v1/tts`。单次上千行可按章节分几次提交，避免单个任务太久。

## 建音色

文字设计（同步，约 5–15 秒）：

```json
POST /v1/voices/design
{"voice_id": "villain", "name": "伯爵", "instruction": "一位阴险狡诈的中年男性，声音低沉沙哑，语速缓慢，带着冷笑。",
 "language": "zh"}
```

`freeze: false` 则不固化（每次按描述重新设计，音色会漂移，不适合同一角色多句）。不满意可换 `seed` 并 `overwrite: true` 重建。
`engine` 指定用哪个引擎生成样本（qwen3 用 VoiceDesign 模型，商用作品选它）；生成的音色两个引擎都能用。

参考音频克隆（multipart/form-data）：字段 `ref_audio`（文件，推荐 5–15 秒清晰人声）、`ref_text`（逐字文本，必须准确）、
`voice_id`、`name`、可选 `instruction`（默认语气）、`overwrite`。服务端本机调用也可用 `ref_audio_path` 代替上传。

其他：`GET /v1/voices/{id}`、`GET /v1/voices/{id}/audio`（试听参考音频）、`DELETE /v1/voices/{id}`。

## 异步任务

请求体 `"wait": false` 立即返回 202 + `id`；`GET /v1/jobs/{id}?wait_seconds=300` 长轮询；
同步请求超过 `wait_seconds` 也会返回 202，照样轮询即可。`DELETE /v1/jobs/{id}` 取消。
任务串行排队（单 GPU），`queue_position` 表示前面还有几个。

## 空闲卸载与自动关闭

默认 10 分钟没有 API 请求（`/health` 轮询不算）就卸载权重释放显存，之后再过 30 分钟没有请求就退出进程。
`/health` 的 `idle` 字段给出 `unload_in_s`、`shutdown_in_s`。手动操作：`POST /v1/unload` 立即卸载，
`POST /v1/shutdown` 在已提交任务完成后关闭。

## OpenAI 兼容

`POST /v1/audio/speech`，`{"model": "qwen3-tts", "input": "...", "voice": "qwen_serena",
"instructions": "...", "response_format": "mp3"}` 直接返回音频字节。`model` 含 qwen 走 qwen3、含 breeze 走 breeze，
其余按音色/默认引擎。OpenAI 音色名（alloy 等）映射到默认音色；`speed` 忽略。

## 错误

400 参数/音色错误、401 未授权、404 音色或任务不存在、422 合成失败（`error` 字段说明原因）、503 引擎未就绪。
`warnings` / `capped: true` 表示该段疑似解码失控（已自动换种子重试），建议试听或换 seed 重做。

## 许可

Breeze TTS 2 权重与自托管输出仅限研究与非商业用途；Qwen3-TTS 为 Apache-2.0 可商用。克隆他人声音须获得授权。
