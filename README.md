# 低延迟流式翻译

基于 faster-whisper + NLLB 的实时低延迟语音翻译应用。采用流式管道架构，支持系统内录和麦克风输入，端到端延迟目标 < 2 秒。

## 架构

```
Audio Capture → Silero VAD → Chunker → faster-whisper ASR →
Text Stabilizer → NLLB MT → 双语字幕浮窗
```

5 个线程通过有界队列通信，互不阻塞：

| 线程 | 职责 |
|-----|------|
| Audio Capture | WASAPI/麦克风采集，20ms 帧 |
| VAD + Chunker | 语音检测 + 1s 分块 |
| ASR Worker | faster-whisper 语音识别 |
| MT Worker | NLLB 文本翻译（仅 final） |
| UI Main Thread | PyQt6 字幕渲染 |

## 系统要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| GPU | GTX 1060 6GB | RTX 3060 12GB |
| CPU | Intel i5-8th | Intel i7-10th |
| 内存 | 8GB | 16GB |
| 存储 | 5GB | 10GB |

- Windows 10/11（优先）或 Linux
- Python 3.10+
- NVIDIA 驱动程序（GPU 加速需要）

## 安装

### 1. 创建虚拟环境

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# Linux
source venv/bin/activate
```

### 2. 安装 PyTorch

根据显卡选择：

```bash
# RTX 40/30 系列
pip install torch --index-url https://download.pytorch.org/whl/cu121

# RTX 50 系列 (Blackwell)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 无 GPU
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 验证

```python
import torch
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## 使用

```bash
python main.py
```

首次运行会自动下载模型：
- Silero VAD（~2MB）
- faster-whisper small（~500MB）
- NLLB-200-distilled-600M（~1.2GB）

### 界面操作

1. 等待模型加载完成（状态栏显示进度）
2. 选择音频输入设备
3. 点击「开始翻译」
4. 字幕浮窗自动显示双语字幕

### 字幕浮窗

- 拖拽移动位置
- 双击切换鼠标穿透模式
- 上方灰色文字：原文（含 partial 实时预览）
- 下方白色文字：译文（final 结果）

## 配置

编辑 `config.yaml` 调整参数：

```yaml
# 音频设置
audio:
  input_mode: loopback  # loopback（系统内录） | mic（麦克风）
  device_index: null    # null 为自动选择，填数字指定设备

# VAD 设置
vad:
  threshold: 0.50       # 语音检测阈值（0-1）
  speech_start_frames: 6   # 连续几帧判定语音开始
  speech_end_frames: 10    # 连续几帧判定语音结束

# 分块设置（直接影响实时性）
chunker:
  chunk_ms: 800         # 推荐 700-900，越小延迟越低
  overlap_ms: 180       # 推荐 120-220，减少断词

# ASR 设置
asr:
  model_size: small.en  # tiny | base | small | medium | medium.en | large-v3
  language: en
  beam_size: 2          # 推荐 1-2，越大越准但更慢
  device: cuda          # cuda | cpu
  compute_type: float16 # float16（GPU） | int8（CPU）

# MT 设置
mt:
  model_name: facebook/nllb-200-distilled-600M
  src_lang: eng_Latn
  tgt_lang: zho_Hans
  cache_size: 4096
  num_beams: 2
  batch_max_wait_ms: 120  # 短句合并等待时间（毫秒）
  batch_max_chars: 220    # 达到该长度立即翻译
  max_chars: 360          # 超长文本按标点截断

# UI 设置
ui:
  font_size: 24
  opacity: 0.8
```

## 项目结构

```
src/
├── core/
│   ├── audio_capture.py      # 音频采集（WASAPI/麦克风）
│   ├── ring_buffer.py        # 环形缓冲区
│   ├── vad.py                # 语音活动检测（Silero VAD）
│   ├── chunker.py            # 音频分块器
│   ├── asr_worker.py         # ASR 工作线程（faster-whisper）
│   ├── text_stabilizer.py    # 文本稳定器（LCP 算法）
│   ├── mt_worker.py          # 翻译工作线程（NLLB）
│   ├── config_manager.py     # 配置管理
│   └── latency_logger.py     # 延迟打点日志
├── ui/
│   ├── main_window.py        # 主控制窗口
│   └── subtitle_window.py    # 字幕浮窗
├── events/
│   └── events.py             # 事件数据类定义
├── queues/
│   └── __init__.py
└── utils/
    └── __init__.py
main.py                       # 入口文件
config.yaml                   # 配置文件
```

## 语言代码

### ASR (Whisper)

| 代码 | 语言 |
|-----|------|
| en | 英语 |
| zh | 中文 |
| ja | 日语 |
| ko | 韩语 |
| es | 西班牙语 |
| fr | 法语 |
| de | 德语 |

### MT (NLLB)

| 代码 | 语言 |
|-----|------|
| eng_Latn | 英语 |
| zho_Hans | 简体中文 |
| zho_Hant | 繁体中文 |
| jpn_Jpan | 日语 |
| kor_Hang | 韩语 |
| spa_Latn | 西班牙语 |
| fra_Latn | 法语 |
| deu_Latn | 德语 |

## 延迟分解

```
总延迟 < 2000ms

├── 音频采集      ~20ms
├── VAD 判定      ~120ms
├── Chunk 累积    ~1000ms
├── ASR 推理      ~300ms
├── 文本稳定化    ~1ms
├── MT 翻译       ~200ms
└── UI 渲染       ~10ms
```

延迟日志自动输出到 `logs/latency.csv`。

## 常见问题

### 没有音频输入

- Windows：选择带「WASAPI」的 loopback 设备捕获系统音频
- 检查设备列表中的设备索引是否正确
- 在 `config.yaml` 中将 `input_mode` 改为 `mic` 使用麦克风

### 显存不足

- 将 ASR 模型改为 `tiny` 或 `base`
- 将 `compute_type` 改为 `int8`
- 将 `device` 改为 `cpu`（速度会降低）

### 翻译质量不佳

- 将 ASR 模型改为 `medium`（需要更多显存）
- 将 NLLB 模型改为 `facebook/nllb-200-1.3B`
- 调整 VAD 阈值减少误触发

## 许可证

本项目仅供学习和研究使用。
- faster-whisper: MIT License
- NLLB: CC-BY-NC License
- Silero VAD: MIT License
