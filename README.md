# 实时双轨翻译
基于 **Silero VAD + faster-whisper + NLLB** 的低延迟流式语音翻译应用。  
当前版本采用双轨 MT 架构：
- 快轨：`NLLB-200-distilled-600M` 负责实时输出
- 慢轨：`NLLB-200-1.3B` 负责回填修正（可关闭）

## 核心特性
- 低延迟流式处理（采集 -> VAD -> ASR -> MT -> 字幕）
- 双轨翻译（实时可读 + 后续提质）
- 句首补偿（pre-roll + VAD start 帧缓存）
- 填充词处理（`uh/umm` 等）与续句合并，减少语义断裂
- 固定三类输入设备选择：
  - 电脑输入
  - 耳机输入
  - 电脑内录（loopback）
- 悬浮双语字幕窗（拖拽、透明度、字体调节、点击穿透）

## 处理流程
```text
AudioCapture -> SileroVAD -> Chunker -> faster-whisper(ASR)
            -> TextStabilizer -> MT(实时600M) -> Subtitle
                               -> MT(回填1.3B) -> 回填替换
```

## 环境要求
- Windows 10/11（当前主要优化平台）
- Python 3.10+
- 推荐 NVIDIA GPU（8GB 显存可运行当前默认配置）

## 安装
```bash
python -m venv venv
# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

如需 CUDA 版 PyTorch，请按你的显卡/CUDA 版本安装官方对应 wheel。

## 启动
```bash
python main.py
```

首次运行会自动下载模型（取决于配置）：
- Silero VAD
- faster-whisper（默认 `small.en`）
- NLLB 600M（实时轨）
- NLLB 1.3B（回填轨，若 `refine.enabled=true`）

## 当前默认配置（摘要）
`config.yaml` 当前关键参数：

```yaml
audio:
  pre_roll_ms: 600

vad:
  threshold: 0.50
  speech_start_frames: 4
  speech_end_frames: 14

chunker:
  chunk_ms: 800
  overlap_ms: 180

streaming:
  pause_merge_ms: 380
  final_merge_window_ms: 900
  final_merge_min_chars: 32

asr:
  model_size: small.en
  beam_size: 2
  beam_size_final: 5
  condition_on_previous_text: false
  no_speech_threshold: 0.35
  device: cuda
  compute_type: float16

mt:
  model_name: facebook/nllb-200-distilled-600M
  flush_each_input: false

refine:
  enabled: true
  model_name: facebook/nllb-200-1.3B
  device: auto
  flush_each_input: true
```

## 常见问题
### 1) 为什么日志显示“模型加载完成”后还在下载？
这是正常现象。双轨模型会分别加载；且 Hugging Face 可能在可用后继续按需拉取其余权重分片。

### 2) 句首偶尔被吞怎么办？
优先检查：
- `audio.pre_roll_ms`
- `vad.speech_start_frames`
- 麦克风输入增益/环境噪声

### 3) 回填太慢怎么办？
- 关闭回填：`refine.enabled: false`
- 或改小回填模型、降低 `num_beams`

## 项目结构
```text
src/
  core/
    audio_capture.py
    vad.py
    chunker.py
    asr_worker.py
    text_stabilizer.py
    mt_worker.py
    latency_logger.py
    config_manager.py
  ui/
    main_window.py
    subtitle_window.py
  events/
    events.py
main.py
config.yaml
README.md
```

## 许可证说明
本项目用于学习与研究。请同时遵守依赖模型与库各自的许可证（如 Whisper/NLLB/Silero 等）。
