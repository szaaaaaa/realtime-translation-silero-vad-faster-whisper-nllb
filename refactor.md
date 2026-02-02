# 实时低延迟翻译应用开发文档（Laptop 本地运行）

## 项目概述

### 目标
- 笔记本电脑本地运行
- 端到端延迟尽可能低（目标 < 2s）
- 语音 → 字幕翻译（S2TT）
- 不做语音 → 语音（S2ST）

### 范围
- Windows 优先（Linux 兼容）
- 系统内录（会议声音）与麦克风输入
- 双语字幕浮窗（置顶、透明、可穿透）

### 与 SeamlessM4T 方案的对比

| 特性 | SeamlessM4T (当前) | Whisper + NLLB (本方案) |
|------|-------------------|------------------------|
| 架构 | 端到端 | 流式管道 |
| 延迟 | 高（2-10s） | 低（< 2s） |
| 显存 | 6-10GB | 2-4GB |
| 模型大小 | ~10GB | ~2GB |
| 流式支持 | 否 | 是 |
| partial 输出 | 否 | 是 |

---

## 项目结构

```
src/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── audio_capture.py      # 音频采集（WASAPI/麦克风）
│   ├── ring_buffer.py        # 环形缓冲区
│   ├── vad.py                # 语音活动检测（Silero VAD）
│   ├── chunker.py            # 音频分块器
│   ├── asr_worker.py         # ASR 工作线程（faster-whisper）
│   ├── text_stabilizer.py    # 文本稳定器
│   ├── mt_worker.py          # 翻译工作线程（NLLB/CTranslate2）
│   ├── config_manager.py     # 配置管理
│   └── latency_logger.py     # 延迟打点日志
├── ui/
│   ├── __init__.py
│   ├── main_window.py        # 主控制窗口
│   └── subtitle_window.py    # 字幕浮窗
├── queues/
│   ├── __init__.py
│   └── priority_queue.py     # 支持 partial/final 优先级的队列
├── events/
│   ├── __init__.py
│   └── events.py             # 事件数据类定义
└── utils/
    ├── __init__.py
    └── audio_utils.py        # 音频工具函数
main.py                       # 入口文件
config.yaml                   # 配置文件
requirements.txt              # 依赖列表
```

---

## 依赖库

### requirements.txt

```txt
# 核心依赖
numpy>=1.24.0
PyQt6>=6.5.0

# 音频采集
sounddevice>=0.4.6
pyaudio>=0.2.13  # WASAPI 支持备选

# VAD
silero-vad>=4.0
onnxruntime>=1.15.0

# ASR
faster-whisper>=0.10.0
ctranslate2>=3.20.0

# MT（二选一）
# 方案 A: CTranslate2 + NLLB
ctranslate2>=3.20.0

# 方案 B: HuggingFace Transformers
# transformers>=4.36.0
# sentencepiece

# PyTorch（GPU 推理）
--extra-index-url https://download.pytorch.org/whl/cu121
torch>=2.1.0

# 配置
pyyaml>=6.0

# 开发/调试
tqdm
```

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| GPU | GTX 1060 6GB | RTX 3060 12GB |
| CPU | Intel i5-8th | Intel i7-10th |
| 内存 | 8GB | 16GB |
| 存储 | 5GB | 10GB |

---

## 总体处理流程

```
Audio (stream)
      ↓
┌─────────────────┐
│  Audio Capture  │  ← WASAPI Loopback / Microphone
└────────┬────────┘
         ↓
┌─────────────────┐
│   Ring Buffer   │  ← 30s 环形缓冲
└────────┬────────┘
         ↓
┌─────────────────┐
│   Silero VAD    │  ← 语音活动检测
└────────┬────────┘
         ↓
┌─────────────────┐
│    Chunker      │  ← 分块 + overlap
└────────┬────────┘
         ↓
┌─────────────────┐
│ faster-whisper  │  ← 流式 ASR
└────────┬────────┘
         ↓
┌─────────────────┐
│ Text Stabilizer │  ← partial 稳定化
└────────┬────────┘
         ↓
┌─────────────────┐
│  NLLB / MT      │  ← 仅翻译 final
└────────┬────────┘
         ↓
┌─────────────────┐
│  Subtitle UI    │  ← 双语字幕浮窗
└─────────────────┘
```

---

## 0. 全局硬约束

- 音频采集线程不阻塞  
- 各模块解耦，仅通过队列通信  
- 只处理增量  
- 队列积压时丢弃 partial，保留 final  
- 禁止在 UI 线程进行 ASR/MT 推理  
- 必须记录每段延迟打点  

---

## 1. 数据流

Audio Capture → Ring Buffer → VAD → Chunker → Streaming ASR → Text Stabilizer → MT → Subtitle UI

---

## 2. 线程模型

- Thread 1：Audio Capture  
- Thread 2：VAD + Chunker  
- Thread 3：ASR Worker  
- Thread 4：MT Worker  
- Thread 5：UI Main Thread  

通信方式  
- 线程之间仅通过有界 Queue 通信  
- 禁止共享可变对象（只传不可变数据结构或拷贝）  

---

## 3. 全局固定设置

音频固定设置  
- sample_rate = 16000  
- channels = 1  
- dtype = float32  
- frame_ms = 20  
- frame_samples = 320  

队列固定设置  
- q_chunks_max = 32  
- q_asr_max = 32  
- q_mt_max = 32  
- q_ui_max = 128  

刷新节流  
- ui_refresh_hz = 10  

---

## 4. 模块定义与参数

### 4.1 Audio Capture

输入源
- mode = loopback（系统内录）
- mode = mic（麦克风）

输出
- 每 20ms 输出 320 样本 float32
- 实时 push 到 Ring Buffer

接口
- start()
- read_frame() -> float32[320]
- stop()

失败处理
- 采集异常时立即重启采集设备
- 重启次数超过 5 次则退出并输出错误码

**Python 类定义**

```python
@dataclass
class AudioFrame:
    """音频帧数据"""
    samples: np.ndarray  # float32[320]
    timestamp: float     # 采集时间戳

class AudioCapture:
    """音频采集器"""

    def __init__(self, mode: str = "loopback",
                 device_index: int = None,
                 sample_rate: int = 16000,
                 frame_ms: int = 20):
        self.mode = mode
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.frame_samples = sample_rate * frame_ms // 1000
        self._stream = None
        self._running = False
        self._restart_count = 0
        self._max_restarts = 5

    def start(self) -> None:
        """启动音频采集"""
        pass

    def read_frame(self) -> Optional[AudioFrame]:
        """读取一帧音频（阻塞）"""
        pass

    def stop(self) -> None:
        """停止音频采集"""
        pass

    def _on_error(self, error: Exception) -> None:
        """错误处理：自动重启"""
        pass
```  

---

### 4.2 Ring Buffer

容量
- buffer_seconds = 30
- buffer_samples = 30 * 16000

行为
- push 追加到尾部
- pop 从头部取走
- peek 只读取不移除
- 溢出时丢弃最旧数据

接口
- push(samples)
- pop(n_samples) -> samples
- peek(n_samples) -> samples
- clear()

**Python 类定义**

```python
class RingBuffer:
    """线程安全的环形缓冲区"""

    def __init__(self, capacity_seconds: float = 30.0,
                 sample_rate: int = 16000):
        self.capacity = int(capacity_seconds * sample_rate)
        self._buffer = np.zeros(self.capacity, dtype=np.float32)
        self._write_pos = 0
        self._read_pos = 0
        self._lock = threading.Lock()

    def push(self, samples: np.ndarray) -> int:
        """追加样本，返回实际写入数量（溢出时丢弃最旧）"""
        pass

    def pop(self, n_samples: int) -> np.ndarray:
        """取出 n 个样本（移除）"""
        pass

    def peek(self, n_samples: int) -> np.ndarray:
        """读取 n 个样本（不移除）"""
        pass

    def clear(self) -> None:
        """清空缓冲区"""
        pass

    @property
    def available(self) -> int:
        """当前可读样本数"""
        pass

    @property
    def usage_ratio(self) -> float:
        """缓冲区使用率"""
        pass
```  

---

### 4.3 VAD

实现
- silero-vad

输入
- 20ms frame（320 samples）

参数
- speech_start_frames = 6
- speech_end_frames = 10
- vad_threshold = 0.50

状态机规则
- 连续 6 帧 speech 判定开始
- 连续 10 帧 non-speech 判定结束

输出事件
- VAD_START(t)
- VAD_SPEECH_FRAME(frame, t)
- VAD_END(t)

**Python 类定义**

```python
class VadState(Enum):
    SILENCE = "silence"
    SPEECH = "speech"

@dataclass
class VadEvent:
    event_type: str  # "start", "frame", "end"
    timestamp: float
    frame: Optional[np.ndarray] = None

class SileroVAD:
    """Silero VAD 封装"""

    def __init__(self,
                 threshold: float = 0.50,
                 speech_start_frames: int = 6,
                 speech_end_frames: int = 10,
                 sample_rate: int = 16000):
        self.threshold = threshold
        self.speech_start_frames = speech_start_frames
        self.speech_end_frames = speech_end_frames
        self.sample_rate = sample_rate

        self._model = None  # Silero VAD model
        self._state = VadState.SILENCE
        self._speech_counter = 0
        self._silence_counter = 0

    def load_model(self) -> None:
        """加载 Silero VAD 模型"""
        import torch
        self._model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )

    def process_frame(self, frame: np.ndarray,
                      timestamp: float) -> Optional[VadEvent]:
        """处理一帧音频，返回 VAD 事件（如有）"""
        pass

    def reset(self) -> None:
        """重置状态机"""
        self._state = VadState.SILENCE
        self._speech_counter = 0
        self._silence_counter = 0
```  

---

### 4.4 Chunker

输入
- 仅接收 VAD 判定为 speech 的帧

参数
- chunk_ms = 1000
- overlap_ms = 200
- max_utterance_ms = 10000
- tail_pad_ms = 120

派生值
- chunk_samples = 16000
- overlap_samples = 3200
- max_utterance_samples = 160000
- tail_pad_samples = 1920

规则
- speech 状态期间，每累计 chunk_samples 输出一个 partial chunk
- 每个 chunk 头部拼接 overlap_samples 的历史音频
- utterance 结束时输出 final chunk
- final chunk 尾部追加 tail_pad_samples 静音

输出结构
- ASRChunk.audio: float32[n]
- ASRChunk.t_start: float
- ASRChunk.t_end: float
- ASRChunk.is_final: bool

丢弃规则
- q_chunks 满时丢弃 partial chunk
- q_chunks 满时保留 final chunk（强制入队：丢弃队头 partial 直到有空间）

**Python 类定义**

```python
@dataclass
class ASRChunk:
    """ASR 输入块"""
    audio: np.ndarray     # float32[n]
    t_start: float        # 起始时间戳
    t_end: float          # 结束时间戳
    is_final: bool        # 是否为 utterance 结束块
    t_emit: float = 0.0   # 出队时间（用于延迟计算）

class Chunker:
    """音频分块器"""

    def __init__(self,
                 chunk_ms: int = 1000,
                 overlap_ms: int = 200,
                 max_utterance_ms: int = 10000,
                 tail_pad_ms: int = 120,
                 sample_rate: int = 16000):
        self.chunk_samples = chunk_ms * sample_rate // 1000
        self.overlap_samples = overlap_ms * sample_rate // 1000
        self.max_utterance_samples = max_utterance_ms * sample_rate // 1000
        self.tail_pad_samples = tail_pad_ms * sample_rate // 1000
        self.sample_rate = sample_rate

        self._buffer = []
        self._overlap_buffer = np.zeros(self.overlap_samples, dtype=np.float32)
        self._utterance_start_time = 0.0

    def on_vad_start(self, timestamp: float) -> None:
        """VAD 开始事件"""
        self._buffer = []
        self._utterance_start_time = timestamp

    def on_speech_frame(self, frame: np.ndarray,
                        timestamp: float) -> Optional[ASRChunk]:
        """处理语音帧，可能返回 partial chunk"""
        pass

    def on_vad_end(self, timestamp: float) -> Optional[ASRChunk]:
        """VAD 结束事件，返回 final chunk"""
        pass

    def reset(self) -> None:
        """重置状态"""
        self._buffer = []
```  

---

### 4.5 Streaming ASR

实现
- faster-whisper + CTranslate2

固定设置
- model_size = small
- language = en
- beam_size = 1
- temperature = 0
- best_of = 1
- vad_filter = False
- word_timestamps = False

设备设置
- device = cuda（有 NVIDIA GPU）
- device = cpu（无 GPU）

精度设置
- compute_type = float16（cuda）
- compute_type = int8（cpu）

输入
- ASRChunk.audio

输出结构
- ASRResult.text: str
- ASRResult.segments: list
- ASRResult.t_start: float
- ASRResult.t_end: float
- ASRResult.is_final: bool

行为
- 每个 chunk 独立推理
- partial chunk 输出 partial ASRResult
- final chunk 输出 final ASRResult

丢弃规则
- q_asr 满时丢弃 partial ASRResult
- q_asr 满时保留 final ASRResult（同 chunk 队列策略）

**Python 类定义**

```python
@dataclass
class ASRResult:
    """ASR 识别结果"""
    text: str                 # 识别文本
    segments: List[dict]      # 分段信息
    t_start: float            # 音频起始时间
    t_end: float              # 音频结束时间
    is_final: bool            # 是否为 final 结果
    t_asr_start: float = 0.0  # ASR 开始时间
    t_asr_end: float = 0.0    # ASR 结束时间

class ASRWorker(threading.Thread):
    """ASR 工作线程"""

    def __init__(self,
                 input_queue: Queue,
                 output_queue: Queue,
                 model_size: str = "small",
                 language: str = "en",
                 device: str = "cuda",
                 compute_type: str = "float16"):
        super().__init__(daemon=True)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = compute_type

        self._model = None
        self._running = False

    def load_model(self) -> None:
        """加载 faster-whisper 模型"""
        from faster_whisper import WhisperModel
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type
        )

    def run(self) -> None:
        """工作线程主循环"""
        self._running = True
        while self._running:
            try:
                chunk = self.input_queue.get(timeout=0.1)
                result = self._transcribe(chunk)
                self._enqueue_result(result)
            except Empty:
                continue
            except Exception as e:
                self._handle_error(e)

    def _transcribe(self, chunk: ASRChunk) -> ASRResult:
        """执行 ASR 推理"""
        t_start = time.time()
        segments, info = self._model.transcribe(
            chunk.audio,
            language=self.language,
            beam_size=1,
            temperature=0,
            vad_filter=False,
            word_timestamps=False
        )
        text = " ".join([s.text for s in segments])
        t_end = time.time()

        return ASRResult(
            text=text.strip(),
            segments=[],
            t_start=chunk.t_start,
            t_end=chunk.t_end,
            is_final=chunk.is_final,
            t_asr_start=t_start,
            t_asr_end=t_end
        )

    def stop(self) -> None:
        """停止工作线程"""
        self._running = False
```  

---

### 4.6 Text Stabilizer

目标
- 生成稳定字幕
- 降低 partial 抖动
- 提供 final 句子增量

状态
- locked_text: str
- buffer_text: str

参数
- lock_min_chars = 12
- buffer_keep_chars = 80
- lcp_min_chars = 8

处理规则（每次收到 ASRResult）
- 将 ASRResult.text 视为当前候选文本 candidate
- 计算 candidate 与 (locked_text + buffer_text) 的最长公共前缀 LCP
- 若 LCP 长度 >= lcp_min_chars，则推进 locked_text 到 LCP 的末尾
- 将剩余文本作为 buffer_text
- buffer_text 只保留末尾 buffer_keep_chars
- final 时将 buffer_text 全部并入 locked_text，并生成 final_append

输出
- partial_src = locked_text + buffer_text
- final_append_src（仅 is_final=True 时产生）

**Python 类定义**

```python
@dataclass
class StabilizerOutput:
    """稳定器输出"""
    partial_src: str              # 当前显示文本
    final_append_src: Optional[str] = None  # final 增量（仅 final 时）
    timestamp: float = 0.0

class TextStabilizer:
    """文本稳定器 - 减少 partial 抖动"""

    def __init__(self,
                 lock_min_chars: int = 12,
                 buffer_keep_chars: int = 80,
                 lcp_min_chars: int = 8):
        self.lock_min_chars = lock_min_chars
        self.buffer_keep_chars = buffer_keep_chars
        self.lcp_min_chars = lcp_min_chars

        self._locked_text = ""
        self._buffer_text = ""
        self._last_final_pos = 0

    def process(self, asr_result: ASRResult) -> StabilizerOutput:
        """处理 ASR 结果，返回稳定化输出"""
        candidate = asr_result.text
        current = self._locked_text + self._buffer_text

        # 计算最长公共前缀
        lcp = self._longest_common_prefix(current, candidate)

        if len(lcp) >= self.lcp_min_chars:
            # 推进 locked_text
            if len(lcp) > len(self._locked_text):
                self._locked_text = lcp[:len(lcp)]

        # 更新 buffer
        self._buffer_text = candidate[len(self._locked_text):]
        if len(self._buffer_text) > self.buffer_keep_chars:
            self._buffer_text = self._buffer_text[-self.buffer_keep_chars:]

        partial_src = self._locked_text + self._buffer_text

        # Final 处理
        final_append_src = None
        if asr_result.is_final:
            final_append_src = candidate[self._last_final_pos:]
            self._locked_text = candidate
            self._buffer_text = ""
            self._last_final_pos = len(candidate)

        return StabilizerOutput(
            partial_src=partial_src,
            final_append_src=final_append_src,
            timestamp=time.time()
        )

    def _longest_common_prefix(self, s1: str, s2: str) -> str:
        """计算最长公共前缀"""
        i = 0
        while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
            i += 1
        return s1[:i]

    def reset(self) -> None:
        """重置状态"""
        self._locked_text = ""
        self._buffer_text = ""
        self._last_final_pos = 0
```  

---

### 4.7 MT（Text → Text）

模式
- 只翻译 final_append_src
- 不翻译 partial

输入
- final_append_src（英文）

输出
- final_append_tgt（中文）

参数
- src_lang = en
- tgt_lang = zh
- batch_size = 1
- max_input_chars = 300

缓存
- cache_enabled = True
- cache_size = 2048
- key = exact final_append_src

失败规则
- 翻译失败时输出空字符串
- 连续失败 3 次重启 MT worker

**MT 模型选择**

| 模型 | 大小 | 速度 | 质量 | 推荐场景 |
|------|------|------|------|---------|
| NLLB-200-distilled-600M | 1.2GB | 快 | 中 | 低显存/CPU |
| NLLB-200-1.3B | 2.6GB | 中 | 高 | GPU 推荐 |
| M2M-100-418M | 0.8GB | 很快 | 中低 | 极低延迟 |

**Python 类定义**

```python
@dataclass
class MTResult:
    """翻译结果"""
    source: str           # 原文
    target: str           # 译文
    t_mt_start: float     # 翻译开始时间
    t_mt_end: float       # 翻译结束时间

class MTWorker(threading.Thread):
    """MT 工作线程"""

    def __init__(self,
                 input_queue: Queue,
                 output_queue: Queue,
                 model_name: str = "facebook/nllb-200-distilled-600M",
                 src_lang: str = "eng_Latn",
                 tgt_lang: str = "zho_Hans",
                 device: str = "cuda",
                 cache_size: int = 2048):
        super().__init__(daemon=True)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.device = device
        self.cache_size = cache_size

        self._model = None
        self._tokenizer = None
        self._cache = {}  # LRU cache
        self._running = False
        self._error_count = 0
        self._max_errors = 3

    def load_model(self) -> None:
        """加载翻译模型"""
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name
        ).to(self.device)

    def run(self) -> None:
        """工作线程主循环"""
        self._running = True
        while self._running:
            try:
                text = self.input_queue.get(timeout=0.1)
                result = self._translate(text)
                self.output_queue.put(result)
                self._error_count = 0
            except Empty:
                continue
            except Exception as e:
                self._handle_error(e)

    def _translate(self, text: str) -> MTResult:
        """执行翻译"""
        # 检查缓存
        if text in self._cache:
            cached = self._cache[text]
            return MTResult(
                source=text,
                target=cached,
                t_mt_start=time.time(),
                t_mt_end=time.time()
            )

        t_start = time.time()

        # 截断过长文本
        if len(text) > 300:
            text = text[:300]

        inputs = self._tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self._model.generate(
            **inputs,
            forced_bos_token_id=self._tokenizer.lang_code_to_id[self.tgt_lang],
            max_length=512
        )
        translated = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        t_end = time.time()

        # 更新缓存
        self._update_cache(text, translated)

        return MTResult(
            source=text,
            target=translated,
            t_mt_start=t_start,
            t_mt_end=t_end
        )

    def _update_cache(self, key: str, value: str) -> None:
        """更新 LRU 缓存"""
        if len(self._cache) >= self.cache_size:
            # 简单策略：删除第一个
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = value

    def _handle_error(self, error: Exception) -> None:
        """错误处理"""
        self._error_count += 1
        if self._error_count >= self._max_errors:
            self._restart()

    def stop(self) -> None:
        """停止工作线程"""
        self._running = False
```  

---

### 4.8 Subtitle UI（浮窗）

显示模式  
- 两行显示  
  - Line 1：source（英文）  
  - Line 2：target（中文）  

更新规则  
- UI 每秒最多刷新 10 次  
- UI 合并同一秒内的多次更新，只显示最新状态  

窗口属性  
- always_on_top = True  
- transparent = True  
- click_through = True  
- draggable = True  

输入  
- partial_src（可选显示）  
- final_append_src  
- final_append_tgt  

显示规则  
- partial_src 显示为临时字幕  
- final_append 追加到历史字幕区  
- 历史字幕区最多保留最近 50 行  

---

## 5. 事件与队列

事件定义  
- AudioFrame(samples, t)  
- VadStart(t)  
- VadEnd(t)  
- ASRChunk(audio, t0, t1, is_final)  
- ASRResult(text, segments, t0, t1, is_final)  
- FinalAppendSrc(text, t)  
- FinalAppendTgt(text, t)  
- UIUpdate(partial_src, final_append_src, final_append_tgt, t)  

队列  
- q_chunks: Chunker → ASR  
- q_asr: ASR → Stabilizer  
- q_mt: Stabilizer → MT  
- q_ui: Stabilizer/MT → UI  

队列策略  
- 满时丢 partial  
- 满时保 final  

---

## 6. 延迟打点（必须输出到日志）

时间戳字段  
- t_capture（采集到 frame）  
- t_vad（VAD 判定）  
- t_chunk_emit（chunk 出队）  
- t_asr_start / t_asr_end  
- t_stabilize  
- t_mt_start / t_mt_end  
- t_ui_render  

输出指标  
- latency_asr = t_asr_end - t_chunk_emit  
- latency_mt = t_mt_end - t_mt_start  
- latency_final_e2e = t_ui_render - t_capture（以 final 为准）  

---

## 7. 开发步骤（按顺序执行）

### Step 1: 音频采集基础

**目标**
- 实现 WASAPI 采集
- 输出 RMS 到控制台
- 连续运行 10 分钟不中断

**实现文件**: `src/core/audio_capture.py`

**测试脚本**:
```python
# test_step1_audio.py
from src.core.audio_capture import AudioCapture
import time

capture = AudioCapture(mode="loopback")
capture.start()

start_time = time.time()
while time.time() - start_time < 600:  # 10 分钟
    frame = capture.read_frame()
    if frame:
        rms = np.sqrt(np.mean(frame.samples ** 2))
        print(f"[{time.time():.2f}] RMS: {rms:.4f}")

capture.stop()
print("Step 1 PASSED: 10 分钟无中断")
```

**验收标准**:
- [ ] 无设备断开错误
- [ ] RMS 数值正常（有声音时 > 0.01）
- [ ] 内存无泄漏

---

### Step 2: 环形缓冲区

**目标**
- 接入 Ring Buffer
- 连续 push/pop 不丢样本
- 记录 buffer 使用率

**实现文件**: `src/core/ring_buffer.py`

**测试脚本**:
```python
# test_step2_buffer.py
from src.core.ring_buffer import RingBuffer
import numpy as np

buffer = RingBuffer(capacity_seconds=30.0)

# 模拟 10 分钟数据流
for i in range(30000):  # 30000 * 20ms = 600s
    samples = np.random.randn(320).astype(np.float32)
    written = buffer.push(samples)

    if buffer.available >= 320:
        data = buffer.pop(320)
        assert len(data) == 320

    if i % 1000 == 0:
        print(f"Usage: {buffer.usage_ratio:.2%}")

print("Step 2 PASSED: 无样本丢失")
```

**验收标准**:
- [ ] push/pop 数据完整
- [ ] 使用率统计正确
- [ ] 线程安全（多线程测试）

---

### Step 3: VAD 集成

**目标**
- 接入 VAD
- 控制台打印 VAD_START/VAD_END
- 连续运行 10 分钟

**实现文件**: `src/core/vad.py`

**测试脚本**:
```python
# test_step3_vad.py
from src.core.audio_capture import AudioCapture
from src.core.vad import SileroVAD

capture = AudioCapture(mode="loopback")
vad = SileroVAD(threshold=0.5)
vad.load_model()

capture.start()
start_time = time.time()

while time.time() - start_time < 600:
    frame = capture.read_frame()
    if frame:
        event = vad.process_frame(frame.samples, frame.timestamp)
        if event:
            print(f"[{event.timestamp:.2f}] {event.event_type.upper()}")

capture.stop()
print("Step 3 PASSED")
```

**验收标准**:
- [ ] VAD_START/VAD_END 成对出现
- [ ] 静音时无误触发
- [ ] 响应延迟 < 200ms

---

### Step 4: Chunker 实现

**目标**
- 实现 Chunker
- 打印每个 chunk 的时长与 is_final
- 验证 chunk_ms 与 overlap_ms 生效

**实现文件**: `src/core/chunker.py`

**测试脚本**:
```python
# test_step4_chunker.py
from src.core.chunker import Chunker

chunker = Chunker(chunk_ms=1000, overlap_ms=200)

# 模拟 VAD 事件流
chunker.on_vad_start(0.0)

for i in range(100):  # 2 秒语音
    frame = np.random.randn(320).astype(np.float32)
    chunk = chunker.on_speech_frame(frame, i * 0.02)
    if chunk:
        duration = len(chunk.audio) / 16000
        print(f"Chunk: {duration:.2f}s, final={chunk.is_final}")

final_chunk = chunker.on_vad_end(2.0)
print(f"Final Chunk: {len(final_chunk.audio)/16000:.2f}s")
```

**验收标准**:
- [ ] partial chunk 约 1s
- [ ] overlap 正确拼接
- [ ] final chunk 包含 tail padding

---

### Step 5: ASR 集成

**目标**
- 接入 faster-whisper
- 打印 ASRResult.text
- 计算 asr RTF（推理耗时 / 音频时长）
- 目标 RTF < 1.0

**实现文件**: `src/core/asr_worker.py`

**测试脚本**:
```python
# test_step5_asr.py
from faster_whisper import WhisperModel
import numpy as np
import time

model = WhisperModel("small", device="cuda", compute_type="float16")

# 测试 1 秒音频
audio = np.random.randn(16000).astype(np.float32)

t_start = time.time()
segments, _ = model.transcribe(audio, language="en", beam_size=1)
text = " ".join([s.text for s in segments])
t_end = time.time()

rtf = (t_end - t_start) / 1.0
print(f"Text: {text}")
print(f"RTF: {rtf:.2f} (target < 1.0)")
```

**验收标准**:
- [ ] RTF < 1.0（GPU）
- [ ] RTF < 3.0（CPU）
- [ ] 文本输出正常

---

### Step 6: Text Stabilizer

**目标**
- 实现 Stabilizer
- partial_src 输出不抖动
- final_append_src 在句尾产生

**实现文件**: `src/core/text_stabilizer.py`

**测试脚本**:
```python
# test_step6_stabilizer.py
from src.core.text_stabilizer import TextStabilizer

stabilizer = TextStabilizer()

# 模拟 ASR 输出序列
texts = [
    "Hello",
    "Hello world",
    "Hello world how",
    "Hello world how are",
    "Hello world how are you",  # final
]

for i, text in enumerate(texts):
    is_final = (i == len(texts) - 1)
    result = ASRResult(text=text, is_final=is_final, ...)
    output = stabilizer.process(result)
    print(f"partial: '{output.partial_src}'")
    if output.final_append_src:
        print(f"FINAL: '{output.final_append_src}'")
```

**验收标准**:
- [ ] partial 前缀稳定
- [ ] final_append 只在句尾产生
- [ ] 无文本丢失

---

### Step 7: MT 集成

**目标**
- 接入 MT
- 仅对 final_append_src 翻译
- 输出 final_append_tgt

**实现文件**: `src/core/mt_worker.py`

**测试脚本**:
```python
# test_step7_mt.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")

text = "Hello world, how are you today?"
inputs = tokenizer(text, return_tensors="pt").to("cuda")

t_start = time.time()
outputs = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.lang_code_to_id["zho_Hans"],
    max_length=128
)
translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
t_end = time.time()

print(f"Source: {text}")
print(f"Target: {translated}")
print(f"Time: {t_end - t_start:.2f}s")
```

**验收标准**:
- [ ] 翻译质量可接受
- [ ] 单句翻译 < 500ms（GPU）
- [ ] 缓存命中时 < 1ms

---

### Step 8: 浮窗 UI

**目标**
- 实现浮窗 UI
- 显示两行字幕
- 支持置顶、透明、可穿透、拖拽
- UI 刷新限制 10Hz

**实现文件**: `src/ui/subtitle_window.py`

**关键代码**:
```python
class SubtitleWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 窗口属性
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowTransparentForInput  # 可穿透
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # 刷新限流
        self._last_update = 0
        self._min_interval = 0.1  # 10Hz

    def update_subtitle(self, partial_src: str,
                        final_src: str, final_tgt: str):
        now = time.time()
        if now - self._last_update < self._min_interval:
            return
        self._last_update = now
        # 更新显示...
```

**验收标准**:
- [ ] 窗口置顶
- [ ] 背景透明
- [ ] 鼠标可穿透
- [ ] 支持拖拽
- [ ] 刷新率 ≤ 10Hz

---

### Step 9: 错误恢复

**目标**
- 加入崩溃重启逻辑
- ASR worker 异常自动重启
- MT worker 异常自动重启
- 设备断开自动重连

**实现文件**: `src/core/supervisor.py`

**关键代码**:
```python
class WorkerSupervisor:
    """工作线程监控器"""

    def __init__(self, worker_class, *args, **kwargs):
        self.worker_class = worker_class
        self.args = args
        self.kwargs = kwargs
        self._worker = None
        self._restart_count = 0
        self._max_restarts = 5
        self._restart_delay = 1.0  # 秒

    def start(self):
        self._worker = self.worker_class(*self.args, **self.kwargs)
        self._worker.start()
        self._start_monitor()

    def _start_monitor(self):
        """监控线程健康状态"""
        def monitor():
            while True:
                if not self._worker.is_alive():
                    self._on_worker_died()
                time.sleep(1.0)
        threading.Thread(target=monitor, daemon=True).start()

    def _on_worker_died(self):
        if self._restart_count < self._max_restarts:
            time.sleep(self._restart_delay)
            self._restart_count += 1
            self.start()
        else:
            raise RuntimeError("Worker 重启次数超限")
```

**验收标准**:
- [ ] Worker 崩溃自动重启
- [ ] 重启次数限制生效
- [ ] 设备断开自动重连
- [ ] 日志记录完整  

---

## 8. 配置文件（config.yaml 固定字段）

必须字段  
- input_mode: loopback | mic  
- sample_rate: 16000  
- frame_ms: 20  
- vad_threshold: 0.50  
- speech_start_frames: 6  
- speech_end_frames: 10  
- chunk_ms: 1000  
- overlap_ms: 200  
- max_utterance_ms: 10000  
- tail_pad_ms: 120  
- asr_model_size: small  
- asr_language: en  
- asr_beam_size: 1  
- asr_temperature: 0  
- asr_compute_type: float16 | int8  
- mt_src_lang: en  
- mt_tgt_lang: zh  
- ui_refresh_hz: 10  
- ui_history_lines: 50  
- queue_max: 32  

---

## 9. 验收标准

- 系统内录模式连续运行 30 分钟不崩溃
- partial 字幕持续滚动
- 句尾停顿后产生 final 译文
- final 字幕端到端延迟稳定且可测
- UI 置顶透明可穿透可拖拽
- 日志包含所有打点字段

---

## 10. 性能优化建议

### 10.1 ASR 优化

| 优化项 | 方法 | 预期提升 |
|-------|------|---------|
| 模型大小 | tiny/base 代替 small | RTF -50% |
| beam_size | 1（已设置） | 最快 |
| 批处理 | 多 chunk 合并推理 | 吞吐 +30% |
| 量化 | int8（CPU）/ float16（GPU） | 速度 +20% |

### 10.2 MT 优化

| 优化项 | 方法 | 预期提升 |
|-------|------|---------|
| 模型选择 | NLLB-distilled-600M | 平衡速度/质量 |
| 缓存 | LRU 缓存常见短语 | 重复句 0ms |
| 批处理 | 积累多句后批量翻译 | 吞吐 +50% |
| 量化 | 8-bit 量化 | 显存 -50% |

### 10.3 内存优化

```python
# 1. 限制队列大小
q_chunks = queue.Queue(maxsize=32)

# 2. 及时释放大数组
del audio_array
gc.collect()

# 3. 使用 numpy 视图而非拷贝
chunk = buffer[start:end]  # 视图
chunk = buffer[start:end].copy()  # 仅在必要时拷贝
```

### 10.4 延迟优化

**目标延迟分解**：
```
总延迟 < 2000ms

├── 音频采集      ~20ms（1 frame）
├── VAD 判定      ~120ms（6 frames）
├── Chunk 累积    ~1000ms
├── ASR 推理      ~300ms（RTF=0.3）
├── 稳定化        ~1ms
├── MT 翻译       ~200ms
└── UI 渲染       ~10ms
```

**优化策略**：
1. 减小 chunk_ms（牺牲识别质量）
2. 减少 speech_start_frames（误触发风险）
3. 使用更小的 ASR 模型
4. MT 仅翻译 final

---

## 11. 错误处理策略

### 11.1 错误分类

| 级别 | 类型 | 处理方式 |
|-----|------|---------|
| FATAL | CUDA OOM | 退出程序 |
| ERROR | 模型加载失败 | 重试 3 次后退出 |
| WARN | 音频设备断开 | 自动重连 |
| WARN | 单次推理超时 | 跳过当前 chunk |
| INFO | 队列丢弃 partial | 记录日志 |

### 11.2 重试策略

```python
def with_retry(func, max_retries=3, delay=1.0):
    """带重试的函数装饰器"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay * (attempt + 1))
```

### 11.3 优雅降级

```python
# GPU 不可用时降级到 CPU
try:
    model = load_model(device="cuda")
except RuntimeError:
    logger.warning("CUDA 不可用，降级到 CPU")
    model = load_model(device="cpu", compute_type="int8")
```

---

## 12. 日志规范

### 12.1 日志格式

```
[2024-01-15 10:30:45.123] [INFO] [audio_capture] 设备已连接: index=3
[2024-01-15 10:30:46.456] [DEBUG] [vad] VAD_START t=1.234
[2024-01-15 10:30:48.789] [INFO] [asr] ASR完成: latency=0.312s, text="Hello world"
[2024-01-15 10:30:49.012] [WARN] [mt] 翻译超时，跳过
```

### 12.2 延迟打点日志

```python
@dataclass
class LatencyLog:
    """延迟打点记录"""
    utterance_id: str
    t_capture: float
    t_vad_start: float
    t_chunk_emit: float
    t_asr_start: float
    t_asr_end: float
    t_stabilize: float
    t_mt_start: float
    t_mt_end: float
    t_ui_render: float

    @property
    def latency_asr(self) -> float:
        return self.t_asr_end - self.t_chunk_emit

    @property
    def latency_mt(self) -> float:
        return self.t_mt_end - self.t_mt_start

    @property
    def latency_e2e(self) -> float:
        return self.t_ui_render - self.t_capture
```

---

## 附录 A: 完整 config.yaml 示例

```yaml
# 音频设置
audio:
  input_mode: loopback  # loopback | mic
  device_index: null    # null 为自动选择
  sample_rate: 16000
  frame_ms: 20

# VAD 设置
vad:
  threshold: 0.50
  speech_start_frames: 6
  speech_end_frames: 10

# 分块设置
chunker:
  chunk_ms: 1000
  overlap_ms: 200
  max_utterance_ms: 10000
  tail_pad_ms: 120

# ASR 设置
asr:
  model_size: small     # tiny | base | small | medium
  language: en
  beam_size: 1
  temperature: 0
  device: cuda          # cuda | cpu
  compute_type: float16 # float16 | int8

# MT 设置
mt:
  model_name: facebook/nllb-200-distilled-600M
  src_lang: eng_Latn
  tgt_lang: zho_Hans
  device: cuda
  cache_size: 2048

# 稳定器设置
stabilizer:
  lock_min_chars: 12
  buffer_keep_chars: 80
  lcp_min_chars: 8

# UI 设置
ui:
  refresh_hz: 10
  history_lines: 50
  font_size: 24
  opacity: 0.8

# 队列设置
queues:
  max_size: 32

# 日志设置
logging:
  level: INFO
  file: logs/app.log
  latency_file: logs/latency.csv
```

---

## 附录 B: 语言代码对照表

### ASR (Whisper) 语言代码

| 代码 | 语言 |
|-----|------|
| en | 英语 |
| zh | 中文 |
| ja | 日语 |
| ko | 韩语 |
| es | 西班牙语 |
| fr | 法语 |
| de | 德语 |

### MT (NLLB) 语言代码

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

---

## 更新日志

| 版本 | 日期 | 变更 |
|-----|------|------|
| v0.1 | - | 初始文档 |
| v0.2 | - | 添加项目结构、依赖库、Python 类定义 |
| v0.3 | - | 添加详细开发步骤、测试脚本、验收标准 |
| v0.4 | - | 添加性能优化、错误处理、日志规范 |  
