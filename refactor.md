# 实时低延迟翻译应用开发文档（Laptop 本地运行）

目标  
- 笔记本电脑本地运行  
- 端到端延迟尽可能低  
- 语音 → 字幕翻译（S2TT）  
- 不做语音 → 语音（S2ST）  

范围  
- Windows 优先  
- 系统内录（会议声音）与麦克风输入  
- 双语字幕浮窗（置顶、透明、可穿透）  

## 总体处理流程
Audio (stream)
↓
Whisper / faster-whisper（流式 ASR）
↓
断句 / 稳定化（VAD + Chunk + Stabilizer + Punctuation）
↓
NLLB / 轻量 MT 模型（文本翻译）
↓
字幕显示（Subtitle UI）
↓
TTS（可选）
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

Step 1  
- 实现 WASAPI 采集  
- 输出 RMS 到控制台  
- 连续运行 10 分钟不中断  

Step 2  
- 接入 Ring Buffer  
- 连续 push/pop 不丢样本  
- 记录 buffer 使用率  

Step 3  
- 接入 VAD  
- 控制台打印 VAD_START/VAD_END  
- 连续运行 10 分钟  

Step 4  
- 实现 Chunker  
- 打印每个 chunk 的时长与 is_final  
- 验证 chunk_ms 与 overlap_ms 生效  

Step 5  
- 接入 faster-whisper  
- 打印 ASRResult.text  
- 计算 asr RTF（推理耗时 / 音频时长）  
- 目标 RTF < 1.0  

Step 6  
- 实现 Stabilizer  
- partial_src 输出不抖动  
- final_append_src 在句尾产生  

Step 7  
- 接入 MT  
- 仅对 final_append_src 翻译  
- 输出 final_append_tgt  

Step 8  
- 实现浮窗 UI  
- 显示两行字幕  
- 支持置顶、透明、可穿透、拖拽  
- UI 刷新限制 10Hz  

Step 9  
- 加入崩溃重启逻辑  
- ASR worker 异常自动重启  
- MT worker 异常自动重启  
- 设备断开自动重连  

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
