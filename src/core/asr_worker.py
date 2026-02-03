"""
ASR 工作线程模块
使用 faster-whisper 进行语音识别
"""
import threading
import time
import logging
from queue import Queue, Empty
from typing import Optional

from src.events import ASRChunk, ASRResult

logger = logging.getLogger(__name__)


class ASRWorker:
    """
    ASR 工作线程
    使用 faster-whisper 进行语音识别
    不继承 threading.Thread，内部管理线程实例以支持重复 start/stop
    """

    def __init__(self,
                 input_queue: Queue,
                 output_queue: Queue,
                 model_size: str = "small",
                 language: str = "en",
                 device: str = "cuda",
                 compute_type: str = "float16"):
        """
        初始化 ASR Worker

        Args:
            input_queue: 输入队列（ASRChunk）
            output_queue: 输出队列（ASRResult）
            model_size: 模型大小 (tiny, base, small, medium, large)
            language: 识别语言
            device: 运行设备 (cuda, cpu)
            compute_type: 计算精度 (float16, int8, float32)
        """
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = compute_type

        self._model = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._error_count = 0
        self._max_errors = 5

    def load_model(self) -> None:
        """加载 faster-whisper 模型"""
        logger.info(f"正在加载 faster-whisper 模型: {self.model_size}")
        logger.info(f"设备: {self.device}, 计算类型: {self.compute_type}")

        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            logger.info("faster-whisper 模型加载成功")
        except Exception as e:
            logger.error(f"加载 faster-whisper 模型失败: {e}")
            # 尝试降级到 CPU
            if self.device == "cuda":
                logger.warning("尝试降级到 CPU 模式...")
                try:
                    from faster_whisper import WhisperModel
                    self._model = WhisperModel(
                        self.model_size,
                        device="cpu",
                        compute_type="int8"
                    )
                    self.device = "cpu"
                    self.compute_type = "int8"
                    logger.info("已降级到 CPU 模式")
                except Exception as e2:
                    logger.error(f"CPU 模式也失败: {e2}")
                    raise
            else:
                raise

    def start(self) -> None:
        """启动工作线程（每次创建新的 Thread 实例）"""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("ASR Worker 已在运行")
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """工作线程主循环"""
        if self._model is None:
            logger.error("ASR 模型未加载")
            return

        logger.info("ASR Worker 已启动")

        while self._running:
            try:
                chunk = self.input_queue.get(timeout=0.1)
                result = self._transcribe(chunk)
                self._enqueue_result(result)
                self._error_count = 0
            except Empty:
                continue
            except Exception as e:
                self._handle_error(e)

        logger.info("ASR Worker 已停止")

    def _transcribe(self, chunk: ASRChunk) -> ASRResult:
        """
        执行 ASR 推理

        Args:
            chunk: 输入的音频块

        Returns:
            ASRResult
        """
        t_start = time.time()

        try:
            segments, info = self._model.transcribe(
                chunk.audio,
                language=self.language,
                beam_size=1,
                temperature=0,
                vad_filter=False,
                word_timestamps=False
            )

            # 合并所有 segment 的文本
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())

            text = " ".join(text_parts)

        except Exception as e:
            logger.error(f"ASR 推理错误: {e}")
            text = ""

        t_end = time.time()

        # 计算 RTF
        audio_duration = len(chunk.audio) / 16000
        rtf = (t_end - t_start) / audio_duration if audio_duration > 0 else 0

        logger.debug(f"ASR: '{text[:50]}...' | RTF={rtf:.2f} | final={chunk.is_final}")

        return ASRResult(
            text=text.strip(),
            segments=[],
            t_start=chunk.t_start,
            t_end=chunk.t_end,
            is_final=chunk.is_final,
            t_asr_start=t_start,
            t_asr_end=t_end
        )

    def _enqueue_result(self, result: ASRResult) -> None:
        """
        将结果放入输出队列

        Args:
            result: ASR 结果
        """
        try:
            if result.is_final:
                # final 结果必须入队
                while True:
                    try:
                        self.output_queue.put_nowait(result)
                        break
                    except:
                        # 队列满，丢弃最旧的 partial
                        try:
                            old = self.output_queue.get_nowait()
                            if old.is_final:
                                # 不能丢弃 final，放回去
                                self.output_queue.put_nowait(old)
                                time.sleep(0.01)
                        except Empty:
                            break
            else:
                # partial 结果可以丢弃
                self.output_queue.put_nowait(result)
        except:
            logger.warning("ASR 输出队列满，丢弃 partial 结果")

    def _handle_error(self, error: Exception) -> None:
        """错误处理"""
        logger.error(f"ASR 错误: {error}")
        self._error_count += 1

        if self._error_count >= self._max_errors:
            logger.critical(f"ASR 错误次数超限 ({self._max_errors})")
            self._running = False

    def stop(self) -> None:
        """停止工作线程"""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    @property
    def is_running(self) -> bool:
        """是否正在运行"""
        return self._running and self._thread is not None and self._thread.is_alive()
