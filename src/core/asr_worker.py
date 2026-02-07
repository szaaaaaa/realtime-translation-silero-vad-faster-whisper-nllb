"""
ASR worker based on faster-whisper.
"""
import logging
import re
import threading
import time
from queue import Empty, Queue
from typing import Optional

from src.events import ASRChunk, ASRResult

logger = logging.getLogger(__name__)


class ASRWorker:
    """
    ASR worker managed by an internal daemon thread.
    Supports repeated start/stop cycles.
    """

    def __init__(
        self,
        input_queue: Queue,
        output_queue: Queue,
        model_size: str = "small",
        language: str = "en",
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 2,
        beam_size_final: int = 5,
        temperature: float = 0.0,
        word_timestamps: bool = False,
        condition_on_previous_text: bool = False,
        compression_ratio_threshold: float = 2.4,
        log_prob_threshold: float = -1.0,
        no_speech_threshold: float = 0.6,
        suppress_blank: bool = True,
        suppress_tokens: str = "-1",
        initial_prompt: str = "",
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = compute_type

        self.beam_size = max(1, int(beam_size))
        self.beam_size_final = max(self.beam_size, int(beam_size_final or self.beam_size))
        self.temperature = float(temperature)
        self.word_timestamps = bool(word_timestamps)
        self.condition_on_previous_text = bool(condition_on_previous_text)

        self.compression_ratio_threshold = float(compression_ratio_threshold)
        self.log_prob_threshold = float(log_prob_threshold)
        self.no_speech_threshold = float(no_speech_threshold)
        self.suppress_blank = bool(suppress_blank)
        self.suppress_tokens = self._parse_suppress_tokens(suppress_tokens)
        self.initial_prompt = (initial_prompt or "").strip() or None

        self._model = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._error_count = 0
        self._max_errors = 5

    @staticmethod
    def _parse_suppress_tokens(tokens) -> list:
        if tokens is None:
            return [-1]
        if isinstance(tokens, list):
            out = []
            for t in tokens:
                try:
                    out.append(int(t))
                except Exception:
                    continue
            return out or [-1]

        text = str(tokens).strip()
        if not text:
            return [-1]

        out = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                out.append(int(part))
            except Exception:
                continue
        return out or [-1]

    @staticmethod
    def _cleanup_text(text: str) -> str:
        text = re.sub(r"\s+", " ", (text or "")).strip()
        if not text:
            return ""

        # Reduce obvious punctuation hallucination bursts.
        text = re.sub(r"([!?.,])\1{4,}", r"\1\1\1", text)

        non_word = len(re.findall(r"[^\w\u4e00-\u9fff]", text))
        if len(text) >= 8 and non_word / max(1, len(text)) > 0.65:
            return ""
        return text

    def load_model(self) -> None:
        """Load faster-whisper model."""
        logger.info(f"Loading faster-whisper model: {self.model_size}")
        logger.info(f"Device: {self.device}, compute_type: {self.compute_type}")

        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info("faster-whisper model loaded")
        except Exception as e:
            logger.error(f"Failed to load faster-whisper model: {e}")
            if self.device == "cuda":
                logger.warning("Retrying ASR model on CPU int8...")
                from faster_whisper import WhisperModel

                self._model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                )
                self.device = "cpu"
                self.compute_type = "int8"
                logger.info("ASR model downgraded to CPU int8")
            else:
                raise

    def start(self) -> None:
        """Start worker thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("ASR Worker already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """Main loop."""
        if self._model is None:
            logger.error("ASR model is not loaded")
            return

        logger.info("ASR Worker started")

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

        logger.info("ASR Worker stopped")

    def _transcribe(self, chunk: ASRChunk) -> ASRResult:
        """Run ASR inference for one chunk."""
        t_start = time.time()

        try:
            beam_size = self.beam_size_final if chunk.is_final else self.beam_size
            decode_kwargs = {
                "language": self.language,
                "beam_size": beam_size,
                "temperature": self.temperature,
                "vad_filter": False,
                "word_timestamps": self.word_timestamps,
                "condition_on_previous_text": self.condition_on_previous_text,
                "compression_ratio_threshold": self.compression_ratio_threshold,
                "log_prob_threshold": self.log_prob_threshold,
                "no_speech_threshold": self.no_speech_threshold,
                "suppress_blank": self.suppress_blank,
                "suppress_tokens": self.suppress_tokens,
            }
            if self.initial_prompt:
                decode_kwargs["initial_prompt"] = self.initial_prompt

            try:
                segments, _info = self._model.transcribe(chunk.audio, **decode_kwargs)
            except TypeError:
                # Compatibility with older faster-whisper builds.
                basic_kwargs = {
                    "language": self.language,
                    "beam_size": beam_size,
                    "temperature": self.temperature,
                    "vad_filter": False,
                    "word_timestamps": self.word_timestamps,
                    "condition_on_previous_text": self.condition_on_previous_text,
                }
                segments, _info = self._model.transcribe(chunk.audio, **basic_kwargs)

            text_parts = [segment.text.strip() for segment in segments]
            text = self._cleanup_text(" ".join(text_parts))

        except Exception as e:
            logger.error(f"ASR inference error: {e}")
            text = ""

        t_end = time.time()
        audio_duration = len(chunk.audio) / 16000
        rtf = (t_end - t_start) / audio_duration if audio_duration > 0 else 0.0
        logger.debug(f"ASR: '{text[:50]}...' | RTF={rtf:.2f} | final={chunk.is_final}")

        return ASRResult(
            text=text,
            segments=[],
            t_start=chunk.t_start,
            t_end=chunk.t_end,
            is_final=chunk.is_final,
            t_asr_start=t_start,
            t_asr_end=t_end,
        )

    def _enqueue_result(self, result: ASRResult) -> None:
        """Push ASR result into output queue."""
        try:
            if result.is_final:
                while True:
                    try:
                        self.output_queue.put_nowait(result)
                        break
                    except Exception:
                        try:
                            old = self.output_queue.get_nowait()
                            if old.is_final:
                                self.output_queue.put_nowait(old)
                                time.sleep(0.01)
                        except Empty:
                            break
            else:
                self.output_queue.put_nowait(result)
        except Exception:
            logger.warning("ASR output queue full, dropping partial result")

    def _handle_error(self, error: Exception) -> None:
        logger.error(f"ASR error: {error}")
        self._error_count += 1
        if self._error_count >= self._max_errors:
            logger.critical(f"ASR error count exceeded ({self._max_errors})")
            self._running = False

    def stop(self) -> None:
        """Stop worker thread."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()
