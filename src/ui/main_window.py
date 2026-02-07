"""
Main window and streaming pipeline controller.
"""
import logging
import re
import time
from queue import Empty, Queue

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.core.asr_worker import ASRWorker
from src.core.audio_capture import AudioCapture
from src.core.chunker import Chunker
from src.core.config_manager import ConfigManager
from src.core.latency_logger import LatencyLogger
from src.core.mt_worker import MTWorker
from src.core.text_stabilizer import TextStabilizer
from src.core.vad import SileroVAD
from src.ui.subtitle_window import SubtitleWindow

logger = logging.getLogger(__name__)


class ModelLoaderThread(QThread):
    """Background thread that loads models to keep UI responsive."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(
        self,
        vad: SileroVAD,
        asr_worker: ASRWorker,
        mt_worker: MTWorker,
        refine_worker: MTWorker = None,
    ):
        super().__init__()
        self.vad = vad
        self.asr_worker = asr_worker
        self.mt_worker = mt_worker
        self.refine_worker = refine_worker
        self.refine_ok = refine_worker is None

    def run(self):
        try:
            self.progress.emit("正在加载 VAD 模型...")
            self.vad.load_model()

            self.progress.emit("正在加载 ASR 模型 (faster-whisper)...")
            self.asr_worker.load_model()

            self.progress.emit("正在加载 MT 模型 (实时)...")
            self.mt_worker.load_model()

            if self.refine_worker is not None:
                self.progress.emit("正在加载 MT 模型 (回填修正)...")
                try:
                    self.refine_worker.load_model()
                    self.refine_ok = True
                except Exception as refine_err:
                    self.refine_ok = False
                    self.progress.emit(f"回填模型加载失败，降级为仅实时翻译: {refine_err}")

            if self.refine_ok:
                self.finished.emit(True, "所有模型加载成功")
            else:
                self.finished.emit(True, "核心模型加载成功（回填已禁用）")
        except Exception as e:
            self.finished.emit(False, f"模型加载失败: {e}")


class PipelineWorker(QThread):
    """Streaming pipeline worker thread."""

    text_updated = pyqtSignal(str, str, str)  # partial_src, final_src, final_tgt
    refine_updated = pyqtSignal(str, str)  # source_text, refined_tgt
    status_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    _FILLER_ONLY_RE = re.compile(
        r"^(?:uh+|um+|umm+|ah+|er+|hmm+|mm+|eh+|"
        r"\u5443+|\u55ef+|\u554a+|\u989d+|\u54ce+)(?:[\s,.!?;:\u3002\uff01\uff1f\uff0c\uff1b\uff1a-]*)$",
        re.IGNORECASE,
    )
    _LEADING_FILLER_RE = re.compile(
        r"^(?:(?:uh+|um+|umm+|ah+|er+|hmm+|mm+|eh+|you know|"
        r"\u5443+|\u55ef+|\u554a+|\u989d+|\u54ce+)"
        r"(?:[\s,.!?;:\u3002\uff01\uff1f\uff0c\uff1b\uff1a-]+))+",
        re.IGNORECASE,
    )
    _INLINE_FILLER_RE = re.compile(
        r"(?i)(?:^|[\s,;:])(uh+|um+|umm+|ah+|er+|hmm+|mm+|eh+)(?=[\s,;:.!?]|$)"
    )
    _CONTINUATION_TAIL_RE = re.compile(
        r"(?:\b(?:and|or|but|because|so|if|when|while|though|although|that|which|who|whom|whose|"
        r"where|to|for|of|with|in|on|at|from|by|as)\b|[,;:\u3001\uff0c\uff1b\uff1a]|\.\.\.)\s*$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        audio_capture: AudioCapture,
        vad: SileroVAD,
        chunker: Chunker,
        asr_worker: ASRWorker,
        text_stabilizer: TextStabilizer,
        mt_worker: MTWorker,
        latency_logger: LatencyLogger,
        refine_worker: MTWorker = None,
        pause_merge_ms: int = 220,
        final_merge_window_ms: int = 900,
        final_merge_min_chars: int = 32,
    ):
        super().__init__()
        self.audio_capture = audio_capture
        self.vad = vad
        self.chunker = chunker
        self.asr_worker = asr_worker
        self.text_stabilizer = text_stabilizer
        self.mt_worker = mt_worker
        self.refine_worker = refine_worker
        self.latency_logger = latency_logger
        self.pause_merge_ms = max(0, int(pause_merge_ms))
        self.final_merge_window_ms = max(0, int(final_merge_window_ms))
        self.final_merge_min_chars = max(8, int(final_merge_min_chars))

        self._pending_vad_end_timestamp = None
        self._pending_vad_end_wall = 0.0
        self._pending_final_src = ""
        self._pending_final_wall = 0.0
        self.is_running = False

    def run(self):
        self.is_running = True

        try:
            self.audio_capture.start()
            self.asr_worker.start()
            self.mt_worker.start()
            if self.refine_worker is not None:
                self.refine_worker.start()

            self.status_updated.emit("正在监听...")

            while self.is_running:
                self._flush_pending_vad_end(force=False)
                self._flush_pending_final(force=False)

                frame = self.audio_capture.read_frame(timeout=0.1)
                if frame is None:
                    self._flush_pending_vad_end(force=False)
                    self._flush_pending_final(force=False)
                    self._process_asr_results()
                    self._process_mt_results()
                    self._process_refine_results()
                    continue

                vad_event = self.vad.process_frame(frame.samples, frame.timestamp)
                if vad_event:
                    if vad_event.event_type == "start":
                        start_audio = vad_event.frame
                        if self._pending_vad_end_timestamp is not None:
                            # Merge short pause into the same utterance.
                            self._clear_pending_vad_end()
                            if start_audio is not None:
                                chunk = self.chunker.on_speech_frame(
                                    start_audio,
                                    vad_event.timestamp,
                                )
                                if chunk:
                                    self.latency_logger.log_chunk_emit()
                                    self.asr_worker.input_queue.put(chunk)
                        else:
                            self.latency_logger.start_utterance()
                            self.latency_logger.log_capture(frame.timestamp)
                            pre_audio = self.audio_capture.get_pre_roll_audio()
                            if start_audio is not None and len(pre_audio) > 0:
                                lead_samples = len(start_audio)
                                if lead_samples > 0:
                                    if len(pre_audio) > lead_samples:
                                        pre_audio = pre_audio[:-lead_samples]
                                    else:
                                        pre_audio = pre_audio[:0]
                            self.chunker.on_vad_start(vad_event.timestamp, pre_audio=pre_audio)
                            if start_audio is not None:
                                chunk = self.chunker.on_speech_frame(
                                    start_audio,
                                    vad_event.timestamp,
                                )
                                if chunk:
                                    self.latency_logger.log_chunk_emit()
                                    self.asr_worker.input_queue.put(chunk)

                        self.status_updated.emit("检测到语音...")

                    elif vad_event.event_type == "frame":
                        chunk = self.chunker.on_speech_frame(vad_event.frame, vad_event.timestamp)
                        if chunk:
                            self.latency_logger.log_chunk_emit()
                            self.asr_worker.input_queue.put(chunk)

                    elif vad_event.event_type == "end":
                        if self.pause_merge_ms > 0:
                            self._pending_vad_end_timestamp = vad_event.timestamp
                            self._pending_vad_end_wall = time.time()
                        else:
                            chunk = self.chunker.on_vad_end(vad_event.timestamp)
                            if chunk:
                                self.latency_logger.log_chunk_emit()
                                self.asr_worker.input_queue.put(chunk)
                            self.status_updated.emit("正在监听...")

                self._process_asr_results()
                self._process_mt_results()
                self._process_refine_results()

        except Exception as e:
            import traceback

            self.error_occurred.emit(f"{e}\n\n{traceback.format_exc()}")
        finally:
            self._flush_pending_vad_end(force=True)
            self._flush_pending_final(force=True)
            self._process_asr_results()
            self._process_mt_results()
            self._process_refine_results()

            self.audio_capture.stop()
            self.asr_worker.stop()
            self.mt_worker.stop()
            if self.refine_worker is not None:
                self.refine_worker.stop()

            self.status_updated.emit("已停止")

    def _clear_pending_vad_end(self):
        self._pending_vad_end_timestamp = None
        self._pending_vad_end_wall = 0.0

    def _flush_pending_vad_end(self, force: bool):
        if self._pending_vad_end_timestamp is None:
            return

        elapsed_ms = (time.time() - self._pending_vad_end_wall) * 1000
        if not force and elapsed_ms < self.pause_merge_ms:
            return

        chunk = self.chunker.on_vad_end(self._pending_vad_end_timestamp)
        self._clear_pending_vad_end()
        if chunk:
            self.latency_logger.log_chunk_emit()
            self.asr_worker.input_queue.put(chunk)
        self.status_updated.emit("正在监听...")

    def _process_asr_results(self):
        try:
            while not self.asr_worker.output_queue.empty():
                result = self.asr_worker.output_queue.get_nowait()

                self.latency_logger.log_asr_start(result.t_asr_start)
                self.latency_logger.log_asr_end(result.t_asr_end)

                stable_output = self.text_stabilizer.process(result)
                self.latency_logger.log_stabilize()

                if result.is_final and stable_output.final_append_src:
                    self._ingest_final_src(stable_output.final_append_src)
                else:
                    self.text_updated.emit(stable_output.partial_src, "", "")
        except Exception as e:
            logger.error(f"处理 ASR 结果错误: {e}")

    @staticmethod
    def _normalize_sentence(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "")).strip()

    @staticmethod
    def _has_sentence_boundary(text: str) -> bool:
        return bool(re.search(r"[.!?\u3002\uff01\uff1f][\"')\]]*\s*$", text))

    def _is_filler_only(self, text: str) -> bool:
        text = self._normalize_sentence(text)
        return bool(text) and bool(self._FILLER_ONLY_RE.match(text))

    def _strip_filler_noise(self, text: str) -> str:
        text = self._normalize_sentence(text)
        if not text:
            return ""
        text = self._LEADING_FILLER_RE.sub("", text)
        text = self._INLINE_FILLER_RE.sub(" ", text)
        return self._normalize_sentence(text)

    def _ends_with_continuation_cue(self, text: str) -> bool:
        text = self._normalize_sentence(text)
        return bool(text) and bool(self._CONTINUATION_TAIL_RE.search(text))

    def _should_hold_for_merge(self, text: str) -> bool:
        text = self._strip_filler_noise(text)
        if not text:
            return False
        if self._has_sentence_boundary(text):
            return False
        if self._is_filler_only(text):
            return True
        if len(text) <= self.final_merge_min_chars:
            return True
        if self._ends_with_continuation_cue(text):
            return True
        if len(text.split()) <= 6:
            return True
        return False

    def _hold_final(self, text: str):
        text = self._strip_filler_noise(text)
        if not text:
            return
        if self._pending_final_src:
            self._pending_final_src = self._normalize_sentence(f"{self._pending_final_src} {text}")
        else:
            self._pending_final_src = text
        self._pending_final_wall = time.time()

    def _flush_pending_final(self, force: bool):
        if not self._pending_final_src:
            return

        elapsed_ms = (time.time() - self._pending_final_wall) * 1000
        if not force and elapsed_ms < self.final_merge_window_ms:
            return

        text = self._pending_final_src
        self._pending_final_src = ""
        self._pending_final_wall = 0.0
        text = self._strip_filler_noise(text)

        if self._is_filler_only(text):
            return
        self._dispatch_final_src(text)

    def _ingest_final_src(self, text: str):
        text = self._strip_filler_noise(text)
        if not text:
            return

        if self._pending_final_src:
            text = self._normalize_sentence(f"{self._pending_final_src} {text}")
            self._pending_final_src = ""
            self._pending_final_wall = 0.0
            text = self._strip_filler_noise(text)

        if self._should_hold_for_merge(text):
            self._hold_final(text)
            return

        self._dispatch_final_src(text)

    def _dispatch_final_src(self, final_src: str):
        final_src = self._strip_filler_noise(final_src)
        if not final_src:
            return

        self.latency_logger.log_mt_start()
        self.mt_worker.input_queue.put(final_src)

        if self.refine_worker is not None:
            self._enqueue_refine_text(final_src)

        self.text_updated.emit("", final_src, "")

    def _enqueue_refine_text(self, text: str):
        """Keep only latest pending sentence for slow refinement."""
        if self.refine_worker is None:
            return

        q = self.refine_worker.input_queue
        try:
            while True:
                q.get_nowait()
        except Empty:
            pass

        try:
            q.put_nowait(text)
        except Exception:
            pass

    def _process_mt_results(self):
        try:
            while not self.mt_worker.output_queue.empty():
                result = self.mt_worker.output_queue.get_nowait()

                self.latency_logger.log_mt_end(result.t_mt_end)
                self.latency_logger.log_ui_render()
                self.latency_logger.end_utterance()

                self.text_updated.emit("", "", result.target)
        except Exception as e:
            logger.error(f"处理 MT 结果错误: {e}")

    def _process_refine_results(self):
        if self.refine_worker is None:
            return
        try:
            while not self.refine_worker.output_queue.empty():
                result = self.refine_worker.output_queue.get_nowait()
                self.refine_updated.emit(result.source, result.target)
        except Exception as e:
            logger.error(f"处理回填 MT 结果错误: {e}")

    def stop(self):
        self.is_running = False


class MainWindow(QMainWindow):
    """Main control window."""

    def __init__(self, config: dict = None):
        super().__init__()
        self.setWindowTitle("低延迟流式翻译")
        self.resize(450, 650)

        self.cm = ConfigManager()
        self._config = config or {}

        self._init_components(self._config)
        self._init_ui()

        self.subtitle_window = SubtitleWindow(self.cm.config.get("display", {}))
        self.subtitle_window.show()

        self.worker = None
        self.model_loader = None
        self._is_translating = False

        self.load_models()

    def _init_components(self, config: dict):
        audio_cfg = self.cm.get("audio")
        vad_cfg = config.get("vad", {})
        chunker_cfg = config.get("chunker", {})
        asr_cfg = config.get("asr", {})
        mt_cfg = config.get("mt", {})
        refine_cfg = config.get("refine", {})
        stabilizer_cfg = config.get("stabilizer", {})

        self.audio_capture = AudioCapture(
            mode=audio_cfg.get("input_mode", "loopback"),
            device_index=audio_cfg.get("device_index"),
            sample_rate=16000,
            frame_ms=20,
            pre_roll_ms=config.get("audio", {}).get("pre_roll_ms", 600),
        )

        self.vad = SileroVAD(
            threshold=vad_cfg.get("threshold", 0.5),
            speech_start_frames=vad_cfg.get("speech_start_frames", 4),
            speech_end_frames=vad_cfg.get("speech_end_frames", 14),
        )

        self.chunker = Chunker(
            chunk_ms=chunker_cfg.get("chunk_ms", 800),
            overlap_ms=chunker_cfg.get("overlap_ms", 180),
            max_utterance_ms=chunker_cfg.get("max_utterance_ms", 10000),
            tail_pad_ms=chunker_cfg.get("tail_pad_ms", 200),
        )

        self.q_asr_in = Queue(maxsize=32)
        self.q_asr_out = Queue(maxsize=32)
        self.q_mt_in = Queue(maxsize=32)
        self.q_mt_out = Queue(maxsize=32)

        self.asr_worker = ASRWorker(
            input_queue=self.q_asr_in,
            output_queue=self.q_asr_out,
            model_size=asr_cfg.get("model_size", "small"),
            language=asr_cfg.get("language", "en"),
            device=asr_cfg.get("device", "cuda"),
            compute_type=asr_cfg.get("compute_type", "float16"),
            beam_size=asr_cfg.get("beam_size", 2),
            beam_size_final=asr_cfg.get("beam_size_final", 5),
            temperature=asr_cfg.get("temperature", 0.0),
            word_timestamps=asr_cfg.get("word_timestamps", False),
            condition_on_previous_text=asr_cfg.get("condition_on_previous_text", False),
            compression_ratio_threshold=asr_cfg.get("compression_ratio_threshold", 2.2),
            log_prob_threshold=asr_cfg.get("log_prob_threshold", -1.0),
            no_speech_threshold=asr_cfg.get("no_speech_threshold", 0.35),
            suppress_blank=asr_cfg.get("suppress_blank", True),
            suppress_tokens=asr_cfg.get("suppress_tokens", "-1"),
            initial_prompt=asr_cfg.get("initial_prompt", ""),
        )

        self.text_stabilizer = TextStabilizer(
            lock_min_chars=stabilizer_cfg.get("lock_min_chars", 12),
            buffer_keep_chars=stabilizer_cfg.get("buffer_keep_chars", 80),
            lcp_min_chars=stabilizer_cfg.get("lcp_min_chars", 8),
        )

        self.mt_worker = MTWorker(
            input_queue=self.q_mt_in,
            output_queue=self.q_mt_out,
            model_name=mt_cfg.get("model_name", "facebook/nllb-200-distilled-600M"),
            src_lang=mt_cfg.get("src_lang", "eng_Latn"),
            tgt_lang=mt_cfg.get("tgt_lang", "zho_Hans"),
            device=mt_cfg.get("device", "cuda"),
            cache_size=mt_cfg.get("cache_size", 4096),
            num_beams=mt_cfg.get("num_beams", 2),
            flush_each_input=mt_cfg.get("flush_each_input", False),
            batch_max_wait_ms=mt_cfg.get("batch_max_wait_ms", 120),
            continuation_wait_ms=mt_cfg.get("continuation_wait_ms", 280),
            continuation_max_chars=mt_cfg.get("continuation_max_chars", 140),
            batch_max_chars=mt_cfg.get("batch_max_chars", 220),
            max_chars=mt_cfg.get("max_chars", 360),
        )

        self.refine_worker = None
        if refine_cfg.get("enabled", False):
            self.q_refine_in = Queue(maxsize=max(1, int(refine_cfg.get("queue_size", 2))))
            self.q_refine_out = Queue(maxsize=max(1, int(refine_cfg.get("queue_size", 2))))
            self.refine_worker = MTWorker(
                input_queue=self.q_refine_in,
                output_queue=self.q_refine_out,
                model_name=refine_cfg.get("model_name", "facebook/nllb-200-1.3B"),
                src_lang=mt_cfg.get("src_lang", "eng_Latn"),
                tgt_lang=mt_cfg.get("tgt_lang", "zho_Hans"),
                device=refine_cfg.get("device", "auto"),
                cache_size=refine_cfg.get("cache_size", 512),
                num_beams=refine_cfg.get("num_beams", 3),
                flush_each_input=refine_cfg.get("flush_each_input", True),
                batch_max_wait_ms=refine_cfg.get("batch_max_wait_ms", 20),
                continuation_wait_ms=refine_cfg.get("continuation_wait_ms", 20),
                continuation_max_chars=refine_cfg.get("continuation_max_chars", 1000),
                batch_max_chars=refine_cfg.get("batch_max_chars", 1000),
                max_chars=refine_cfg.get("max_chars", 420),
            )

        self.latency_logger = LatencyLogger()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        status_group = QGroupBox("模型状态")
        status_layout = QVBoxLayout()

        self.lbl_model_status = QLabel("模型未加载")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()

        status_layout.addWidget(self.lbl_model_status)
        status_layout.addWidget(self.progress_bar)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        control_group = QGroupBox("控制")
        control_layout = QHBoxLayout()

        self.btn_start = QPushButton("开始翻译")
        self.btn_start.clicked.connect(self.toggle_translation)
        self.btn_start.setEnabled(False)
        self.btn_start.setStyleSheet(
            "background-color: #38A169; color: white; font-weight: bold; padding: 10px;"
        )

        control_layout.addWidget(self.btn_start)
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        audio_group = QGroupBox("音频源")
        audio_layout = QVBoxLayout()

        self.combo_devices = QComboBox()
        self.refresh_devices()
        self.combo_devices.currentIndexChanged.connect(self.save_audio_settings)

        refresh_btn = QPushButton("刷新设备")
        refresh_btn.clicked.connect(self.refresh_devices)

        audio_layout.addWidget(QLabel("选择输入设备:"))
        audio_layout.addWidget(self.combo_devices)
        audio_layout.addWidget(refresh_btn)
        audio_group.setLayout(audio_layout)
        layout.addWidget(audio_group)

        ui_group = QGroupBox("外观")
        ui_layout = QVBoxLayout()

        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(10, 100)
        self.slider_opacity.setValue(int(self.cm.get("display").get("opacity", 0.8) * 100))
        self.slider_opacity.valueChanged.connect(self.update_appearance)

        self.spin_font_size = QSpinBox()
        self.spin_font_size.setRange(12, 72)
        self.spin_font_size.setValue(self.cm.get("display").get("font_size", 24))
        self.spin_font_size.valueChanged.connect(self.update_appearance)

        ui_layout.addWidget(QLabel("透明度:"))
        ui_layout.addWidget(self.slider_opacity)
        ui_layout.addWidget(QLabel("字体大小:"))
        ui_layout.addWidget(self.spin_font_size)
        ui_group.setLayout(ui_layout)
        layout.addWidget(ui_group)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(150)
        layout.addWidget(QLabel("日志:"))
        layout.addWidget(self.log_area)

    def load_models(self):
        self.lbl_model_status.setText("正在加载模型...")
        self.progress_bar.show()
        self.btn_start.setEnabled(False)

        self.model_loader = ModelLoaderThread(
            self.vad,
            self.asr_worker,
            self.mt_worker,
            self.refine_worker,
        )
        self.model_loader.progress.connect(self.on_model_progress)
        self.model_loader.finished.connect(self.on_model_loaded)
        self.model_loader.start()

    def on_model_progress(self, message: str):
        self.lbl_model_status.setText(message)
        self.log(message)

    def on_model_loaded(self, success: bool, message: str):
        if self.model_loader and not self.model_loader.refine_ok:
            self.refine_worker = None

        self.progress_bar.hide()
        self.lbl_model_status.setText(message)
        self.log(message)

        if success:
            self.btn_start.setEnabled(True)
            self.lbl_model_status.setStyleSheet("color: green;")
        else:
            self.lbl_model_status.setStyleSheet("color: red;")

    def refresh_devices(self):
        self.combo_devices.blockSignals(True)
        self.combo_devices.clear()

        options = self.audio_capture.list_preferred_input_options()
        audio_cfg = self.cm.get("audio")
        default_index = audio_cfg.get("device_index")
        default_mode = audio_cfg.get("input_mode", "loopback")

        selected_idx = -1
        mode_matched_idx = -1
        first_available_idx = -1

        for i, option in enumerate(options):
            if option["available"]:
                label = (
                    f"{option['label']} - "
                    f"[{option['index']}] {option['name']} ({option['hostapi']})"
                )
                data = {
                    "index": option["index"],
                    "mode": option["mode"],
                    "profile": option["profile"],
                }
                if first_available_idx < 0:
                    first_available_idx = i
                if default_index is not None and option["index"] == default_index:
                    selected_idx = i
                if default_index is None and option["mode"] == default_mode and mode_matched_idx < 0:
                    mode_matched_idx = i
            else:
                label = f"{option['label']}（未检测到）"
                data = None

            self.combo_devices.addItem(label, data)

        if selected_idx < 0:
            selected_idx = mode_matched_idx if mode_matched_idx >= 0 else first_available_idx
        if selected_idx is None or selected_idx < 0:
            selected_idx = 0

        self.combo_devices.setCurrentIndex(selected_idx)
        self.combo_devices.blockSignals(False)
        self.save_audio_settings()

    def save_audio_settings(self):
        selected = self.combo_devices.currentData()
        if not selected:
            return

        idx = selected.get("index")
        mode = selected.get("mode", "mic")
        if idx is None:
            return

        self.cm.set("audio", "device_index", idx)
        self.cm.set("audio", "input_mode", mode)
        self.audio_capture.device_index = idx
        self.audio_capture.mode = mode

    def update_appearance(self):
        opacity = self.slider_opacity.value() / 100.0
        font_size = self.spin_font_size.value()

        self.cm.set("display", "opacity", opacity)
        self.cm.set("display", "font_size", font_size)

        display_config = self.cm.get("display")
        self.subtitle_window.config = display_config
        self.subtitle_window.update_styles()

    def toggle_translation(self):
        if self._is_translating:
            if self.worker:
                self.worker.stop()
                self.worker.wait()
            self._on_translation_stopped()
            self.log("翻译已停止")
        else:
            self.worker = PipelineWorker(
                self.audio_capture,
                self.vad,
                self.chunker,
                self.asr_worker,
                self.text_stabilizer,
                self.mt_worker,
                self.latency_logger,
                refine_worker=self.refine_worker,
                pause_merge_ms=self._config.get("streaming", {}).get("pause_merge_ms", 220),
                final_merge_window_ms=self._config.get("streaming", {}).get("final_merge_window_ms", 900),
                final_merge_min_chars=self._config.get("streaming", {}).get("final_merge_min_chars", 32),
            )
            self.worker.text_updated.connect(self.subtitle_window.update_subtitle)
            self.worker.refine_updated.connect(self.subtitle_window.apply_refinement)
            self.worker.status_updated.connect(self.log)
            self.worker.error_occurred.connect(lambda e: self.log(f"错误: {e}"))
            self.worker.finished.connect(self._on_worker_finished)
            self.worker.start()

            self._is_translating = True
            self.btn_start.setText("停止翻译")
            self.btn_start.setStyleSheet(
                "background-color: #E53E3E; color: white; font-weight: bold; padding: 10px;"
            )
            self.log("翻译已开始")

    def _on_worker_finished(self):
        if self._is_translating:
            self._on_translation_stopped()

    def _on_translation_stopped(self):
        self._is_translating = False
        self.btn_start.setText("开始翻译")
        self.btn_start.setStyleSheet(
            "background-color: #38A169; color: white; font-weight: bold; padding: 10px;"
        )

        self.vad.reset()
        self.chunker.reset()
        self.text_stabilizer.reset()

    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_area.append(f"[{timestamp}] {message}")
        sb = self.log_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    def closeEvent(self, event):
        if self._is_translating and self.worker:
            self.worker.stop()
            self.worker.wait()
        self.subtitle_window.close()
        event.accept()
