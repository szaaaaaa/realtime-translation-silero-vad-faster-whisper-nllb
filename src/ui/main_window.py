"""
主窗口 - 流式翻译管道控制
"""
import time
import logging
from queue import Queue
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QComboBox, QPushButton, QSlider,
                             QTextEdit, QGroupBox, QSpinBox, QProgressBar)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer

from src.core.config_manager import ConfigManager
from src.core.audio_capture import AudioCapture
from src.core.vad import SileroVAD
from src.core.chunker import Chunker
from src.core.asr_worker import ASRWorker
from src.core.text_stabilizer import TextStabilizer
from src.core.mt_worker import MTWorker
from src.core.latency_logger import LatencyLogger
from src.ui.subtitle_window import SubtitleWindow

logger = logging.getLogger(__name__)


class ModelLoaderThread(QThread):
    """模型加载线程"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, vad: SileroVAD, asr_worker: ASRWorker, mt_worker: MTWorker):
        super().__init__()
        self.vad = vad
        self.asr_worker = asr_worker
        self.mt_worker = mt_worker

    def run(self):
        try:
            self.progress.emit("正在加载 VAD 模型...")
            self.vad.load_model()

            self.progress.emit("正在加载 ASR 模型 (faster-whisper)...")
            self.asr_worker.load_model()

            self.progress.emit("正在加载 MT 模型 (NLLB)...")
            self.mt_worker.load_model()

            self.finished.emit(True, "所有模型加载成功")
        except Exception as e:
            self.finished.emit(False, f"模型加载失败: {str(e)}")


class PipelineWorker(QThread):
    """流式处理管道工作线程"""
    text_updated = pyqtSignal(str, str, str)  # partial_src, final_src, final_tgt
    status_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self,
                 audio_capture: AudioCapture,
                 vad: SileroVAD,
                 chunker: Chunker,
                 asr_worker: ASRWorker,
                 text_stabilizer: TextStabilizer,
                 mt_worker: MTWorker,
                 latency_logger: LatencyLogger):
        super().__init__()
        self.audio_capture = audio_capture
        self.vad = vad
        self.chunker = chunker
        self.asr_worker = asr_worker
        self.text_stabilizer = text_stabilizer
        self.mt_worker = mt_worker
        self.latency_logger = latency_logger
        self.is_running = False

    def run(self):
        self.is_running = True

        try:
            self.audio_capture.start()
            self.asr_worker.start()
            self.mt_worker.start()
            self.status_updated.emit("正在监听...")

            while self.is_running:
                # 1. 读取音频帧
                frame = self.audio_capture.read_frame(timeout=0.1)
                if frame is None:
                    self._process_asr_results()
                    self._process_mt_results()
                    continue

                # 2. VAD 处理
                vad_event = self.vad.process_frame(frame.samples, frame.timestamp)

                if vad_event:
                    if vad_event.event_type == "start":
                        self.latency_logger.start_utterance()
                        self.latency_logger.log_capture(frame.timestamp)
                        self.chunker.on_vad_start(vad_event.timestamp)
                        self.status_updated.emit("检测到语音...")

                    elif vad_event.event_type == "frame":
                        chunk = self.chunker.on_speech_frame(
                            vad_event.frame,
                            vad_event.timestamp
                        )
                        if chunk:
                            self.latency_logger.log_chunk_emit()
                            self.asr_worker.input_queue.put(chunk)

                    elif vad_event.event_type == "end":
                        chunk = self.chunker.on_vad_end(vad_event.timestamp)
                        if chunk:
                            self.latency_logger.log_chunk_emit()
                            self.asr_worker.input_queue.put(chunk)
                        self.status_updated.emit("正在监听...")

                # 3. 处理 ASR 结果
                self._process_asr_results()

                # 4. 处理 MT 结果
                self._process_mt_results()

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)
        finally:
            self.audio_capture.stop()
            self.asr_worker.stop()
            self.mt_worker.stop()
            self.status_updated.emit("已停止")

    def _process_asr_results(self):
        """处理 ASR 结果"""
        try:
            while not self.asr_worker.output_queue.empty():
                result = self.asr_worker.output_queue.get_nowait()

                self.latency_logger.log_asr_start(result.t_asr_start)
                self.latency_logger.log_asr_end(result.t_asr_end)

                # 文本稳定化
                stable_output = self.text_stabilizer.process(result)
                self.latency_logger.log_stabilize()

                if result.is_final and stable_output.final_append_src:
                    # 发送到 MT
                    self.latency_logger.log_mt_start()
                    self.mt_worker.input_queue.put(stable_output.final_append_src)
                    # 更新 UI（原文 final）
                    self.text_updated.emit("", stable_output.final_append_src, "")
                else:
                    # 更新 UI（partial）
                    self.text_updated.emit(stable_output.partial_src, "", "")
        except Exception as e:
            logger.error(f"处理 ASR 结果错误: {e}")

    def _process_mt_results(self):
        """处理 MT 结果"""
        try:
            while not self.mt_worker.output_queue.empty():
                result = self.mt_worker.output_queue.get_nowait()

                self.latency_logger.log_mt_end(result.t_mt_end)
                self.latency_logger.log_ui_render()
                self.latency_logger.end_utterance()

                # 更新 UI（译文 final）
                self.text_updated.emit("", "", result.target)
        except Exception as e:
            logger.error(f"处理 MT 结果错误: {e}")

    def stop(self):
        self.is_running = False


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self, config: dict = None):
        super().__init__()
        self.setWindowTitle("低延迟流式翻译")
        self.resize(450, 650)

        self.cm = ConfigManager()
        self._init_components(config or {})
        self._init_ui()

        self.subtitle_window = SubtitleWindow(self.cm.config.get("display", {}))
        self.subtitle_window.show()

        self.worker = None
        self.model_loader = None

        self.load_models()

    def _init_components(self, config: dict):
        """初始化管道组件"""
        # 从配置获取参数
        audio_cfg = self.cm.get("audio")
        vad_cfg = config.get("vad", {})
        chunker_cfg = config.get("chunker", {})
        asr_cfg = config.get("asr", {})
        mt_cfg = config.get("mt", {})

        # 音频采集
        self.audio_capture = AudioCapture(
            mode=audio_cfg.get("input_mode", "loopback"),
            device_index=audio_cfg.get("device_index"),
            sample_rate=16000,
            frame_ms=20
        )

        # VAD
        self.vad = SileroVAD(
            threshold=vad_cfg.get("threshold", 0.5),
            speech_start_frames=vad_cfg.get("speech_start_frames", 6),
            speech_end_frames=vad_cfg.get("speech_end_frames", 10)
        )

        # Chunker
        self.chunker = Chunker(
            chunk_ms=chunker_cfg.get("chunk_ms", 1000),
            overlap_ms=chunker_cfg.get("overlap_ms", 200),
            max_utterance_ms=chunker_cfg.get("max_utterance_ms", 10000),
            tail_pad_ms=chunker_cfg.get("tail_pad_ms", 120)
        )

        # 队列
        self.q_asr_in = Queue(maxsize=32)
        self.q_asr_out = Queue(maxsize=32)
        self.q_mt_in = Queue(maxsize=32)
        self.q_mt_out = Queue(maxsize=32)

        # ASR Worker
        self.asr_worker = ASRWorker(
            input_queue=self.q_asr_in,
            output_queue=self.q_asr_out,
            model_size=asr_cfg.get("model_size", "small"),
            language=asr_cfg.get("language", "en"),
            device=asr_cfg.get("device", "cuda"),
            compute_type=asr_cfg.get("compute_type", "float16")
        )

        # Text Stabilizer
        self.text_stabilizer = TextStabilizer()

        # MT Worker
        self.mt_worker = MTWorker(
            input_queue=self.q_mt_in,
            output_queue=self.q_mt_out,
            model_name=mt_cfg.get("model_name", "facebook/nllb-200-distilled-600M"),
            src_lang=mt_cfg.get("src_lang", "eng_Latn"),
            tgt_lang=mt_cfg.get("tgt_lang", "zho_Hans"),
            device=mt_cfg.get("device", "cuda")
        )

        # Latency Logger
        self.latency_logger = LatencyLogger()

    def _init_ui(self):
        """初始化 UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 状态显示
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

        # 控制按钮
        control_group = QGroupBox("控制")
        control_layout = QHBoxLayout()

        self.btn_start = QPushButton("开始翻译")
        self.btn_start.clicked.connect(self.toggle_translation)
        self.btn_start.setEnabled(False)
        self.btn_start.setStyleSheet(
            "background-color: #38A169; color: white; "
            "font-weight: bold; padding: 10px;"
        )

        control_layout.addWidget(self.btn_start)
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # 音频设置
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

        # 外观设置
        ui_group = QGroupBox("外观")
        ui_layout = QVBoxLayout()

        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(10, 100)
        self.slider_opacity.setValue(
            int(self.cm.get("display").get("opacity", 0.8) * 100)
        )
        self.slider_opacity.valueChanged.connect(self.update_appearance)

        self.spin_font_size = QSpinBox()
        self.spin_font_size.setRange(12, 72)
        self.spin_font_size.setValue(
            self.cm.get("display").get("font_size", 24)
        )
        self.spin_font_size.valueChanged.connect(self.update_appearance)

        ui_layout.addWidget(QLabel("透明度:"))
        ui_layout.addWidget(self.slider_opacity)
        ui_layout.addWidget(QLabel("字体大小:"))
        ui_layout.addWidget(self.spin_font_size)
        ui_group.setLayout(ui_layout)
        layout.addWidget(ui_group)

        # 日志
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(150)
        layout.addWidget(QLabel("日志:"))
        layout.addWidget(self.log_area)

    def load_models(self):
        """加载所有模型"""
        self.lbl_model_status.setText("正在加载模型...")
        self.progress_bar.show()
        self.btn_start.setEnabled(False)

        self.model_loader = ModelLoaderThread(
            self.vad,
            self.asr_worker,
            self.mt_worker
        )
        self.model_loader.progress.connect(self.on_model_progress)
        self.model_loader.finished.connect(self.on_model_loaded)
        self.model_loader.start()

    def on_model_progress(self, message: str):
        """模型加载进度回调"""
        self.lbl_model_status.setText(message)
        self.log(message)

    def on_model_loaded(self, success: bool, message: str):
        """模型加载完成回调"""
        self.progress_bar.hide()
        self.lbl_model_status.setText(message)
        self.log(message)

        if success:
            self.btn_start.setEnabled(True)
            self.lbl_model_status.setStyleSheet("color: green;")
        else:
            self.lbl_model_status.setStyleSheet("color: red;")

    def refresh_devices(self):
        """刷新音频设备列表"""
        self.combo_devices.clear()
        devices = self.audio_capture.list_devices()

        default_index = self.cm.get("audio").get("device_index")
        current_idx = 0

        for i, dev in enumerate(devices):
            name = f"[{dev['index']}] {dev['name']} ({dev['hostapi']})"
            self.combo_devices.addItem(name, dev['index'])
            if default_index is not None and dev['index'] == default_index:
                current_idx = i

        self.combo_devices.setCurrentIndex(current_idx)

    def save_audio_settings(self):
        """保存音频设置"""
        idx = self.combo_devices.currentData()
        if idx is not None:
            self.cm.set("audio", "device_index", idx)
            self.audio_capture.device_index = idx

    def update_appearance(self):
        """更新外观设置"""
        opacity = self.slider_opacity.value() / 100.0
        font_size = self.spin_font_size.value()

        self.cm.set("display", "opacity", opacity)
        self.cm.set("display", "font_size", font_size)

        display_config = self.cm.get("display")
        self.subtitle_window.config = display_config
        self.subtitle_window.update_styles()

    def toggle_translation(self):
        """切换翻译状态"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            self.btn_start.setText("开始翻译")
            self.btn_start.setStyleSheet(
                "background-color: #38A169; color: white; "
                "font-weight: bold; padding: 10px;"
            )
            self.log("翻译已停止")

            # 重置组件
            self.vad.reset()
            self.chunker.reset()
            self.text_stabilizer.reset()
        else:
            self.worker = PipelineWorker(
                self.audio_capture,
                self.vad,
                self.chunker,
                self.asr_worker,
                self.text_stabilizer,
                self.mt_worker,
                self.latency_logger
            )
            self.worker.text_updated.connect(self.subtitle_window.update_subtitle)
            self.worker.status_updated.connect(self.log)
            self.worker.error_occurred.connect(lambda e: self.log(f"错误: {e}"))
            self.worker.start()

            self.btn_start.setText("停止翻译")
            self.btn_start.setStyleSheet(
                "background-color: #E53E3E; color: white; "
                "font-weight: bold; padding: 10px;"
            )
            self.log("翻译已开始")

    def log(self, message: str):
        """添加日志"""
        timestamp = time.strftime('%H:%M:%S')
        self.log_area.append(f"[{timestamp}] {message}")
        sb = self.log_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        self.subtitle_window.close()
        event.accept()
