"""
主窗口 - PyQt6 实现
"""
import time
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QComboBox, QPushButton, QSlider,
                             QTextEdit, QGroupBox, QSpinBox, QProgressBar)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from src.core.config_manager import ConfigManager
from src.core.audio_capture import AudioCapture
from src.core.seamless_translator import SeamlessTranslator, TARGET_LANGUAGE_CODES
from src.ui.subtitle_window import SubtitleWindow


class ModelLoaderThread(QThread):
    """模型加载线程"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, translator: SeamlessTranslator):
        super().__init__()
        self.translator = translator

    def run(self):
        try:
            self.translator.load_model(progress_callback=self.progress.emit)
            self.finished.emit(True, "模型加载成功")
        except Exception as e:
            self.finished.emit(False, f"模型加载失败: {str(e)}")


class TranslationWorker(QThread):
    """翻译工作线程"""
    text_updated = pyqtSignal(str, str)
    status_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, translator: SeamlessTranslator,
                 audio_capture: AudioCapture,
                 config_manager: ConfigManager):
        super().__init__()
        self.translator = translator
        self.audio_capture = audio_capture
        self.cm = config_manager
        self.is_running = False

        # VAD tuning for lower latency and fewer hallucinations
        self.silence_threshold = 0.015
        self.min_duration = 1.0
        self.max_duration = 6.0
        self.silence_duration = 0.3
        self.min_speech_duration = 0.3
        self.clear_after_silence = 1.5
        self.last_speech_time = 0.0
        self.last_translation_time = 0.0
        self.subtitles_visible = False

    def run(self):
        self.is_running = True

        try:
            trans_config = self.cm.get("translation")
            source_lang = trans_config.get("source_lang", "auto")
            target_lang = trans_config.get("target_lang", "zh-CN")

            audio_config = self.cm.get("audio")
            self.audio_capture.device_index = audio_config.get("device_index")

            self.audio_capture.start()
            self.status_updated.emit("正在监听...")

            audio_buffer = []
            current_duration = 0.0
            silence_counter = 0.0
            speech_duration = 0.0
            sample_rate = self.audio_capture.current_sample_rate

            while self.is_running:
                chunk = self.audio_capture.get_audio_chunk(timeout=0.5)
                if chunk is None:
                    continue

                audio_buffer.append(chunk)
                chunk_duration = len(chunk) / sample_rate
                current_duration += chunk_duration

                rms = np.sqrt(np.mean(chunk ** 2))
                if rms < self.silence_threshold:
                    silence_counter += chunk_duration
                else:
                    silence_counter = 0.0
                    speech_duration += chunk_duration
                    self.last_speech_time = time.time()

                should_translate = False
                if current_duration > self.min_duration and silence_counter > self.silence_duration:
                    should_translate = True
                elif current_duration > self.max_duration:
                    should_translate = True

                if should_translate and audio_buffer:
                    if speech_duration < self.min_speech_duration:
                        # Treat as silence/noise: clear buffer and subtitles
                        audio_buffer = []
                        current_duration = 0.0
                        silence_counter = 0.0
                        speech_duration = 0.0
                        self.status_updated.emit("????...")
                        continue

                    full_audio = np.concatenate(audio_buffer, axis=0)
                    full_audio = full_audio.flatten()

                    self.status_updated.emit("正在翻译...")

                    try:
                        original, translated = self.translator.translate(
                            audio=full_audio,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            original_sr=sample_rate
                        )

                        if translated.strip():
                            self.text_updated.emit(original, translated)
                            self.subtitles_visible = True
                            self.last_translation_time = time.time()

                    except Exception as e:
                        self.status_updated.emit(f"翻译错误: {str(e)}")

                    audio_buffer = []
                    current_duration = 0.0
                    silence_counter = 0.0
                    speech_duration = 0.0
                    self.status_updated.emit("正在监听...")

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)
        finally:
            self.audio_capture.stop()
            self.status_updated.emit("已停止")

    def stop(self):
        self.is_running = False
        self.audio_capture.stop()


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Seamless 实时翻译")
        self.resize(450, 650)

        self.cm = ConfigManager()
        self.audio_capture = AudioCapture(chunk_duration=0.25)
        self.translator = SeamlessTranslator(device="cuda", use_8bit=True)

        self.subtitle_window = SubtitleWindow(self.cm.config.get("display", {}))
        self.worker = None
        self.model_loader = None

        self.init_ui()
        self.subtitle_window.show()
        self.load_model()

    def init_ui(self):
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

        # 语言设置
        lang_group = QGroupBox("语言")
        lang_layout = QHBoxLayout()

        self.combo_src = QComboBox()
        self.combo_src.addItems(["auto", "en", "zh", "ja", "ko", "es", "fr", "de", "ru"])
        self.combo_src.setCurrentText(
            self.cm.get("translation").get("source_lang", "auto")
        )

        self.combo_target = QComboBox()
        self.combo_target.addItems(list(TARGET_LANGUAGE_CODES.keys()))
        self.combo_target.setCurrentText(
            self.cm.get("translation").get("target_lang", "zh-CN")
        )

        self.combo_src.currentTextChanged.connect(self.save_lang_settings)
        self.combo_target.currentTextChanged.connect(self.save_lang_settings)

        lang_layout.addWidget(QLabel("源语言:"))
        lang_layout.addWidget(self.combo_src)
        lang_layout.addWidget(QLabel("目标语言:"))
        lang_layout.addWidget(self.combo_target)
        lang_group.setLayout(lang_layout)
        layout.addWidget(lang_group)

        # 外观设置
        ui_group = QGroupBox("外观")
        ui_layout = QVBoxLayout()

        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(10, 100)
        self.slider_opacity.setValue(
            int(self.cm.get("display").get("opacity", 0.7) * 100)
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

    def load_model(self):
        """加载 SeamlessM4T 模型"""
        self.lbl_model_status.setText("正在加载模型...")
        self.progress_bar.show()
        self.btn_start.setEnabled(False)

        self.model_loader = ModelLoaderThread(self.translator)
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

    def save_lang_settings(self):
        """保存语言设置"""
        self.cm.set("translation", "source_lang", self.combo_src.currentText())
        self.cm.set("translation", "target_lang", self.combo_target.currentText())

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
        else:
            self.worker = TranslationWorker(
                self.translator,
                self.audio_capture,
                self.cm
            )
            self.worker.text_updated.connect(self.subtitle_window.update_text)
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
