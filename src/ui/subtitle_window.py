"""
Subtitle window for bilingual real-time display.
"""
import time
from collections import deque

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QMainWindow, QLabel, QWidget, QVBoxLayout


class SubtitleWindow(QMainWindow):
    """
    Floating subtitle window.
    - Top-most, translucent, draggable.
    - Shows source (history + partial) and target history.
    """

    update_signal = pyqtSignal(str, str, str)  # partial_src, final_src, final_tgt
    refine_signal = pyqtSignal(str, str)  # source_text, refined_tgt

    def __init__(self, config: dict = None):
        super().__init__()
        self.config = config or {}

        self._last_update = 0.0
        self._min_interval = 0.1  # 10Hz
        self._pending_update = None

        self._history_lines = self.config.get("history_lines", 50)
        self._src_history = deque(maxlen=self._history_lines)
        self._tgt_history = deque(maxlen=self._history_lines)
        self._display_lines = max(3, int(self.config.get("display_lines", 8)))

        self._partial_src = ""
        self._drag_pos = None
        self._click_through = self.config.get("click_through", False)

        self.init_ui()

        self.update_signal.connect(self._do_update)
        self.refine_signal.connect(self._do_refine)

        self._throttle_timer = QTimer()
        self._throttle_timer.timeout.connect(self._flush_pending)
        self._throttle_timer.start(100)

    def init_ui(self):
        """Initialize UI elements."""
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        if self._click_through:
            flags |= Qt.WindowType.WindowTransparentForInput

        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        central_widget = QWidget()
        opacity = int(self.config.get("opacity", 0.8) * 255)
        central_widget.setStyleSheet(
            f"background-color: rgba(0, 0, 0, {opacity}); border-radius: 10px;"
        )
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(5)

        self.src_label = QLabel("")
        self.src_label.setWordWrap(True)
        self.src_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.tgt_label = QLabel("")
        self.tgt_label.setWordWrap(True)
        self.tgt_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        layout.addWidget(self.src_label)
        layout.addWidget(self.tgt_label)

        self.update_styles()

        x = self.config.get("position_x", 100)
        y = self.config.get("position_y", 800)
        w = self.config.get("width", 800)
        h = self.config.get("height", 120)
        self.setGeometry(x, y, w, h)
        self.setMinimumWidth(400)

    def update_styles(self):
        """Update visual styles from config."""
        font_size = self.config.get("font_size", 24)
        src_color = self.config.get("src_color", "#AAAAAA")
        tgt_color = self.config.get("tgt_color", "#FFFFFF")

        src_font = QFont("Microsoft YaHei", int(font_size * 0.75))
        self.src_label.setFont(src_font)
        self.src_label.setStyleSheet(f"color: {src_color};")

        tgt_font = QFont("Microsoft YaHei", font_size)
        tgt_font.setBold(True)
        self.tgt_label.setFont(tgt_font)
        self.tgt_label.setStyleSheet(f"color: {tgt_color};")

    def update_subtitle(self, partial_src: str = "", final_src: str = "", final_tgt: str = ""):
        """Thread-safe update entry for streaming subtitles."""
        self.update_signal.emit(partial_src, final_src, final_tgt)

    def apply_refinement(self, source_text: str, refined_tgt: str):
        """Thread-safe entry: apply refined translation backfill."""
        self.refine_signal.emit(source_text or "", refined_tgt or "")

    def _do_update(self, partial_src: str, final_src: str, final_tgt: str):
        now = time.time()
        if now - self._last_update < self._min_interval:
            self._pending_update = (partial_src, final_src, final_tgt)
            return

        self._apply_update(partial_src, final_src, final_tgt)
        self._last_update = now

    def _flush_pending(self):
        if self._pending_update:
            partial_src, final_src, final_tgt = self._pending_update
            self._pending_update = None
            self._apply_update(partial_src, final_src, final_tgt)
            self._last_update = time.time()

    def _apply_update(self, partial_src: str, final_src: str, final_tgt: str):
        if partial_src:
            self._partial_src = partial_src

        if final_src:
            self._src_history.append(final_src)
            self._partial_src = ""

        if final_tgt:
            self._tgt_history.append(final_tgt)

        self._render_labels()

    def _do_refine(self, source_text: str, refined_tgt: str):
        if not refined_tgt:
            return

        if not self._tgt_history:
            self._tgt_history.append(refined_tgt)
            self._render_labels()
            return

        src_list = list(self._src_history)
        tgt_len = len(self._tgt_history)
        idx = -1
        source_norm = source_text.strip()

        if source_norm:
            for i in range(len(src_list) - 1, -1, -1):
                if src_list[i].strip() == source_norm:
                    idx = i
                    break

        if 0 <= idx < tgt_len:
            self._tgt_history[idx] = refined_tgt
        else:
            self._tgt_history[-1] = refined_tgt

        self._render_labels()

    def _render_labels(self):
        src_lines = list(self._src_history)
        if self._partial_src:
            src_lines.append(f"▌ {self._partial_src}")

        self.src_label.setText("\n".join(src_lines[-self._display_lines:]))
        self.tgt_label.setText("\n".join(list(self._tgt_history)[-self._display_lines:]))

    def update_text(self, original: str, translated: str):
        """Backward-compatible method."""
        self.update_subtitle(partial_src=original, final_tgt=translated)

    def set_click_through(self, enabled: bool):
        """Toggle click-through mode."""
        self._click_through = enabled
        flags = self.windowFlags()

        if enabled:
            flags |= Qt.WindowType.WindowTransparentForInput
        else:
            flags &= ~Qt.WindowType.WindowTransparentForInput

        self.setWindowFlags(flags)
        self.show()

    def clear(self):
        """Clear all subtitles."""
        self._src_history.clear()
        self._tgt_history.clear()
        self._partial_src = ""
        self.src_label.setText("")
        self.tgt_label.setText("")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if self._drag_pos:
            delta = event.globalPosition().toPoint() - self._drag_pos
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self._drag_pos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.set_click_through(not self._click_through)
