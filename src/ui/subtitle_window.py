"""
字幕浮窗模块
显示双语字幕，支持透明、置顶、可穿透、拖拽
"""
import time
from collections import deque
from PyQt6.QtWidgets import QMainWindow, QLabel, QWidget, QVBoxLayout
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont


class SubtitleWindow(QMainWindow):
    """
    字幕浮窗
    - 置顶、透明、可穿透、可拖拽
    - 双语显示（原文 + 译文）
    - 刷新限流（10Hz）
    """

    # 更新信号
    update_signal = pyqtSignal(str, str, str)  # partial_src, final_src, final_tgt

    def __init__(self, config: dict = None):
        super().__init__()
        self.config = config or {}

        # 刷新限流
        self._last_update = 0
        self._min_interval = 0.1  # 10Hz
        self._pending_update = None

        # 历史字幕
        self._history_lines = self.config.get('history_lines', 50)
        self._src_history = deque(maxlen=self._history_lines)
        self._tgt_history = deque(maxlen=self._history_lines)

        # 当前 partial 文本
        self._partial_src = ""

        # 拖拽状态
        self._drag_pos = None
        self._click_through = self.config.get('click_through', False)

        self.init_ui()

        # 连接信号
        self.update_signal.connect(self._do_update)

        # 限流定时器
        self._throttle_timer = QTimer()
        self._throttle_timer.timeout.connect(self._flush_pending)
        self._throttle_timer.start(100)  # 100ms

    def init_ui(self):
        """初始化 UI"""
        # 窗口属性
        flags = (
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )

        # 可穿透设置
        if self._click_through:
            flags |= Qt.WindowType.WindowTransparentForInput

        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # 中心 widget
        central_widget = QWidget()
        opacity = int(self.config.get('opacity', 0.8) * 255)
        central_widget.setStyleSheet(
            f"background-color: rgba(0, 0, 0, {opacity}); "
            f"border-radius: 10px;"
        )
        self.setCentralWidget(central_widget)

        # 布局
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(5)

        # 原文标签（partial + history）
        self.src_label = QLabel("")
        self.src_label.setWordWrap(True)
        self.src_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # 译文标签
        self.tgt_label = QLabel("")
        self.tgt_label.setWordWrap(True)
        self.tgt_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        layout.addWidget(self.src_label)
        layout.addWidget(self.tgt_label)

        self.update_styles()

        # 初始位置和大小
        x = self.config.get('position_x', 100)
        y = self.config.get('position_y', 800)
        w = self.config.get('width', 800)
        h = self.config.get('height', 120)
        self.setGeometry(x, y, w, h)
        self.setMinimumWidth(400)

    def update_styles(self):
        """更新样式"""
        font_size = self.config.get('font_size', 24)
        src_color = self.config.get('src_color', '#AAAAAA')
        tgt_color = self.config.get('tgt_color', '#FFFFFF')

        # 原文字体（较小）
        src_font = QFont("Microsoft YaHei", int(font_size * 0.75))
        self.src_label.setFont(src_font)
        self.src_label.setStyleSheet(f"color: {src_color};")

        # 译文字体（较大，加粗）
        tgt_font = QFont("Microsoft YaHei", font_size)
        tgt_font.setBold(True)
        self.tgt_label.setFont(tgt_font)
        self.tgt_label.setStyleSheet(f"color: {tgt_color};")

    def update_subtitle(self, partial_src: str = "",
                        final_src: str = "", final_tgt: str = ""):
        """
        更新字幕（线程安全，通过信号）

        Args:
            partial_src: partial 原文（实时更新）
            final_src: final 原文增量
            final_tgt: final 译文增量
        """
        self.update_signal.emit(partial_src, final_src, final_tgt)

    def _do_update(self, partial_src: str, final_src: str, final_tgt: str):
        """实际更新（在主线程执行）"""
        now = time.time()

        # 限流检查
        if now - self._last_update < self._min_interval:
            # 保存待处理更新
            self._pending_update = (partial_src, final_src, final_tgt)
            return

        self._apply_update(partial_src, final_src, final_tgt)
        self._last_update = now

    def _flush_pending(self):
        """刷新待处理更新"""
        if self._pending_update:
            partial_src, final_src, final_tgt = self._pending_update
            self._pending_update = None
            self._apply_update(partial_src, final_src, final_tgt)
            self._last_update = time.time()

    def _apply_update(self, partial_src: str, final_src: str, final_tgt: str):
        """应用更新到 UI"""
        # 更新 partial
        if partial_src:
            self._partial_src = partial_src

        # 添加 final 到历史
        if final_src:
            self._src_history.append(final_src)
            self._partial_src = ""  # 清空 partial
        if final_tgt:
            self._tgt_history.append(final_tgt)

        # 构建显示文本
        # 原文：历史 + partial
        src_lines = list(self._src_history)
        if self._partial_src:
            src_lines.append(f"▸ {self._partial_src}")  # partial 用特殊符号标记
        src_text = "\n".join(src_lines[-3:])  # 只显示最近 3 行

        # 译文：历史
        tgt_text = "\n".join(list(self._tgt_history)[-3:])  # 只显示最近 3 行

        self.src_label.setText(src_text)
        self.tgt_label.setText(tgt_text)

    def update_text(self, original: str, translated: str):
        """兼容旧接口"""
        self.update_subtitle(partial_src=original, final_tgt=translated)

    def set_click_through(self, enabled: bool):
        """设置是否可穿透"""
        self._click_through = enabled
        flags = self.windowFlags()

        if enabled:
            flags |= Qt.WindowType.WindowTransparentForInput
        else:
            flags &= ~Qt.WindowType.WindowTransparentForInput

        self.setWindowFlags(flags)
        self.show()  # 重新显示以应用新 flags

    def clear(self):
        """清空字幕"""
        self._src_history.clear()
        self._tgt_history.clear()
        self._partial_src = ""
        self.src_label.setText("")
        self.tgt_label.setText("")

    # 鼠标事件（拖拽）
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
        """双击切换穿透模式"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.set_click_through(not self._click_through)
