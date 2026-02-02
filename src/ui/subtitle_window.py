from PyQt6.QtWidgets import QMainWindow, QLabel, QWidget, QVBoxLayout
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette

class SubtitleWindow(QMainWindow):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.init_ui()
        self.old_pos = None

    def init_ui(self):
        # Window flags for transparent, frameless, always-on-top window
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowStaysOnTopHint | 
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Central widget
        central_widget = QWidget()
        central_widget.setStyleSheet(f"background-color: rgba(0, 0, 0, {int(self.config.get('opacity', 0.7) * 255)}); border-radius: 10px;")
        self.setCentralWidget(central_widget)

        # Layout
        layout = QVBoxLayout(central_widget)
        
        # Labels
        self.original_label = QLabel("")
        self.translated_label = QLabel("Waiting for audio...")
        
        self.update_styles()

        layout.addWidget(self.original_label)
        layout.addWidget(self.translated_label)
        
        # Initial geometry
        x = self.config.get('position_x', 100)
        y = self.config.get('position_y', 800)
        w = self.config.get('width', 800)
        h = self.config.get('height', 150)
        self.setGeometry(x, y, w, h)

    def update_styles(self):
        font_size = self.config.get('font_size', 24)
        color = self.config.get('font_color', '#FFFFFF')
        
        font = QFont("Microsoft YaHei", font_size)
        font.setBold(True)
        
        # Original text smaller
        orig_font = QFont("Microsoft YaHei", int(font_size * 0.7))
        
        self.original_label.setFont(orig_font)
        self.original_label.setStyleSheet(f"color: #AAAAAA;")
        self.original_label.setWordWrap(True)
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.translated_label.setFont(font)
        self.translated_label.setStyleSheet(f"color: {color};")
        self.translated_label.setWordWrap(True)
        self.translated_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def update_text(self, original, translated):
        if original is not None:
            self.original_label.setText(original)
        if translated is not None:
            self.translated_label.setText(translated)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.old_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if self.old_pos:
            delta = event.globalPosition().toPoint() - self.old_pos
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.old_pos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event):
        self.old_pos = None
        # Save position to config could be done here or in main window
