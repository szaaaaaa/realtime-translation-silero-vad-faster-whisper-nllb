import os
import sys

# ============================================================================
# Fix: OpenMP DLL conflict between torch and ctranslate2
# Both libraries depend on libiomp5md.dll (Intel OpenMP). In PyInstaller-packed
# environments, loading two copies causes WinError 1114 during DLL initialization.
#
# Solution:
# 1. Set KMP_DUPLICATE_LIB_OK=TRUE to allow duplicate OpenMP runtime loading
# 2. Import torch BEFORE any library that depends on ctranslate2 (faster_whisper)
#    This ensures torch's OpenMP is initialized first and ctranslate2 reuses it.
#
# These lines MUST be at the very top, before any other imports that might
# trigger the torch/ctranslate2 import chain.
# ============================================================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Pre-import torch to ensure its DLLs load first
try:
    import torch
except ImportError:
    pass  # torch may not be directly installed, but collected by PyInstaller

from PyQt6.QtWidgets import QApplication
from src.ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    
    # Set app style/font if needed
    font = app.font()
    font.setFamily("Microsoft YaHei")
    font.setPointSize(10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    print("Starting Trae Translation App...")
    main()
