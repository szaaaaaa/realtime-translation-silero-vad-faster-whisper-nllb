"""
Seamless 实时翻译应用入口
"""
import sys


def main():
    # 检查 CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA 可用: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("警告: CUDA 不可用，将使用 CPU（性能会很慢）")
    except ImportError:
        print("错误: 未安装 PyTorch")
        sys.exit(1)

    from PyQt6.QtWidgets import QApplication
    from src.ui.main_window import MainWindow

    app = QApplication(sys.argv)

    # 设置字体
    font = app.font()
    font.setFamily("Microsoft YaHei")
    font.setPointSize(10)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
