"""
低延迟流式翻译应用入口
Whisper + NLLB 流式管道架构
"""
import sys
import logging
import yaml
from pathlib import Path


def setup_logging():
    """配置日志"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "app.log", encoding='utf-8')
        ]
    )


def load_config() -> dict:
    """加载配置文件"""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


def check_gpu():
    """检查 GPU 状态"""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"CUDA 可用: {device_name}")
            print(f"显存: {memory_gb:.1f} GB")
            return "cuda"
        else:
            print("警告: CUDA 不可用，将使用 CPU（性能会较慢）")
            return "cpu"
    except ImportError:
        print("错误: 未安装 PyTorch")
        sys.exit(1)


def main():
    """主函数"""
    # 配置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("启动低延迟流式翻译应用")

    # 检查 GPU
    device = check_gpu()

    # 加载配置
    config = load_config()

    # 设置设备
    if 'asr' not in config:
        config['asr'] = {}
    if 'mt' not in config:
        config['mt'] = {}
    config['asr']['device'] = device
    config['mt']['device'] = device

    # 启动 Qt 应用
    from PyQt6.QtWidgets import QApplication
    from src.ui.main_window import MainWindow

    app = QApplication(sys.argv)

    # 设置字体
    font = app.font()
    font.setFamily("Microsoft YaHei")
    font.setPointSize(10)
    app.setFont(font)

    # 创建主窗口
    window = MainWindow(config=config)
    window.show()

    logger.info("应用启动完成")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
