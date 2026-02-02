# Seamless 实时翻译

基于 Meta SeamlessM4T v2 的端到端实时语音翻译应用。支持 100+ 种语言的语音识别和翻译，无需依赖第三方翻译 API。

## 功能特点

- **端到端翻译**：使用 SeamlessM4T v2 模型，直接从语音翻译到目标语言文本
- **实时字幕**：悬浮字幕窗口，支持自定义透明度和字体大小
- **多语言支持**：支持 100+ 种语言输入，主流语言输出
- **本地运行**：所有处理在本地完成，无需网络连接（首次需下载模型）
- **8-bit 量化**：支持 8GB 显存显卡运行

## 系统要求

### 硬件要求

| 配置 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | NVIDIA 8GB 显存 | NVIDIA 12GB+ 显存 |
| 内存 | 16GB | 32GB |
| 存储 | 15GB 可用空间 | 20GB+ 可用空间 |

**支持的显卡**：
- RTX 50 系列 (5070/5080/5090) - 需要 CUDA 12.8+
- RTX 40 系列 (4060/4070/4080/4090)
- RTX 30 系列 (3060/3070/3080/3090)

### 软件要求

- Windows 10/11 或 Linux
- Python 3.10+
- NVIDIA 驱动程序（最新版本）
- CUDA Toolkit（根据显卡型号选择版本）

## 安装指南

### 1. 安装 NVIDIA 驱动和 CUDA

确保已安装最新的 NVIDIA 驱动程序。根据显卡型号选择 CUDA 版本：

| 显卡系列 | CUDA 版本 |
|---------|----------|
| RTX 50 系列 | CUDA 12.8+ |
| RTX 40/30 系列 | CUDA 12.1+ |

### 2. 创建虚拟环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux:
source venv/bin/activate
```

### 3. 安装 PyTorch

根据显卡型号选择安装命令：

**RTX 50 系列 (Blackwell)**：
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**RTX 40/30 系列**：
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. 安装其他依赖

```bash
pip install transformers accelerate sentencepiece sounddevice PyQt6 numpy
```

### 5. 安装 8-bit 量化支持（8GB 显存必需）

```bash
# Windows
pip install bitsandbytes>=0.43.0

# 如果上面失败，使用预编译版本
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.43.0-py3-none-win_amd64.whl
```

### 6. 验证安装

```python
import torch
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"显卡: {torch.cuda.get_device_name(0)}")
```

## 使用方法

### 启动应用

```bash
python main.py
```

### 首次运行

首次运行时，程序会自动下载 SeamlessM4T v2 模型（约 10GB），请确保网络连接稳定。默认缓存目录在 C 盘用户目录（`C:/Users/<你的用户名>/.cache/huggingface/`），可通过环境变量将缓存路径改到 D 盘（例如 `D:/hf_cache`）。

### 界面说明

1. **模型状态**：显示模型加载进度和状态
2. **控制**：开始/停止翻译按钮
3. **音频源**：选择音频输入设备（麦克风或系统回环）
4. **语言**：
   - 源语言：选择输入音频的语言（auto 为自动检测）
   - 目标语言：选择翻译输出的语言
5. **外观**：调整字幕窗口的透明度和字体大小
6. **日志**：显示运行状态和错误信息

### 字幕窗口

- 字幕窗口会悬浮在所有窗口上方
- 可以拖拽移动位置
- 上方显示原文，下方显示翻译

## 配置文件

配置保存在 `config.json`，包含以下设置：

```json
{
    "audio": {
        "device_index": null,
        "sample_rate": 16000
    },
    "translation": {
        "source_lang": "auto",
        "target_lang": "zh-CN"
    },
    "display": {
        "opacity": 0.7,
        "font_size": 24,
        "show_original": true
    },
    "model": {
        "name": "facebook/seamless-m4t-v2-large",
        "device": "cuda",
        "dtype": "float16"
    }
}
```

## 支持的语言

### 源语言（语音输入）

| 代码 | 语言 |
|-----|------|
| auto | 自动检测 |
| en | 英语 |
| zh | 中文 |
| ja | 日语 |
| ko | 韩语 |
| es | 西班牙语 |
| fr | 法语 |
| de | 德语 |
| ru | 俄语 |

### 目标语言（翻译输出）

| 代码 | 语言 |
|-----|------|
| zh-CN | 简体中文 |
| zh-TW | 繁体中文 |
| en | 英语 |
| ja | 日语 |
| ko | 韩语 |
| es | 西班牙语 |
| fr | 法语 |
| de | 德语 |
| ru | 俄语 |

## 打包发布

使用 PyInstaller 打包：

```bash
python build.py
```

打包后的程序位于 `dist/SeamlessTranslator/` 目录。

**注意**：模型不会打包进程序，首次运行时仍需下载。

## 常见问题

### Q: 显存不足怎么办？

A: 程序默认使用 8-bit 量化，需要约 5-6GB 显存。如果仍然不足，可以尝试：
- 关闭其他占用显存的程序
- 降低系统分辨率

### Q: 模型下载失败？

A: 可以手动下载模型：
```bash
# 使用 huggingface-cli
pip install huggingface_hub
huggingface-cli download facebook/seamless-m4t-v2-large
```

### Q: 翻译延迟较高？

A: 这是端到端模型的特点，翻译需要等待一段完整的语音后才能处理。默认设置：
- 最小音频时长：2 秒
- 最大音频时长：10 秒
- 静音触发时长：0.5 秒

### Q: 没有音频输入？

A: 检查以下几点：
1. 确保选择了正确的音频设备
2. Windows 用户需要选择 WASAPI 设备以捕获系统音频
3. 检查麦克风权限设置

## 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                      主窗口 (MainWindow)                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  音频设置   │  │  语言设置   │  │  外观设置   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                  翻译工作线程 (TranslationWorker)         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ AudioCapture│─▶│     VAD     │─▶│ Translator  │     │
│  │  音频采集   │  │  语音检测   │  │ SeamlessM4T │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   字幕窗口 (SubtitleWindow)               │
│  ┌─────────────────────────────────────────────────┐   │
│  │                    原文                          │   │
│  │                   译文                           │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## 许可证

本项目仅供学习和研究使用。SeamlessM4T 模型由 Meta 发布，请遵守其使用条款。

## 参考资源

- [SeamlessM4T v2 - HuggingFace](https://huggingface.co/facebook/seamless-m4t-v2-large)
- [Seamless Communication - GitHub](https://github.com/facebookresearch/seamless_communication)
- [HuggingFace Transformers 文档](https://huggingface.co/docs/transformers/main/en/model_doc/seamless_m4t_v2)
