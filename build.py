"""
PyInstaller 打包脚本
注意：SeamlessM4T 模型很大，不建议打包进 exe
建议：打包程序本身，模型在首次运行时自动下载到用户目录
"""
import PyInstaller.__main__
import os
import shutil


def build():
    # 清理
    for folder in ['dist', 'build']:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    print("开始打包...")

    PyInstaller.__main__.run([
        'main.py',
        '--name=SeamlessTranslator',
        '--windowed',
        '--onedir',
        '--clean',

        # 收集依赖
        '--collect-all=transformers',
        '--collect-all=torch',
        '--collect-all=torchaudio',
        '--collect-all=sounddevice',
        '--collect-all=sentencepiece',

        # 隐藏导入
        '--hidden-import=PyQt6',
        '--hidden-import=numpy',

        # 路径
        '--paths=.',

        # 排除不需要的模块
        '--exclude-module=matplotlib',
        '--exclude-module=PIL',
        '--exclude-module=tkinter',
    ])

    print("打包完成！")
    print("注意：首次运行时会自动下载 SeamlessM4T 模型（约 10GB）")


if __name__ == "__main__":
    build()
