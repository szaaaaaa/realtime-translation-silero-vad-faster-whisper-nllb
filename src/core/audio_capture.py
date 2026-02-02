"""
音频采集模块 - 使用 sounddevice 采集系统音频
"""
import sounddevice as sd
import numpy as np
from typing import Generator, Optional, List, Dict
import queue


class AudioCapture:
    """音频采集类，支持系统回环和麦克风输入"""

    def __init__(self, sample_rate: int = 16000, channels: int = 1,
                 chunk_duration: float = 0.5):
        """
        初始化音频采集器

        Args:
            sample_rate: 采样率，SeamlessM4T 要求 16000Hz
            channels: 声道数，单声道
            chunk_duration: 每个音频块的时长（秒）
        """
        self.sample_rate = sample_rate
        # Actual capture sample rate (may differ from model sample rate)
        self.current_sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)

        self.device_index: Optional[int] = None
        self.stream: Optional[sd.InputStream] = None
        self.audio_queue: queue.Queue = queue.Queue()
        self.is_running = False

    def list_devices(self) -> List[Dict]:
        """列出所有可用的音频输入设备"""
        devices = []
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:
                devices.append({
                    'index': i,
                    'name': dev['name'],
                    'channels': dev['max_input_channels'],
                    'sample_rate': dev['default_samplerate'],
                    'hostapi': sd.query_hostapis(dev['hostapi'])['name']
                })
        return devices

    def _audio_callback(self, indata, frames, time, status):
        """音频流回调函数"""
        if status:
            print(f"Audio status: {status}")
        # 转换为 float32 并放入队列
        audio_data = indata.copy().astype(np.float32)
        if self.channels == 1 and audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        self.audio_queue.put(audio_data)

    def start(self):
        """开始音频采集"""
        if self.is_running:
            return

        self.is_running = True
        self.audio_queue = queue.Queue()

        # 获取设备的原生采样率
        if self.device_index is not None:
            device_info = sd.query_devices(self.device_index)
            native_sr = int(device_info['default_samplerate'])
            self.current_sample_rate = native_sr
        else:
            self.current_sample_rate = self.sample_rate

        self.stream = sd.InputStream(
            device=self.device_index,
            samplerate=self.current_sample_rate,
            channels=self.channels,
            dtype=np.float32,
            blocksize=int(self.current_sample_rate * self.chunk_duration),
            callback=self._audio_callback
        )
        self.stream.start()

    def stop(self):
        """停止音频采集"""
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        获取一个音频块

        Args:
            timeout: 超时时间（秒）

        Returns:
            音频数据 numpy 数组，如果超时返回 None
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_audio_generator(self) -> Generator[np.ndarray, None, None]:
        """返回音频数据生成器"""
        while self.is_running:
            chunk = self.get_audio_chunk()
            if chunk is not None:
                yield chunk
