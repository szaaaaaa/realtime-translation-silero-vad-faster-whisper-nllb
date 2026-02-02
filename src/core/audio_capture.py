"""
音频采集模块 - WASAPI/麦克风采集
支持系统回环和麦克风输入，输出 20ms 帧
"""
import sounddevice as sd
import numpy as np
from typing import Optional, List, Dict
import queue
import time
import logging

from src.events import AudioFrame

logger = logging.getLogger(__name__)


class AudioCapture:
    """
    音频采集器
    支持 loopback（系统内录）和 mic（麦克风）模式
    """

    def __init__(self,
                 mode: str = "loopback",
                 device_index: Optional[int] = None,
                 sample_rate: int = 16000,
                 frame_ms: int = 20):
        """
        初始化音频采集器

        Args:
            mode: 采集模式 ("loopback" 或 "mic")
            device_index: 设备索引，None 为自动选择
            sample_rate: 采样率，固定 16000Hz
            frame_ms: 帧时长（毫秒），固定 20ms
        """
        self.mode = mode
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = sample_rate * frame_ms // 1000  # 320

        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._restart_count = 0
        self._max_restarts = 5
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1000)

    def list_devices(self) -> List[Dict]:
        """列出所有可用的音频输入设备"""
        devices = []
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:
                hostapi = sd.query_hostapis(dev['hostapi'])
                devices.append({
                    'index': i,
                    'name': dev['name'],
                    'channels': dev['max_input_channels'],
                    'sample_rate': dev['default_samplerate'],
                    'hostapi': hostapi['name'],
                    'is_wasapi': 'WASAPI' in hostapi['name'],
                    'is_loopback': 'loopback' in dev['name'].lower() or 'stereo mix' in dev['name'].lower()
                })
        return devices

    def _find_device(self) -> int:
        """根据模式自动选择设备"""
        if self.device_index is not None:
            return self.device_index

        devices = self.list_devices()

        if self.mode == "loopback":
            # 优先选择 WASAPI loopback 设备
            for dev in devices:
                if dev['is_wasapi'] and dev['is_loopback']:
                    logger.info(f"自动选择 loopback 设备: [{dev['index']}] {dev['name']}")
                    return dev['index']
            # 次选任何 loopback 设备
            for dev in devices:
                if dev['is_loopback']:
                    logger.info(f"自动选择 loopback 设备: [{dev['index']}] {dev['name']}")
                    return dev['index']
            logger.warning("未找到 loopback 设备，使用默认输入")

        # mic 模式或未找到 loopback
        return None  # 使用系统默认

    def _audio_callback(self, indata, frames, time_info, status):
        """音频流回调函数"""
        if status:
            logger.warning(f"音频状态: {status}")
            if status.input_overflow:
                logger.warning("输入缓冲区溢出")

        # 转换为 float32 单声道
        audio_data = indata.copy().astype(np.float32)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        # 创建 AudioFrame
        frame = AudioFrame(
            samples=audio_data.flatten(),
            timestamp=time.time()
        )

        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            # 队列满时丢弃最旧的帧
            try:
                self._frame_queue.get_nowait()
                self._frame_queue.put_nowait(frame)
            except queue.Empty:
                pass

    def start(self) -> None:
        """启动音频采集"""
        if self._running:
            return

        device_idx = self._find_device()

        # 获取设备原生采样率
        if device_idx is not None:
            try:
                device_info = sd.query_devices(device_idx)
                native_sr = int(device_info['default_samplerate'])
            except Exception:
                native_sr = self.sample_rate
        else:
            native_sr = self.sample_rate

        # 计算 blocksize（基于原生采样率）
        blocksize = int(native_sr * self.frame_ms / 1000)

        try:
            self._stream = sd.InputStream(
                device=device_idx,
                samplerate=native_sr,
                channels=1,
                dtype=np.float32,
                blocksize=blocksize,
                callback=self._audio_callback
            )
            self._stream.start()
            self._running = True
            self._restart_count = 0
            logger.info(f"音频采集已启动: device={device_idx}, sr={native_sr}, blocksize={blocksize}")
        except Exception as e:
            logger.error(f"启动音频采集失败: {e}")
            self._on_error(e)

    def read_frame(self, timeout: float = 0.1) -> Optional[AudioFrame]:
        """
        读取一帧音频（阻塞）

        Args:
            timeout: 超时时间（秒）

        Returns:
            AudioFrame 或 None（超时）
        """
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        """停止音频采集"""
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.error(f"停止音频流错误: {e}")
            self._stream = None
        logger.info("音频采集已停止")

    def _on_error(self, error: Exception) -> None:
        """错误处理：自动重启"""
        logger.error(f"音频采集错误: {error}")
        self._restart_count += 1

        if self._restart_count > self._max_restarts:
            logger.critical(f"音频采集重启次数超限 ({self._max_restarts})，退出")
            raise RuntimeError(f"音频采集失败，重启次数超过 {self._max_restarts}")

        logger.warning(f"尝试重启音频采集 ({self._restart_count}/{self._max_restarts})")
        self.stop()
        time.sleep(0.5)  # 短暂等待
        self.start()

    @property
    def is_running(self) -> bool:
        """是否正在运行"""
        return self._running
