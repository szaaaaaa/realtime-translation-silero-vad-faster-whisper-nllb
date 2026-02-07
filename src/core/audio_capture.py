"""
音频采集模块 - WASAPI/麦克风采集
支持系统回环和麦克风输入，输出 20ms 帧
"""
import sounddevice as sd
import numpy as np
from typing import Optional, List, Dict
from collections import deque
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

    # Silero VAD 要求 16000Hz 下每帧恰好 512 个采样点
    VAD_FRAME_SAMPLES = 512
    PROFILE_COMPUTER_INPUT = "computer_input"
    PROFILE_HEADSET_INPUT = "headset_input"
    PROFILE_COMPUTER_LOOPBACK = "computer_loopback"

    _HEADSET_KEYWORDS = (
        "headset", "headphone", "earphone", "airpods", "buds", "hands-free",
        "bluetooth", "耳机", "耳麦"
    )
    _LOOPBACK_KEYWORDS = (
        "loopback", "stereo mix", "what u hear", "monitor", "立体声混音", "内录"
    )
    _COMPUTER_MIC_KEYWORDS = (
        "microphone", "mic", "array", "line in", "麦克风", "阵列", "内置"
    )
    _VIRTUAL_DEVICE_KEYWORDS = (
        "cable", "virtual", "voicemeeter", "obs", "nvidia broadcast", "blackhole"
    )

    def __init__(self,
                 mode: str = "loopback",
                 device_index: Optional[int] = None,
                 sample_rate: int = 16000,
                 frame_ms: int = 20,
                 channels: int = 1,
                 ring_buffer_seconds: int = 20,
                 pre_roll_ms: int = 400):
        """
        初始化音频采集器

        Args:
            mode: 采集模式 ("loopback" 或 "mic")
            device_index: 设备索引，None 为自动选择
            sample_rate: 目标采样率，固定 16000Hz
            frame_ms: 设备端帧时长（毫秒），用于计算 blocksize
        """
        self.mode = mode
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.channels = channels
        self.frame_samples = self.VAD_FRAME_SAMPLES  # 512 for Silero VAD
        self.pre_roll_ms = pre_roll_ms

        self._native_sr: int = sample_rate  # 设备原生采样率，start() 时更新
        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._restart_count = 0
        self._max_restarts = 5
        max_frames = max(100, int(ring_buffer_seconds * 1000 / self.frame_ms))
        self._frame_queue: queue.Queue = queue.Queue(maxsize=max_frames)
        self._pre_roll_frames: deque = deque(
            maxlen=max(1, int(self.pre_roll_ms / self.frame_ms))
        )
        self._resample_buf = np.array([], dtype=np.float32)  # 重采样后的累积缓冲区

    def list_devices(self) -> List[Dict]:
        """列出所有可用的音频输入设备"""
        devices = []
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:
                hostapi = sd.query_hostapis(dev['hostapi'])
                name_lower = dev['name'].lower()
                devices.append({
                    'index': i,
                    'name': dev['name'],
                    'channels': dev['max_input_channels'],
                    'sample_rate': dev['default_samplerate'],
                    'hostapi': hostapi['name'],
                    'is_wasapi': 'WASAPI' in hostapi['name'],
                    'is_loopback': (
                        'loopback' in name_lower
                        or 'stereo mix' in name_lower
                        or '立体声混音' in dev['name']
                    )
                })
        return devices

    def get_default_devices(self) -> Dict[str, Optional[int]]:
        """获取系统默认输入/输出设备索引"""
        try:
            default_in, default_out = sd.default.device
        except Exception:
            return {'input': None, 'output': None}

        def _normalize(idx):
            if idx is None:
                return None
            try:
                idx = int(idx)
            except Exception:
                return None
            return idx if idx >= 0 else None

        return {
            'input': _normalize(default_in),
            'output': _normalize(default_out)
        }

    @staticmethod
    def _contains_any(text: str, keywords: tuple) -> bool:
        return any(keyword in text for keyword in keywords)

    def _infer_profile(self, dev: Dict) -> str:
        name = dev['name'].lower()

        if dev.get('is_loopback') or self._contains_any(name, self._LOOPBACK_KEYWORDS):
            return self.PROFILE_COMPUTER_LOOPBACK
        if self._contains_any(name, self._HEADSET_KEYWORDS):
            return self.PROFILE_HEADSET_INPUT
        return self.PROFILE_COMPUTER_INPUT

    def _score_device(self, dev: Dict, profile: str, defaults: Dict[str, Optional[int]]) -> int:
        name = dev['name'].lower()
        score = 0

        if dev.get('is_wasapi'):
            score += 30

        if profile == self.PROFILE_COMPUTER_LOOPBACK:
            if dev.get('is_loopback'):
                score += 60
            if self._contains_any(name, self._LOOPBACK_KEYWORDS):
                score += 40
            if defaults.get('output') == dev['index']:
                score += 25
        elif profile == self.PROFILE_HEADSET_INPUT:
            if self._contains_any(name, self._HEADSET_KEYWORDS):
                score += 50
            if defaults.get('input') == dev['index']:
                score += 25
        else:
            if self._contains_any(name, self._COMPUTER_MIC_KEYWORDS):
                score += 35
            if self._contains_any(name, self._HEADSET_KEYWORDS):
                score -= 15
            if defaults.get('input') == dev['index']:
                score += 25

        if self._contains_any(name, self._VIRTUAL_DEVICE_KEYWORDS):
            score -= 20

        return score

    def pick_preferred_devices(self) -> Dict[str, Optional[Dict]]:
        """
        选择三类首选输入设备：
        - computer_input: 电脑输入（内置/外接麦克风）
        - headset_input: 耳机输入（耳机麦克风）
        - computer_loopback: 电脑内录（系统回环）
        """
        devices = self.list_devices()
        defaults = self.get_default_devices()
        best = {
            self.PROFILE_COMPUTER_INPUT: None,
            self.PROFILE_HEADSET_INPUT: None,
            self.PROFILE_COMPUTER_LOOPBACK: None
        }
        best_scores = {
            self.PROFILE_COMPUTER_INPUT: -10**9,
            self.PROFILE_HEADSET_INPUT: -10**9,
            self.PROFILE_COMPUTER_LOOPBACK: -10**9
        }

        for dev in devices:
            profile = self._infer_profile(dev)
            score = self._score_device(dev, profile, defaults)
            if score > best_scores[profile]:
                best_scores[profile] = score
                best[profile] = dev

        if best[self.PROFILE_COMPUTER_INPUT] is None:
            default_in = defaults.get('input')
            for dev in devices:
                if dev['index'] == default_in:
                    best[self.PROFILE_COMPUTER_INPUT] = dev
                    break

        if best[self.PROFILE_HEADSET_INPUT] is None and best[self.PROFILE_COMPUTER_INPUT] is not None:
            # 没有耳机输入时，允许回落到电脑输入，避免 UI 出现不可用项
            best[self.PROFILE_HEADSET_INPUT] = best[self.PROFILE_COMPUTER_INPUT]

        return best

    def list_preferred_input_options(self) -> List[Dict]:
        """返回固定三类输入选项（UI 只展示这三类）"""
        best = self.pick_preferred_devices()
        profile_specs = [
            (self.PROFILE_COMPUTER_INPUT, "电脑输入", "mic"),
            (self.PROFILE_HEADSET_INPUT, "耳机输入", "mic"),
            (self.PROFILE_COMPUTER_LOOPBACK, "电脑内录", "loopback"),
        ]

        options: List[Dict] = []
        for profile, label, mode in profile_specs:
            dev = best.get(profile)
            options.append({
                "profile": profile,
                "label": label,
                "mode": mode,
                "available": dev is not None,
                "index": None if dev is None else dev['index'],
                "name": "" if dev is None else dev['name'],
                "hostapi": "" if dev is None else dev['hostapi']
            })
        return options

    def _find_device(self) -> int:
        """根据模式自动选择设备"""
        if self.device_index is not None:
            return self.device_index

        preferred = self.pick_preferred_devices()

        if self.mode == "loopback":
            dev = preferred.get(self.PROFILE_COMPUTER_LOOPBACK)
            if dev:
                logger.info(f"自动选择电脑内录设备: [{dev['index']}] {dev['name']}")
                return dev['index']
            logger.warning("未找到电脑内录设备，回落到系统默认输入")
            return None

        # mic 模式优先电脑输入，再回落耳机输入
        dev = preferred.get(self.PROFILE_COMPUTER_INPUT) or preferred.get(self.PROFILE_HEADSET_INPUT)
        if dev:
            logger.info(f"自动选择麦克风输入设备: [{dev['index']}] {dev['name']}")
            return dev['index']
        return None

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """线性插值重采样"""
        if orig_sr == target_sr:
            return audio
        target_len = int(len(audio) * target_sr / orig_sr)
        indices = np.linspace(0, len(audio) - 1, target_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

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
        audio_data = audio_data.flatten()

        # 重采样到目标采样率 (16000 Hz)
        if self._native_sr != self.sample_rate:
            audio_data = self._resample(audio_data, self._native_sr, self.sample_rate)

        # 累积到缓冲区，按 512 采样切帧
        self._resample_buf = np.concatenate([self._resample_buf, audio_data])
        now = time.time()

        while len(self._resample_buf) >= self.VAD_FRAME_SAMPLES:
            chunk = self._resample_buf[:self.VAD_FRAME_SAMPLES]
            self._resample_buf = self._resample_buf[self.VAD_FRAME_SAMPLES:]

            frame = AudioFrame(samples=chunk, timestamp=now)
            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    self._frame_queue.get_nowait()
                    self._frame_queue.put_nowait(frame)
                except queue.Empty:
                    pass
            self._pre_roll_frames.append(chunk.copy())

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
            self._native_sr = native_sr
            self._resample_buf = np.array([], dtype=np.float32)

            self._stream = sd.InputStream(
                device=device_idx,
                samplerate=native_sr,
                channels=self.channels,
                dtype=np.float32,
                blocksize=blocksize,
                callback=self._audio_callback
            )
            self._stream.start()
            self._running = True
            self._restart_count = 0
            logger.info(f"音频采集已启动: device={device_idx}, native_sr={native_sr}, "
                        f"target_sr={self.sample_rate}, blocksize={blocksize}")
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

    def get_pre_roll_audio(self) -> np.ndarray:
        """获取最近一小段预滚音频（用于句首补偿）"""
        if not self._pre_roll_frames:
            return np.array([], dtype=np.float32)
        return np.concatenate(list(self._pre_roll_frames), axis=0)

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
