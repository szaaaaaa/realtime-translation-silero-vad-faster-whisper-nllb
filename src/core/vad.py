"""
VAD (Voice Activity Detection) 模块
使用 Silero VAD 进行语音活动检测
"""
import numpy as np
import torch
import logging
from typing import Optional

from src.events import VadState, VadEvent

logger = logging.getLogger(__name__)


class SileroVAD:
    """
    Silero VAD 封装
    实现语音活动检测状态机
    """

    def __init__(self,
                 threshold: float = 0.50,
                 speech_start_frames: int = 6,
                 speech_end_frames: int = 10,
                 sample_rate: int = 16000):
        """
        初始化 VAD

        Args:
            threshold: VAD 阈值（0-1）
            speech_start_frames: 连续多少帧 speech 判定开始
            speech_end_frames: 连续多少帧 silence 判定结束
            sample_rate: 采样率（仅支持 16000 或 8000）
        """
        self.threshold = threshold
        self.speech_start_frames = speech_start_frames
        self.speech_end_frames = speech_end_frames
        self.sample_rate = sample_rate

        self._model = None
        self._state = VadState.SILENCE
        self._speech_counter = 0
        self._silence_counter = 0
        self._vad_iterator = None

    def load_model(self) -> None:
        """加载 Silero VAD 模型"""
        logger.info("正在加载 Silero VAD 模型...")

        try:
            self._model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self._model.eval()
            logger.info("Silero VAD 模型加载成功")
        except Exception as e:
            logger.error(f"加载 Silero VAD 模型失败: {e}")
            raise

    def process_frame(self, frame: np.ndarray,
                      timestamp: float) -> Optional[VadEvent]:
        """
        处理一帧音频，返回 VAD 事件（如有）

        Args:
            frame: 音频帧数据 (float32)
            timestamp: 帧时间戳

        Returns:
            VadEvent 或 None
        """
        if self._model is None:
            raise RuntimeError("VAD 模型未加载，请先调用 load_model()")

        # 转换为 tensor
        audio_tensor = torch.from_numpy(frame.astype(np.float32))

        # 获取 VAD 概率
        with torch.no_grad():
            speech_prob = self._model(audio_tensor, self.sample_rate).item()

        is_speech = speech_prob >= self.threshold

        # 状态机处理
        event = None

        if self._state == VadState.SILENCE:
            if is_speech:
                self._speech_counter += 1
                self._silence_counter = 0

                if self._speech_counter >= self.speech_start_frames:
                    # 转换到 SPEECH 状态
                    self._state = VadState.SPEECH
                    self._speech_counter = 0
                    event = VadEvent(
                        event_type="start",
                        timestamp=timestamp,
                        frame=frame.copy()
                    )
                    logger.debug(f"VAD_START at {timestamp:.2f}")
            else:
                self._speech_counter = 0

        elif self._state == VadState.SPEECH:
            if not is_speech:
                self._silence_counter += 1
                self._speech_counter = 0

                if self._silence_counter >= self.speech_end_frames:
                    # 转换到 SILENCE 状态
                    self._state = VadState.SILENCE
                    self._silence_counter = 0
                    event = VadEvent(
                        event_type="end",
                        timestamp=timestamp
                    )
                    logger.debug(f"VAD_END at {timestamp:.2f}")
                else:
                    # 还在语音中，继续输出帧
                    event = VadEvent(
                        event_type="frame",
                        timestamp=timestamp,
                        frame=frame.copy()
                    )
            else:
                self._silence_counter = 0
                self._speech_counter += 1
                # 输出语音帧
                event = VadEvent(
                    event_type="frame",
                    timestamp=timestamp,
                    frame=frame.copy()
                )

        return event

    def reset(self) -> None:
        """重置状态机"""
        self._state = VadState.SILENCE
        self._speech_counter = 0
        self._silence_counter = 0
        if self._model is not None:
            self._model.reset_states()
        logger.debug("VAD 状态已重置")

    @property
    def state(self) -> VadState:
        """当前 VAD 状态"""
        return self._state

    @property
    def is_speech(self) -> bool:
        """当前是否在语音状态"""
        return self._state == VadState.SPEECH
