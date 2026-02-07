"""
VAD (Voice Activity Detection) module using Silero VAD.
"""

import logging
from typing import Optional

import numpy as np
import torch

from src.events import VadEvent, VadState

logger = logging.getLogger(__name__)


class SileroVAD:
    """
    Silero VAD wrapper with a simple start/end state machine.
    """

    def __init__(
        self,
        threshold: float = 0.50,
        speech_start_frames: int = 6,
        speech_end_frames: int = 10,
        sample_rate: int = 16000,
    ):
        self.threshold = float(threshold)
        self.speech_start_frames = max(1, int(speech_start_frames))
        self.speech_end_frames = max(1, int(speech_end_frames))
        self.sample_rate = int(sample_rate)

        self._model = None
        self._state = VadState.SILENCE
        self._speech_counter = 0
        self._silence_counter = 0
        # Cache pre-start speech frames so sentence head is not lost.
        self._start_frame_buffer: list[np.ndarray] = []

    def load_model(self) -> None:
        """Load Silero VAD model."""
        logger.info("正在加载 Silero VAD 模型...")

        try:
            self._model, _utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )
            self._model.eval()
            logger.info("Silero VAD 模型加载成功")
        except Exception as e:
            logger.error(f"加载 Silero VAD 模型失败: {e}")
            raise

    def process_frame(self, frame: np.ndarray, timestamp: float) -> Optional[VadEvent]:
        """
        Process one audio frame and return VAD event if any.
        """
        if self._model is None:
            raise RuntimeError("VAD 模型未加载，请先调用 load_model()")

        audio_tensor = torch.from_numpy(frame.astype(np.float32))
        with torch.no_grad():
            speech_prob = float(self._model(audio_tensor, self.sample_rate).item())

        is_speech = speech_prob >= self.threshold
        event: Optional[VadEvent] = None

        if self._state == VadState.SILENCE:
            if is_speech:
                self._speech_counter += 1
                self._silence_counter = 0

                self._start_frame_buffer.append(frame.copy())
                keep = self.speech_start_frames + 2
                if len(self._start_frame_buffer) > keep:
                    self._start_frame_buffer = self._start_frame_buffer[-keep:]

                if self._speech_counter >= self.speech_start_frames:
                    self._state = VadState.SPEECH
                    self._speech_counter = 0
                    start_audio = (
                        np.concatenate(self._start_frame_buffer, axis=0)
                        if self._start_frame_buffer
                        else frame.copy()
                    )
                    self._start_frame_buffer = []
                    event = VadEvent(event_type="start", timestamp=timestamp, frame=start_audio)
                    logger.debug(f"VAD_START at {timestamp:.2f}")
            else:
                self._speech_counter = 0
                self._start_frame_buffer = []

        elif self._state == VadState.SPEECH:
            if not is_speech:
                self._silence_counter += 1
                self._speech_counter = 0

                if self._silence_counter >= self.speech_end_frames:
                    self._state = VadState.SILENCE
                    self._silence_counter = 0
                    self._start_frame_buffer = []
                    event = VadEvent(event_type="end", timestamp=timestamp)
                    logger.debug(f"VAD_END at {timestamp:.2f}")
                else:
                    event = VadEvent(event_type="frame", timestamp=timestamp, frame=frame.copy())
            else:
                self._silence_counter = 0
                self._speech_counter += 1
                event = VadEvent(event_type="frame", timestamp=timestamp, frame=frame.copy())

        return event

    def reset(self) -> None:
        """Reset VAD state machine."""
        self._state = VadState.SILENCE
        self._speech_counter = 0
        self._silence_counter = 0
        self._start_frame_buffer = []

        if self._model is not None:
            self._model.reset_states()

        logger.debug("VAD 状态已重置")

    @property
    def state(self) -> VadState:
        """Current VAD state."""
        return self._state

    @property
    def is_speech(self) -> bool:
        """Whether VAD is currently in speech state."""
        return self._state == VadState.SPEECH
