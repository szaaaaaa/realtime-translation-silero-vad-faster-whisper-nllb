"""
音频分块器模块
将 VAD 判定的语音帧分块输出给 ASR
"""
import numpy as np
import time
import logging
from typing import Optional, List

from src.events import ASRChunk

logger = logging.getLogger(__name__)


class Chunker:
    """
    音频分块器
    将连续的语音帧分块，支持 overlap
    """

    def __init__(self,
                 chunk_ms: int = 1000,
                 overlap_ms: int = 200,
                 max_utterance_ms: int = 10000,
                 tail_pad_ms: int = 120,
                 sample_rate: int = 16000):
        """
        初始化分块器

        Args:
            chunk_ms: 每个 chunk 的时长（毫秒）
            overlap_ms: overlap 时长（毫秒）
            max_utterance_ms: 最大 utterance 时长（毫秒）
            tail_pad_ms: final chunk 尾部填充静音（毫秒）
            sample_rate: 采样率
        """
        self.chunk_samples = chunk_ms * sample_rate // 1000
        self.overlap_samples = overlap_ms * sample_rate // 1000
        self.max_utterance_samples = max_utterance_ms * sample_rate // 1000
        self.tail_pad_samples = tail_pad_ms * sample_rate // 1000
        self.sample_rate = sample_rate

        self._buffer: List[np.ndarray] = []
        self._buffer_samples = 0
        self._overlap_buffer = np.zeros(self.overlap_samples, dtype=np.float32)
        self._utterance_start_time = 0.0
        self._last_chunk_time = 0.0
        self._total_utterance_samples = 0

    def on_vad_start(self, timestamp: float) -> None:
        """
        VAD 开始事件

        Args:
            timestamp: 开始时间戳
        """
        self._buffer = []
        self._buffer_samples = 0
        self._utterance_start_time = timestamp
        self._last_chunk_time = timestamp
        self._total_utterance_samples = 0
        logger.debug(f"Chunker: utterance started at {timestamp:.2f}")

    def on_speech_frame(self, frame: np.ndarray,
                        timestamp: float) -> Optional[ASRChunk]:
        """
        处理语音帧，可能返回 partial chunk

        Args:
            frame: 语音帧数据
            timestamp: 帧时间戳

        Returns:
            ASRChunk 或 None
        """
        frame = frame.astype(np.float32).flatten()
        self._buffer.append(frame)
        self._buffer_samples += len(frame)
        self._total_utterance_samples += len(frame)

        # 检查是否需要输出 partial chunk
        if self._buffer_samples >= self.chunk_samples:
            return self._emit_chunk(timestamp, is_final=False)

        # 检查是否超过最大 utterance 长度
        if self._total_utterance_samples >= self.max_utterance_samples:
            logger.debug(f"Chunker: max utterance reached, forcing final")
            return self._emit_chunk(timestamp, is_final=True)

        return None

    def on_vad_end(self, timestamp: float) -> Optional[ASRChunk]:
        """
        VAD 结束事件，返回 final chunk

        Args:
            timestamp: 结束时间戳

        Returns:
            ASRChunk（final）或 None
        """
        if not self._buffer:
            logger.debug("Chunker: VAD ended with empty buffer")
            return None

        return self._emit_chunk(timestamp, is_final=True)

    def _emit_chunk(self, timestamp: float, is_final: bool) -> ASRChunk:
        """
        输出一个 chunk

        Args:
            timestamp: 当前时间戳
            is_final: 是否为 final chunk

        Returns:
            ASRChunk
        """
        # 合并缓冲区
        if self._buffer:
            audio = np.concatenate(self._buffer, axis=0)
        else:
            audio = np.array([], dtype=np.float32)

        # 添加 overlap（头部历史音频）
        if len(self._overlap_buffer) > 0 and len(audio) > 0:
            audio = np.concatenate([self._overlap_buffer, audio])

        # final chunk 添加尾部静音
        if is_final and self.tail_pad_samples > 0:
            tail_pad = np.zeros(self.tail_pad_samples, dtype=np.float32)
            audio = np.concatenate([audio, tail_pad])

        # 更新 overlap buffer（保存当前 chunk 的尾部）
        if len(audio) >= self.overlap_samples:
            self._overlap_buffer = audio[-self.overlap_samples:].copy()
        else:
            self._overlap_buffer = audio.copy()

        # 创建 ASRChunk
        chunk = ASRChunk(
            audio=audio,
            t_start=self._last_chunk_time,
            t_end=timestamp,
            is_final=is_final,
            t_emit=time.time()
        )

        # 清空缓冲区（非 final 时保留一些用于下一个 chunk）
        if is_final:
            self._buffer = []
            self._buffer_samples = 0
            self._total_utterance_samples = 0
        else:
            # 保留 overlap 部分用于下一个 chunk
            self._buffer = []
            self._buffer_samples = 0

        self._last_chunk_time = timestamp

        duration = len(chunk.audio) / self.sample_rate
        logger.debug(f"Chunker: emitted {'final' if is_final else 'partial'} chunk, "
                     f"duration={duration:.2f}s")

        return chunk

    def reset(self) -> None:
        """重置状态"""
        self._buffer = []
        self._buffer_samples = 0
        self._overlap_buffer = np.zeros(self.overlap_samples, dtype=np.float32)
        self._utterance_start_time = 0.0
        self._last_chunk_time = 0.0
        self._total_utterance_samples = 0
        logger.debug("Chunker: reset")
