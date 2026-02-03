"""
事件数据类定义
所有线程间通信的数据结构
"""
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum
import numpy as np
import time


class VadState(Enum):
    """VAD 状态"""
    SILENCE = "silence"
    SPEECH = "speech"


@dataclass
class AudioFrame:
    """音频帧数据"""
    samples: np.ndarray  # float32[320]
    timestamp: float     # 采集时间戳


@dataclass
class VadEvent:
    """VAD 事件"""
    event_type: str  # "start", "frame", "end"
    timestamp: float
    frame: Optional[np.ndarray] = None


@dataclass
class ASRChunk:
    """ASR 输入块"""
    audio: np.ndarray     # float32[n]
    t_start: float        # 起始时间戳
    t_end: float          # 结束时间戳
    is_final: bool        # 是否为 utterance 结束块
    t_emit: float = 0.0   # 出队时间（用于延迟计算）

    def __post_init__(self):
        if self.t_emit == 0.0:
            self.t_emit = time.time()


@dataclass
class ASRResult:
    """ASR 识别结果"""
    text: str                 # 识别文本
    segments: List[dict]      # 分段信息
    t_start: float            # 音频起始时间
    t_end: float              # 音频结束时间
    is_final: bool            # 是否为 final 结果
    t_asr_start: float = 0.0  # ASR 开始时间
    t_asr_end: float = 0.0    # ASR 结束时间


@dataclass
class StabilizerOutput:
    """稳定器输出"""
    partial_src: str                    # 当前显示文本
    final_append_src: Optional[str] = None  # final 增量（仅 final 时）
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class MTResult:
    """翻译结果"""
    source: str           # 原文
    target: str           # 译文
    t_mt_start: float     # 翻译开始时间
    t_mt_end: float       # 翻译结束时间


@dataclass
class UIUpdate:
    """UI 更新事件"""
    partial_src: str = ""
    final_append_src: str = ""
    final_append_tgt: str = ""
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class LatencyLog:
    """延迟打点记录"""
    utterance_id: str = ""
    t_capture: float = 0.0
    t_vad_start: float = 0.0
    t_chunk_emit: float = 0.0
    t_asr_start: float = 0.0
    t_asr_end: float = 0.0
    t_stabilize: float = 0.0
    t_mt_start: float = 0.0
    t_mt_end: float = 0.0
    t_ui_render: float = 0.0

    @property
    def latency_asr(self) -> float:
        """ASR 延迟"""
        return self.t_asr_end - self.t_chunk_emit

    @property
    def latency_mt(self) -> float:
        """MT 延迟"""
        return self.t_mt_end - self.t_mt_start

    @property
    def latency_e2e(self) -> float:
        """端到端延迟"""
        return self.t_ui_render - self.t_capture
