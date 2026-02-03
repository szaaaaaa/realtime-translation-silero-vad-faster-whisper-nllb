"""
延迟打点日志模块
记录各阶段处理延迟
"""
import time
import logging
import csv
import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LatencyRecord:
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
        """ASR 延迟（从 chunk emit 到 ASR 完成）"""
        if self.t_asr_end > 0 and self.t_chunk_emit > 0:
            return self.t_asr_end - self.t_chunk_emit
        return 0.0

    @property
    def latency_mt(self) -> float:
        """MT 延迟"""
        if self.t_mt_end > 0 and self.t_mt_start > 0:
            return self.t_mt_end - self.t_mt_start
        return 0.0

    @property
    def latency_e2e(self) -> float:
        """端到端延迟（从采集到 UI 渲染）"""
        if self.t_ui_render > 0 and self.t_capture > 0:
            return self.t_ui_render - self.t_capture
        return 0.0

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'utterance_id': self.utterance_id,
            't_capture': self.t_capture,
            't_vad_start': self.t_vad_start,
            't_chunk_emit': self.t_chunk_emit,
            't_asr_start': self.t_asr_start,
            't_asr_end': self.t_asr_end,
            't_stabilize': self.t_stabilize,
            't_mt_start': self.t_mt_start,
            't_mt_end': self.t_mt_end,
            't_ui_render': self.t_ui_render,
            'latency_asr': self.latency_asr,
            'latency_mt': self.latency_mt,
            'latency_e2e': self.latency_e2e,
        }


class LatencyLogger:
    """
    延迟打点日志记录器
    将延迟数据写入 CSV 文件
    """

    def __init__(self, log_file: str = "logs/latency.csv"):
        """
        初始化延迟日志记录器

        Args:
            log_file: 日志文件路径
        """
        self.log_file = Path(log_file)
        self._records = []
        self._current_record: Optional[LatencyRecord] = None
        self._utterance_counter = 0

        # 确保日志目录存在
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # 写入 CSV 头
        if not self.log_file.exists():
            self._write_header()

    def _write_header(self):
        """写入 CSV 头"""
        headers = [
            'utterance_id', 't_capture', 't_vad_start', 't_chunk_emit',
            't_asr_start', 't_asr_end', 't_stabilize',
            't_mt_start', 't_mt_end', 't_ui_render',
            'latency_asr', 'latency_mt', 'latency_e2e'
        ]
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def start_utterance(self) -> str:
        """
        开始新的 utterance

        Returns:
            utterance ID
        """
        self._utterance_counter += 1
        utterance_id = f"utt_{self._utterance_counter:06d}"

        self._current_record = LatencyRecord(
            utterance_id=utterance_id,
            t_vad_start=time.time()
        )

        return utterance_id

    def log_capture(self, timestamp: float = None):
        """记录采集时间"""
        if self._current_record:
            self._current_record.t_capture = timestamp or time.time()

    def log_vad_start(self, timestamp: float = None):
        """记录 VAD 开始时间"""
        if self._current_record:
            self._current_record.t_vad_start = timestamp or time.time()

    def log_chunk_emit(self, timestamp: float = None):
        """记录 chunk 出队时间"""
        if self._current_record:
            self._current_record.t_chunk_emit = timestamp or time.time()

    def log_asr_start(self, timestamp: float = None):
        """记录 ASR 开始时间"""
        if self._current_record:
            self._current_record.t_asr_start = timestamp or time.time()

    def log_asr_end(self, timestamp: float = None):
        """记录 ASR 结束时间"""
        if self._current_record:
            self._current_record.t_asr_end = timestamp or time.time()

    def log_stabilize(self, timestamp: float = None):
        """记录稳定化时间"""
        if self._current_record:
            self._current_record.t_stabilize = timestamp or time.time()

    def log_mt_start(self, timestamp: float = None):
        """记录 MT 开始时间"""
        if self._current_record:
            self._current_record.t_mt_start = timestamp or time.time()

    def log_mt_end(self, timestamp: float = None):
        """记录 MT 结束时间"""
        if self._current_record:
            self._current_record.t_mt_end = timestamp or time.time()

    def log_ui_render(self, timestamp: float = None):
        """记录 UI 渲染时间"""
        if self._current_record:
            self._current_record.t_ui_render = timestamp or time.time()

    def end_utterance(self):
        """结束当前 utterance 并写入日志"""
        if self._current_record:
            self._write_record(self._current_record)

            # 输出到日志
            record = self._current_record
            logger.info(
                f"[{record.utterance_id}] "
                f"ASR={record.latency_asr*1000:.0f}ms "
                f"MT={record.latency_mt*1000:.0f}ms "
                f"E2E={record.latency_e2e*1000:.0f}ms"
            )

            self._current_record = None

    def _write_record(self, record: LatencyRecord):
        """写入一条记录到 CSV"""
        try:
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    record.utterance_id,
                    f"{record.t_capture:.3f}",
                    f"{record.t_vad_start:.3f}",
                    f"{record.t_chunk_emit:.3f}",
                    f"{record.t_asr_start:.3f}",
                    f"{record.t_asr_end:.3f}",
                    f"{record.t_stabilize:.3f}",
                    f"{record.t_mt_start:.3f}",
                    f"{record.t_mt_end:.3f}",
                    f"{record.t_ui_render:.3f}",
                    f"{record.latency_asr*1000:.1f}",
                    f"{record.latency_mt*1000:.1f}",
                    f"{record.latency_e2e*1000:.1f}",
                ])
        except Exception as e:
            logger.error(f"写入延迟日志失败: {e}")

    @property
    def current_record(self) -> Optional[LatencyRecord]:
        """当前记录"""
        return self._current_record
