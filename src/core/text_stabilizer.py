"""
文本稳定器模块
减少 partial ASR 结果的抖动
"""
import time
import logging
from typing import Optional

from src.events import ASRResult, StabilizerOutput

logger = logging.getLogger(__name__)


class TextStabilizer:
    """
    文本稳定器 - 减少 partial 抖动
    使用 LCP（最长公共前缀）算法稳定输出
    """

    def __init__(self,
                 lock_min_chars: int = 12,
                 buffer_keep_chars: int = 80,
                 lcp_min_chars: int = 8):
        """
        初始化稳定器

        Args:
            lock_min_chars: 锁定文本的最小字符数
            buffer_keep_chars: 缓冲区保留的最大字符数
            lcp_min_chars: LCP 匹配的最小字符数
        """
        self.lock_min_chars = lock_min_chars
        self.buffer_keep_chars = buffer_keep_chars
        self.lcp_min_chars = lcp_min_chars

        self._locked_text = ""
        self._buffer_text = ""
        self._last_final_pos = 0

    def process(self, asr_result: ASRResult) -> StabilizerOutput:
        """
        处理 ASR 结果，返回稳定化输出

        Args:
            asr_result: ASR 识别结果

        Returns:
            StabilizerOutput
        """
        candidate = asr_result.text
        current = self._locked_text + self._buffer_text

        # 计算最长公共前缀
        lcp = self._longest_common_prefix(current, candidate)

        # 推进 locked_text
        if len(lcp) >= self.lcp_min_chars:
            if len(lcp) > len(self._locked_text):
                # 可以锁定更多文本
                self._locked_text = lcp

        # 更新 buffer
        if len(candidate) > len(self._locked_text):
            self._buffer_text = candidate[len(self._locked_text):]
        else:
            self._buffer_text = ""

        # 限制 buffer 长度
        if len(self._buffer_text) > self.buffer_keep_chars:
            self._buffer_text = self._buffer_text[-self.buffer_keep_chars:]

        partial_src = self._locked_text + self._buffer_text

        # Final 处理
        final_append_src = None
        if asr_result.is_final:
            # final 时，将所有文本作为最终输出
            final_append_src = candidate[self._last_final_pos:] if len(candidate) > self._last_final_pos else candidate

            # 更新状态
            self._locked_text = ""
            self._buffer_text = ""
            self._last_final_pos = 0  # 重置

            logger.debug(f"Stabilizer: final_append='{final_append_src[:50]}...'")

        return StabilizerOutput(
            partial_src=partial_src,
            final_append_src=final_append_src,
            timestamp=time.time()
        )

    def _longest_common_prefix(self, s1: str, s2: str) -> str:
        """
        计算最长公共前缀

        Args:
            s1: 字符串1
            s2: 字符串2

        Returns:
            最长公共前缀
        """
        i = 0
        min_len = min(len(s1), len(s2))
        while i < min_len and s1[i] == s2[i]:
            i += 1
        return s1[:i]

    def reset(self) -> None:
        """重置状态"""
        self._locked_text = ""
        self._buffer_text = ""
        self._last_final_pos = 0
        logger.debug("Stabilizer: reset")

    @property
    def current_text(self) -> str:
        """当前稳定的文本"""
        return self._locked_text + self._buffer_text
