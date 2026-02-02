"""
环形缓冲区模块
线程安全的音频环形缓冲区实现
"""
import numpy as np
import threading
from typing import Optional


class RingBuffer:
    """
    线程安全的环形缓冲区
    用于音频数据的缓冲存储
    """

    def __init__(self,
                 capacity_seconds: float = 30.0,
                 sample_rate: int = 16000):
        """
        初始化环形缓冲区

        Args:
            capacity_seconds: 缓冲区容量（秒）
            sample_rate: 采样率
        """
        self.capacity = int(capacity_seconds * sample_rate)
        self.sample_rate = sample_rate
        self._buffer = np.zeros(self.capacity, dtype=np.float32)
        self._write_pos = 0
        self._available = 0  # 可读数据量
        self._lock = threading.Lock()

    def push(self, samples: np.ndarray) -> int:
        """
        追加样本，返回实际写入数量

        Args:
            samples: 要写入的样本数据

        Returns:
            实际写入的样本数量
        """
        samples = np.asarray(samples, dtype=np.float32).flatten()
        n = len(samples)

        with self._lock:
            if n >= self.capacity:
                # 数据超过容量，只保留最新的 capacity 个样本
                samples = samples[-self.capacity:]
                n = self.capacity
                self._buffer[:] = samples
                self._write_pos = 0
                self._available = self.capacity
                return n

            # 分两段写入（处理环形）
            first_part = min(n, self.capacity - self._write_pos)
            self._buffer[self._write_pos:self._write_pos + first_part] = samples[:first_part]

            if first_part < n:
                # 需要绕回到开头
                second_part = n - first_part
                self._buffer[:second_part] = samples[first_part:]

            self._write_pos = (self._write_pos + n) % self.capacity
            self._available = min(self._available + n, self.capacity)

            return n

    def pop(self, n_samples: int) -> np.ndarray:
        """
        取出 n 个样本（移除）

        Args:
            n_samples: 要取出的样本数量

        Returns:
            取出的样本数据
        """
        with self._lock:
            n = min(n_samples, self._available)
            if n == 0:
                return np.array([], dtype=np.float32)

            # 计算读取位置
            read_pos = (self._write_pos - self._available) % self.capacity

            # 分两段读取
            result = np.zeros(n, dtype=np.float32)
            first_part = min(n, self.capacity - read_pos)
            result[:first_part] = self._buffer[read_pos:read_pos + first_part]

            if first_part < n:
                second_part = n - first_part
                result[first_part:] = self._buffer[:second_part]

            self._available -= n
            return result

    def peek(self, n_samples: int) -> np.ndarray:
        """
        读取 n 个样本（不移除）

        Args:
            n_samples: 要读取的样本数量

        Returns:
            读取的样本数据
        """
        with self._lock:
            n = min(n_samples, self._available)
            if n == 0:
                return np.array([], dtype=np.float32)

            # 计算读取位置
            read_pos = (self._write_pos - self._available) % self.capacity

            # 分两段读取
            result = np.zeros(n, dtype=np.float32)
            first_part = min(n, self.capacity - read_pos)
            result[:first_part] = self._buffer[read_pos:read_pos + first_part]

            if first_part < n:
                second_part = n - first_part
                result[first_part:] = self._buffer[:second_part]

            return result

    def peek_recent(self, n_samples: int) -> np.ndarray:
        """
        读取最近的 n 个样本（不移除）

        Args:
            n_samples: 要读取的样本数量

        Returns:
            最近的样本数据
        """
        with self._lock:
            n = min(n_samples, self._available)
            if n == 0:
                return np.array([], dtype=np.float32)

            # 计算读取起始位置（从 write_pos 往前数 n 个）
            start_pos = (self._write_pos - n) % self.capacity

            # 分两段读取
            result = np.zeros(n, dtype=np.float32)
            first_part = min(n, self.capacity - start_pos)
            result[:first_part] = self._buffer[start_pos:start_pos + first_part]

            if first_part < n:
                second_part = n - first_part
                result[first_part:] = self._buffer[:second_part]

            return result

    def clear(self) -> None:
        """清空缓冲区"""
        with self._lock:
            self._buffer.fill(0)
            self._write_pos = 0
            self._available = 0

    @property
    def available(self) -> int:
        """当前可读样本数"""
        with self._lock:
            return self._available

    @property
    def usage_ratio(self) -> float:
        """缓冲区使用率"""
        with self._lock:
            return self._available / self.capacity if self.capacity > 0 else 0.0

    @property
    def available_seconds(self) -> float:
        """当前可读时长（秒）"""
        with self._lock:
            return self._available / self.sample_rate if self.sample_rate > 0 else 0.0
