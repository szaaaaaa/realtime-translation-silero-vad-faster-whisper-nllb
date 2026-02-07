"""
MT (Machine Translation) 工作线程模块
使用 NLLB 进行文本翻译
"""
import threading
import time
import logging
from queue import Queue, Empty
import re
from typing import Optional, List
from collections import OrderedDict

from src.events import MTResult

logger = logging.getLogger(__name__)


class LRUCache:
    """简单的 LRU 缓存实现"""

    def __init__(self, capacity: int = 2048):
        self.capacity = capacity
        self._cache = OrderedDict()

    def get(self, key: str) -> Optional[str]:
        if key in self._cache:
            # 移动到末尾（最近使用）
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, value: str) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.capacity:
                # 删除最旧的
                self._cache.popitem(last=False)
        self._cache[key] = value

    def __len__(self) -> int:
        return len(self._cache)


class MTWorker:
    """
    MT 工作线程
    使用 NLLB 进行文本翻译
    不继承 threading.Thread，内部管理线程实例以支持重复 start/stop
    """

    _INCOMPLETE_TAIL_RE = re.compile(
        r"(?:\b(?:and|or|but|because|if|when|while|that|which|who|whose|whom|where|"
        r"to|of|for|with|as|than|from|into|about|without|within|through)\b|"
        r"(?:\u56e0\u4e3a|\u6240\u4ee5|\u4f46\u662f|\u800c\u4e14|\u5e76\u4e14|"
        r"\u5982\u679c|\u867d\u7136|\u7136\u800c|\u5e76|\u4e14|\u548c|\u4e0e|"
        r"\u5728|\u5bf9|\u628a|\u88ab))\s*$",
        re.IGNORECASE
    )

    def __init__(self,
                 input_queue: Queue,
                 output_queue: Queue,
                 model_name: str = "facebook/nllb-200-distilled-600M",
                 src_lang: str = "eng_Latn",
                 tgt_lang: str = "zho_Hans",
                 device: str = "cuda",
                 cache_size: int = 2048,
                 num_beams: int = 1,
                 flush_each_input: bool = False,
                 batch_max_wait_ms: int = 120,
                 continuation_wait_ms: int = 280,
                 continuation_max_chars: int = 140,
                 batch_max_chars: int = 220,
                 max_chars: int = 360):
        """
        初始化 MT Worker

        Args:
            input_queue: 输入队列（str）
            output_queue: 输出队列（MTResult）
            model_name: NLLB 模型名称
            src_lang: 源语言代码
            tgt_lang: 目标语言代码
            device: 运行设备 (cuda, cpu)
            cache_size: 缓存大小
        """
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.device = device
        self.cache_size = cache_size
        self.num_beams = max(1, int(num_beams))
        self.flush_each_input = bool(flush_each_input)
        self.batch_max_wait_ms = max(20, int(batch_max_wait_ms))
        self.continuation_wait_ms = max(self.batch_max_wait_ms, int(continuation_wait_ms))
        self.continuation_max_chars = max(40, int(continuation_max_chars))
        self.batch_max_chars = max(50, int(batch_max_chars))
        self.max_chars = max(120, int(max_chars))

        self._model = None
        self._tokenizer = None
        self._cache = LRUCache(cache_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._error_count = 0
        self._max_errors = 3
        self._pending_parts: List[str] = []
        self._pending_started_at = 0.0

    def load_model(self) -> None:
        """Load translation model."""
        logger.info(f"Loading NLLB model: {self.model_name}")
        logger.info(f"Direction: {self.src_lang} -> {self.tgt_lang}")

        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            if self.device == "auto":
                model_kwargs = {"device_map": "auto"}
                if torch.cuda.is_available():
                    model_kwargs["torch_dtype"] = torch.float16
                try:
                    self._model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.model_name,
                        **model_kwargs
                    )
                    logger.info("NLLB model loaded with device_map=auto")
                except Exception as e:
                    logger.warning(f"device_map=auto failed, fallback to CPU: {e}")
                    self.device = "cpu"
                    self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            else:
                self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                if self.device == "cuda":
                    if torch.cuda.is_available():
                        self._model = self._model.to("cuda")
                        logger.info("NLLB model loaded on GPU")
                    else:
                        logger.warning("CUDA unavailable, fallback to CPU")
                        self.device = "cpu"
                else:
                    logger.info("NLLB model loaded on CPU")
                    self.device = "cpu"

            self._model.eval()
            logger.info("NLLB model load completed")

        except Exception as e:
            logger.error(f"Failed to load NLLB model: {e}")
            raise

    def start(self) -> None:
        """启动工作线程（每次创建新的 Thread 实例）"""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("MT Worker 已在运行")
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """Worker main loop."""
        if self._model is None:
            logger.error("MT model not loaded")
            return

        logger.info("MT Worker started")

        while self._running:
            try:
                text = self.input_queue.get(timeout=0.05)

                if self.flush_each_input:
                    if text and text.strip():
                        result = self._translate(text)
                        self.output_queue.put(result)
                        self._error_count = 0
                    continue

                if text and text.strip():
                    self._collect_pending(text)

                flushed = self._flush_pending_if_needed(force=False)
                if flushed is not None:
                    self.output_queue.put(flushed)
                    self._error_count = 0
            except Empty:
                if self.flush_each_input:
                    continue
                flushed = self._flush_pending_if_needed(force=True)
                if flushed is not None:
                    self.output_queue.put(flushed)
                continue
            except Exception as e:
                self._handle_error(e)

        logger.info("MT Worker stopped")

    def _collect_pending(self, text: str) -> None:
        """收集短文本，降低每次 generate 的开销并提升上下文完整性。"""
        normalized = self._normalize_text(text)
        if not normalized:
            return

        if not self._pending_parts:
            self._pending_started_at = time.time()

        self._pending_parts.append(normalized)

    def _flush_pending_if_needed(self, force: bool) -> Optional[MTResult]:
        """Flush pending text only when context is likely complete."""
        if not self._pending_parts:
            return None

        joined = self._normalize_text(" ".join(self._pending_parts))
        pending_chars = len(joined)
        waited_ms = (time.time() - self._pending_started_at) * 1000

        boundary = self._has_sentence_boundary(joined)
        should_flush = pending_chars >= self.batch_max_chars or boundary

        if not should_flush and force:
            wait_ms = self.batch_max_wait_ms
            if self._looks_like_incomplete_clause(joined) and pending_chars <= self.continuation_max_chars:
                wait_ms = self.continuation_wait_ms
            should_flush = waited_ms >= wait_ms

        if not should_flush:
            return None

        self._pending_parts.clear()
        self._pending_started_at = 0.0
        return self._translate(joined)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _has_sentence_boundary(text: str) -> bool:
        return bool(re.search(r"[.!?\u3002\uff01\uff1f][\"\)\]]*\s*$", text))

    def _looks_like_incomplete_clause(self, text: str) -> bool:
        if not text or self._has_sentence_boundary(text):
            return False
        if re.search(r"[,;:\uff0c\uff1b\uff1a]\s*$", text):
            return True
        if self._INCOMPLETE_TAIL_RE.search(text):
            return True
        return bool(re.search(r"[\w\u4e00-\u9fff]$", text))

    @staticmethod
    def _truncate_on_boundary(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        window = text[:max_chars]
        markers = ['。', '！', '？', '.', '!', '?', ',', ';', '，', '；']
        cut = max((window.rfind(m) for m in markers), default=-1)
        if cut >= int(max_chars * 0.6):
            return window[:cut + 1]
        return window

    def _translate(self, text: str) -> MTResult:
        """
        执行翻译

        Args:
            text: 输入文本

        Returns:
            MTResult
        """
        normalized = self._normalize_text(text)

        # 检查缓存
        cached = self._cache.get(normalized)
        if cached is not None:
            logger.debug(f"MT cache hit: '{text[:30]}...'")
            return MTResult(
                source=normalized,
                target=cached,
                t_mt_start=time.time(),
                t_mt_end=time.time()
            )

        t_start = time.time()

        truncated = self._truncate_on_boundary(normalized, self.max_chars)

        try:
            import torch

            # 设置源语言（兼容新旧版 transformers）
            try:
                self._tokenizer.src_lang = self.src_lang
            except AttributeError:
                pass  # 部分 tokenizer 后端不支持直接设置 src_lang

            # 编码输入
            inputs = self._tokenizer(
                truncated,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )

            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # 获取目标语言 token ID（兼容新旧版 transformers）
            if hasattr(self._tokenizer, 'lang_code_to_id'):
                forced_bos_token_id = self._tokenizer.lang_code_to_id[self.tgt_lang]
            else:
                forced_bos_token_id = self._tokenizer.convert_tokens_to_ids(self.tgt_lang)

            # 生成翻译
            with torch.inference_mode():
                outputs = self._model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_new_tokens=128,
                    num_beams=self.num_beams,
                    do_sample=False
                )

            # 解码输出
            translated = self._tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

        except Exception as e:
            logger.error(f"翻译错误: {e}")
            translated = ""

        t_end = time.time()

        # 更新缓存
        if translated:
            self._cache.put(normalized, translated)

        logger.debug(f"MT: '{text[:30]}...' -> '{translated[:30]}...' | {(t_end-t_start)*1000:.0f}ms")

        return MTResult(
            source=normalized,
            target=translated,
            t_mt_start=t_start,
            t_mt_end=t_end
        )

    def _handle_error(self, error: Exception) -> None:
        """错误处理"""
        logger.error(f"MT 错误: {error}")
        self._error_count += 1

        if self._error_count >= self._max_errors:
            logger.warning(f"MT 连续错误 {self._max_errors} 次，尝试重启...")
            self._restart()

    def _restart(self) -> None:
        """重启模型"""
        try:
            logger.info("重新加载 MT 模型...")
            self.load_model()
            self._error_count = 0
        except Exception as e:
            logger.error(f"重启失败: {e}")
            self._running = False

    def stop(self) -> None:
        """停止工作线程"""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    @property
    def is_running(self) -> bool:
        """是否正在运行"""
        return self._running and self._thread is not None and self._thread.is_alive()

    @property
    def cache_size_used(self) -> int:
        """缓存使用量"""
        return len(self._cache)

