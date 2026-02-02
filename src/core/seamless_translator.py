"""
SeamlessM4T v2 语音翻译模块
端到端语音识别 + 翻译
"""
import torch
import torchaudio
import numpy as np
from typing import Optional, Tuple
from transformers import AutoProcessor, SeamlessM4Tv2Model

# SeamlessM4T 语言代码映射
LANGUAGE_CODES = {
    "auto": None,
    "zh": "cmn",
    "en": "eng",
    "ja": "jpn",
    "ko": "kor",
    "es": "spa",
    "fr": "fra",
    "de": "deu",
    "ru": "rus",
    "ar": "arb",
    "pt": "por",
    "it": "ita",
    "vi": "vie",
    "th": "tha",
    "id": "ind",
    "ms": "zsm",
    "hi": "hin",
    "tr": "tur",
    "pl": "pol",
    "nl": "nld",
    "sv": "swe",
    "da": "dan",
    "fi": "fin",
    "no": "nob",
    "cs": "ces",
    "el": "ell",
    "he": "heb",
    "uk": "ukr",
    "ro": "ron",
    "hu": "hun",
}

# 目标语言代码
TARGET_LANGUAGE_CODES = {
    "zh-CN": "cmn",
    "zh-TW": "cmn",
    "en": "eng",
    "ja": "jpn",
    "ko": "kor",
    "es": "spa",
    "fr": "fra",
    "de": "deu",
    "ru": "rus",
}


class SeamlessTranslator:
    """
    SeamlessM4T v2 语音翻译器
    支持语音到文本翻译（S2TT）
    """

    def __init__(self,
                 model_name: str = "facebook/seamless-m4t-v2-large",
                 device: str = "cuda",
                 use_8bit: bool = True):
        """
        初始化翻译器

        Args:
            model_name: HuggingFace 模型名称
            device: 运行设备 ("cuda" 或 "cpu")
            use_8bit: 是否使用 8-bit 量化（8GB 显存建议开启）
        """
        self.device = device
        self.use_8bit = use_8bit
        self.model_name = model_name

        self.processor = None
        self.model = None
        self.sample_rate = 16000

        self._loaded = False

    def load_model(self, progress_callback=None):
        """
        加载模型

        Args:
            progress_callback: 进度回调函数，接收 (message: str) 参数
        """
        if self._loaded:
            return

        if progress_callback:
            progress_callback("正在加载 SeamlessM4T 处理器...")

        self.processor = AutoProcessor.from_pretrained(self.model_name)

        if progress_callback:
            if self.use_8bit:
                progress_callback("正在加载 SeamlessM4T 模型（8-bit 量化模式）...")
            else:
                progress_callback("正在加载 SeamlessM4T 模型...")

        if self.use_8bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=False
            )

            self.model = SeamlessM4Tv2Model.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            self.model = SeamlessM4Tv2Model.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto" if self.device == "cuda" else None
            )

            if self.device == "cuda" and not self.use_8bit:
                self.model = self.model.to(self.device)

        self.model.eval()
        self._loaded = True

        if progress_callback:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                progress_callback(f"模型加载完成！显存占用: {allocated:.1f} GB")

    def _ensure_sample_rate(self, audio: np.ndarray,
                            original_sr: int) -> np.ndarray:
        """确保音频采样率为 16kHz"""
        if original_sr != self.sample_rate:
            audio_tensor = torch.from_numpy(audio).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            resampled = torchaudio.functional.resample(
                audio_tensor,
                orig_freq=original_sr,
                new_freq=self.sample_rate
            )
            return resampled.squeeze().numpy()
        return audio

    def translate(self,
                  audio: np.ndarray,
                  source_lang: str = "auto",
                  target_lang: str = "zh-CN",
                  original_sr: int = 16000) -> Tuple[str, str]:
        """
        翻译音频

        Args:
            audio: 音频数据 (numpy array, float32, mono)
            source_lang: 源语言代码
            target_lang: 目标语言代码
            original_sr: 音频原始采样率

        Returns:
            (original_text, translated_text) 元组
        """
        if not self._loaded:
            raise RuntimeError("模型尚未加载，请先调用 load_model()")

        audio = self._ensure_sample_rate(audio, original_sr)

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        src_lang_code = LANGUAGE_CODES.get(source_lang, source_lang)
        tgt_lang_code = TARGET_LANGUAGE_CODES.get(target_lang, target_lang)

        audio_inputs = self.processor(
            audio=audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )

        audio_inputs = {k: v.to(self.device) for k, v in audio_inputs.items()}

        with torch.no_grad():
            output_tokens = self.model.generate(
                **audio_inputs,
                tgt_lang=tgt_lang_code,
                generate_speech=False,
                num_beams=1,
                do_sample=False,
            )

            translated_text = self.processor.batch_decode(
                output_tokens[0],
                skip_special_tokens=True
            )[0]

            original_text = ""
            if src_lang_code and src_lang_code != tgt_lang_code:
                try:
                    asr_tokens = self.model.generate(
                        **audio_inputs,
                        tgt_lang=src_lang_code if src_lang_code else "eng",
                        generate_speech=False,
                        num_beams=1,
                        do_sample=False,
                    )
                    original_text = self.processor.batch_decode(
                        asr_tokens[0],
                        skip_special_tokens=True
                    )[0]
                except:
                    original_text = "[原文获取失败]"

        return original_text, translated_text

    def translate_speech_to_text(self,
                                  audio: np.ndarray,
                                  target_lang: str = "zh-CN",
                                  original_sr: int = 16000) -> str:
        """
        简化版翻译，只返回翻译结果
        """
        _, translated = self.translate(audio, "auto", target_lang, original_sr)
        return translated

    def transcribe(self,
                   audio: np.ndarray,
                   language: str = "en",
                   original_sr: int = 16000) -> str:
        """
        语音识别（ASR），不翻译
        """
        if not self._loaded:
            raise RuntimeError("模型尚未加载")

        audio = self._ensure_sample_rate(audio, original_sr)

        lang_code = LANGUAGE_CODES.get(language, language)

        audio_inputs = self.processor(
            audio=audio.astype(np.float32),
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )
        audio_inputs = {k: v.to(self.device) for k, v in audio_inputs.items()}

        with torch.no_grad():
            output_tokens = self.model.generate(
                **audio_inputs,
                tgt_lang=lang_code,
                generate_speech=False,
                num_beams=1,
            )
            text = self.processor.batch_decode(
                output_tokens[0],
                skip_special_tokens=True
            )[0]

        return text

    @property
    def is_loaded(self) -> bool:
        """模型是否已加载"""
        return self._loaded

    def unload(self):
        """卸载模型释放显存"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
