"""Arabic-to-multiple-languages translation engine (local CPU/GPU inference).

Save this file as ``translator_core.py`` in the project directory.
Models are downloaded from Hugging Face on first use and cached on disk.
Only one model is retained in RAM by default.
"""

from __future__ import annotations

import gc
import os
import threading
from copy import deepcopy
from typing import Dict, List, Optional

# Set Hub timeouts before importing transformers / huggingface_hub.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "20")

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

SUPPORTED_LANGUAGES: Dict[str, Dict[str, str]] = {
    "en": {"name": "English 🇺🇸", "model": "Helsinki-NLP/opus-mt-tc-big-ar-en", "description": "Arabic → English"},
    "de": {"name": "German 🇩🇪", "model": "Helsinki-NLP/opus-mt-ar-de", "description": "Arabic → German"},
    "fr": {"name": "French 🇫🇷", "model": "Helsinki-NLP/opus-mt-ar-fr", "description": "Arabic → French"},
    "es": {"name": "Spanish 🇪🇸", "model": "Helsinki-NLP/opus-mt-ar-es", "description": "Arabic → Spanish"},
    "it": {"name": "Italian 🇮🇹", "model": "Helsinki-NLP/opus-mt-ar-it", "description": "Arabic → Italian"},
    "ru": {"name": "Russian 🇷🇺", "model": "Helsinki-NLP/opus-mt-ar-ru", "description": "Arabic → Russian"},
    "tr": {"name": "Turkish 🇹🇷", "model": "Helsinki-NLP/opus-mt-ar-tr", "description": "Arabic → Turkish"},
    "pl": {"name": "Polish 🇵🇱", "model": "Helsinki-NLP/opus-mt-ar-pl", "description": "Arabic → Polish"},
    "he": {"name": "Hebrew 🇮🇱", "model": "Helsinki-NLP/opus-mt-ar-he", "description": "Arabic → Hebrew"},
    "el": {"name": "Greek 🇬🇷", "model": "Helsinki-NLP/opus-mt-ar-el", "description": "Arabic → Greek"},
}


class ArabicTranslator:
    """Model-on-demand translator. Keeps only one language model in RAM.

    A lock protects the cached model because Streamlit may share its cached
    translator instance across sessions and PyTorch inference isn't guarded
    against concurrent eviction otherwise.
    """

    def __init__(self, device: Optional[str] = None):
        resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if resolved not in {"cuda", "cpu"}:
            raise ValueError("device must be 'cpu', 'cuda', or None")
        if resolved == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but isn't available")
        self.device = resolved
        self._cache: Dict[str, tuple] = {}
        self._lock = threading.RLock()
        print(f"🔧 Translator initialized on device: {self.device}")

    def _clear_cache_unlocked(self) -> None:
        self._cache.clear()
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _load_model(self, lang_code: str):
        """Load a model; caller must hold self._lock when sharing an instance."""
        if lang_code not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {lang_code}")
        if lang_code in self._cache:
            return self._cache[lang_code]

        # Evict the previous model before loading the next one to limit RAM.
        self._clear_cache_unlocked()
        model_id = SUPPORTED_LANGUAGES[lang_code]["model"]
        print(f"📥 Loading model for {SUPPORTED_LANGUAGES[lang_code]['name']} ...", flush=True)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            model = model.to(self.device)
            model.eval()
        except Exception:
            # No partially loaded model is retained in the instance.
            gc.collect()
            raise
        self._cache[lang_code] = (tokenizer, model)
        print(f"✅ Model loaded: {model_id}", flush=True)
        return tokenizer, model

    def translate(
        self,
        text: str,
        target_lang: str,
        max_length: int = 256,
        num_beams: int = 2,
        early_stopping: bool = True,
    ) -> str:
        """Translate Arabic text; `max_length` bounds input and output tokens.

        `early_stopping` remains in the signature for backwards compatibility,
        but isn't passed for greedy decoding (num_beams=1).
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Please enter a non-empty Arabic text.")
        if not isinstance(max_length, int) or not 8 <= max_length <= 1024:
            raise ValueError("max_length must be an integer between 8 and 1024")
        if not isinstance(num_beams, int) or not 1 <= num_beams <= 10:
            raise ValueError("num_beams must be an integer between 1 and 10")

        with self._lock:
            tokenizer, model = self._load_model(target_lang)
            tokenized = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(self.device)
            gen_args = {"max_new_tokens": max_length, "num_beams": num_beams}
            with torch.inference_mode():
                output_ids = model.generate(**tokenized, **gen_args)
            return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def translate_batch(
        self,
        texts: List[str],
        target_lang: str,
        max_length: int = 256,
        num_beams: int = 2,
        early_stopping: bool = True,
    ) -> List[str]:
        if not isinstance(texts, list) or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("texts must be a list of non-empty strings")
        if not texts:
            return []
        if not isinstance(max_length, int) or not 8 <= max_length <= 1024:
            raise ValueError("max_length must be an integer between 8 and 1024")
        if not isinstance(num_beams, int) or not 1 <= num_beams <= 10:
            raise ValueError("num_beams must be an integer between 1 and 10")
        with self._lock:
            tokenizer, model = self._load_model(target_lang)
            tokenized = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
            ).to(self.device)
            gen_args = {"max_new_tokens": max_length, "num_beams": num_beams}
            with torch.inference_mode():
                output_ids = model.generate(**tokenized, **gen_args)
            return tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def get_supported_languages(self) -> Dict[str, Dict[str, str]]:
        return deepcopy(SUPPORTED_LANGUAGES)

    def unload_model(self, lang_code: str) -> None:
        with self._lock:
            if lang_code in self._cache:
                self._clear_cache_unlocked()

    def unload_all(self) -> None:
        with self._lock:
            self._clear_cache_unlocked()
