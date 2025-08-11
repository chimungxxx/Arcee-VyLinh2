# local_translator.py
"""
Local translator loader adapted for sail/Sailor-4B with optional 4-bit quantization.
- Prefers fast tokenizer (use_fast=True) to avoid sentencepiece when possible.
- If CUDA+bitsandbytes available, attempts to load model in 4-bit (very memory efficient).
- If quantization not available, falls back to normal load (may be large).
"""

import os
import shutil
from typing import Tuple, Union
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
)
# BitsAndBytesConfig may not be available until bitsandbytes / transformers are recent:
try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    BitsAndBytesConfig = None
    _HAS_BNB = False

SEQ2SEQ_TYPES = {"t5", "mt5", "mbart", "marian", "m2m_100", "fsmt", "pegasus"}


def safe_auto_tokenizer(pretrained_or_local: str, use_fast_prefer: bool = True, **kwargs) -> AutoTokenizer:
    """
    Try fast tokenizer first (no sentencepiece). If that fails, try slow tokenizer and give
    an actionable error if sentencepiece is required.
    """
    last_exc = None
    if use_fast_prefer:
        try:
            return AutoTokenizer.from_pretrained(pretrained_or_local, use_fast=True, **kwargs)
        except Exception as e:
            last_exc = e
    try:
        return AutoTokenizer.from_pretrained(pretrained_or_local, use_fast=False, **kwargs)
    except Exception as e:
        raise RuntimeError(
            "Failed to load tokenizer. Likely missing 'sentencepiece' for this model's slow tokenizer.\n\n"
            "Options:\n"
            " 1) Install sentencepiece (conda recommended): `conda install -c conda-forge sentencepiece`\n"
            " 2) Ensure a fast tokenizer (tokenizer.json) exists in the model folder (you can download it from the HF repo)\n"
            " 3) Use another model that ships a fast tokenizer.\n\n"
            f"Original error: {e}"
        ) from last_exc


class Translator:
    def __init__(
        self,
        hf_model_name: str = "sail/Sailor-4B",
        local_base: str = None,
        device: Union[str, None] = None,
        prefer_4bit: bool = True,   # try 4-bit quantization if possible
    ):
        self.HF_MODEL_NAME = hf_model_name
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.local_base = local_base or os.path.join(self.script_dir, "local_translator_models")
        os.makedirs(self.local_base, exist_ok=True)
        self.LOCAL_MODEL_DIR = os.path.join(self.local_base, self.HF_MODEL_NAME.replace("/", "_"))

        if device is None:
            self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.DEVICE = device

        self.prefer_4bit = prefer_4bit
        # ensure tokenizer saved locally (fast tokenizer if available)
        self.ensure_tokenizer_local()

        # load config (from local if saved; else from HF)
        try:
            self.config = AutoConfig.from_pretrained(self.LOCAL_MODEL_DIR)
        except Exception:
            self.config = AutoConfig.from_pretrained(self.HF_MODEL_NAME)

        self.model_type = getattr(self.config, "model_type", "").lower() or "causal"
        # Sailor is a causal-style LM derived from Qwen; we'll treat as causal.
        self.tokenizer = safe_auto_tokenizer(self.LOCAL_MODEL_DIR, use_fast_prefer=True)

        # now load model (quantized if possible)
        self.model = self.load_model()

    def ensure_tokenizer_local(self):
        """
        Ensure tokenizer files are present in LOCAL_MODEL_DIR. If not, attempt to download tokenizer only.
        This keeps us from forcing a full-model download / save on disk.
        """
        if os.path.isdir(self.LOCAL_MODEL_DIR) and os.listdir(self.LOCAL_MODEL_DIR):
            # If tokenizer.json present, assume fast tokenizer works
            try:
                AutoTokenizer.from_pretrained(self.LOCAL_MODEL_DIR, use_fast=True)
                return
            except Exception:
                # try to remove if partial or broken and redownload
                shutil.rmtree(self.LOCAL_MODEL_DIR, ignore_errors=True)

        # Download fast tokenizer from HF and save tokenizer files locally.
        tok = safe_auto_tokenizer(self.HF_MODEL_NAME, use_fast_prefer=True)
        os.makedirs(self.LOCAL_MODEL_DIR, exist_ok=True)
        tok.save_pretrained(self.LOCAL_MODEL_DIR)
        # don't download + save full weights here (large). We'll load via from_pretrained below,
        # possibly in quantized mode which may not be saved back to disk.

    def load_model(self):
        """
        Load the model. Priority:
          1) If CUDA available and bitsandbytes installed & prefer_4bit: load quantized 4-bit with device_map="auto".
          2) Else attempt standard from_pretrained to CPU/GPU depending on availability.
        """
        use_cuda = (self.DEVICE == "cuda")
        # Try quantized 4-bit load if possible
        if use_cuda and self.prefer_4bit and _HAS_BNB:
            try:
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    quant_type="nf4"  # NF4 often gives good results on LLMs
                )
                # device_map "auto" will place layers on GPU(s). This requires accelerate & bitsandbytes config supported.
                model = AutoModelForCausalLM.from_pretrained(
                    self.HF_MODEL_NAME,
                    quantization_config=bnb_cfg,
                    device_map="auto",
                    trust_remote_code=True,  # some models may require it
                )
                return model
            except Exception as e:
                # fallback and inform (do not crash here — try non-quantized)
                print("[Translator] 4-bit quantized load failed, falling back to normal load. Error:", e)

        # If quantized load unavailable, attempt normal load. This may be expensive on CPU.
        try:
            # If user previously saved/checkpoint local model files, try loading from local dir first
            if os.path.isdir(self.LOCAL_MODEL_DIR) and os.listdir(self.LOCAL_MODEL_DIR):
                model = AutoModelForCausalLM.from_pretrained(self.LOCAL_MODEL_DIR, device_map="auto" if use_cuda else None)
            else:
                model = AutoModelForCausalLM.from_pretrained(self.HF_MODEL_NAME, device_map="auto" if use_cuda else None)
            # If device_map didn't place model automatically and we are on single GPU, move to device
            if not use_cuda:
                # ensure model on CPU
                model.to("cpu")
            return model
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model weights for {self.HF_MODEL_NAME}. "
                "If you intended to use quantized mode, ensure 'bitsandbytes' is installed and CUDA is available. "
                f"Original error: {e}"
            ) from e

    def translate(self, text: str, max_new_tokens: int = 256, **gen_kwargs) -> str:
        """Basic translate pipeline using a causal prompt (Sailor is causal-style)."""
        if not text or not text.strip():
            return ""
        prompt = (
            "Translate the following Chinese text into fluent Vietnamese.\n"
            "Preserve named entities when appropriate.\n\n"
            f"Chinese: {text}\n\nVietnamese:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.model.device)
        generate_params = dict(max_new_tokens=max_new_tokens, do_sample=False, temperature=0.2)
        generate_params.update(gen_kwargs)
        outputs = self.model.generate(**inputs, **generate_params)
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # try to extract text after marker
        if "Vietnamese:" in decoded:
            return decoded.split("Vietnamese:", 1)[1].strip()
        # fallback: remove prompt prefix if present
        if decoded.startswith(prompt):
            return decoded[len(prompt):].strip()
        return decoded.strip()


class TranslatorManager:
    def __init__(self, model_names, **kwargs):
        self.translators = {}
        for name in model_names:
            try:
                self.translators[name] = Translator(name, **kwargs)
            except Exception as e:
                self.translators[name] = f"Error loading model: {e}"

    def translate_all(self, text: str):
        results = {}
        for name, translator in self.translators.items():
            if isinstance(translator, Translator):
                try:
                    results[name] = translator.translate(text)
                except Exception as e:
                    results[name] = f"Error during translate: {e}"
            else:
                results[name] = str(translator)
        return results
