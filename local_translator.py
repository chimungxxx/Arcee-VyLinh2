# local_translator.py
"""
Local translator loader adapted for sail/Sailor-4B with robust handling of missing local weights.
Key fixes:
 - Check whether local model dir actually contains model weight files before trying to load from it.
 - If weights are absent, load from HF model name (attempt quantized 4-bit if requested and possible).
 - Provide clearer errors and guidance on next steps.
"""

import os
import shutil
import glob
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


def model_weights_exist(local_dir: str) -> bool:
    """Return True if the local_dir contains known model weight files."""
    if not os.path.isdir(local_dir):
        return False
    # common patterns for HF model weights
    patterns = [
        "pytorch_model.bin",
        "pytorch_model-*.bin",
        "*.safetensors",
        "tf_model.h5",
        "flax_model.msgpack",
        "model.ckpt",
        "pytorch_model.pt",
    ]
    for pat in patterns:
        matches = glob.glob(os.path.join(local_dir, pat))
        if matches:
            return True
    # also check for shard style files e.g., pytorch_model-00001-of-00002.bin
    shard_matches = glob.glob(os.path.join(local_dir, "pytorch_model-*of-*"))
    if shard_matches:
        return True
    return False


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
        # local folder name sanitized
        self.LOCAL_MODEL_DIR = os.path.join(self.local_base, self.HF_MODEL_NAME.replace("/", "_"))

        if device is None:
            self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.DEVICE = device

        self.prefer_4bit = prefer_4bit

        # ensure tokenizer saved locally (fast tokenizer if available)
        self.ensure_tokenizer_local()

        # load config (prefer local config if present)
        try:
            if model_weights_exist(self.LOCAL_MODEL_DIR):
                self.config = AutoConfig.from_pretrained(self.LOCAL_MODEL_DIR)
            else:
                self.config = AutoConfig.from_pretrained(self.HF_MODEL_NAME)
        except Exception:
            # best-effort fallback
            self.config = AutoConfig.from_pretrained(self.HF_MODEL_NAME)

        self.model_type = getattr(self.config, "model_type", "").lower() or "causal"
        self.tokenizer = safe_auto_tokenizer(self.LOCAL_MODEL_DIR, use_fast_prefer=True)

        # now load model (quantized if possible)
        self.model = self.load_model()

    def ensure_tokenizer_local(self):
        """
        Ensure tokenizer files are present in LOCAL_MODEL_DIR. If not, attempt to download tokenizer only.
        This keeps us from forcing a full-model download/save on disk when the target is constrained.
        """
        if os.path.isdir(self.LOCAL_MODEL_DIR) and os.listdir(self.LOCAL_MODEL_DIR):
            # If fast tokenizer present, we're good
            try:
                AutoTokenizer.from_pretrained(self.LOCAL_MODEL_DIR, use_fast=True)
                return
            except Exception:
                # leave the dir, we'll try to (re)save tokenizer
                pass

        # Download tokenizer (fast preferred) and save it locally. This will not download weights.
        tok = safe_auto_tokenizer(self.HF_MODEL_NAME, use_fast_prefer=True)
        os.makedirs(self.LOCAL_MODEL_DIR, exist_ok=True)
        tok.save_pretrained(self.LOCAL_MODEL_DIR)
        # do not save weights here (could be huge) — weights handled in load_model()

    def load_model(self):
        """
        Load the model. Priority:
          1) If CUDA available and bitsandbytes installed & prefer_4bit: load quantized 4-bit from HF directly.
          2) Else if local weights exist: load from local folder.
          3) Else: download from HF (non-quantized) and load.
        """
        use_cuda = (self.DEVICE == "cuda")
        # 1) Try quantized 4-bit load from HF name if CUDA+bnb available & preferred
        if use_cuda and self.prefer_4bit and _HAS_BNB:
            try:
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    quant_type="nf4"
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.HF_MODEL_NAME,
                    quantization_config=bnb_cfg,
                    device_map="auto",
                    trust_remote_code=True,
                )
                return model
            except Exception as e:
                # Quantized load failed; continue to other options
                print("[Translator] 4-bit quantized load failed (continuing fallback). Error:", e)

        # 2) If local weights exist, try to load from local dir
        if model_weights_exist(self.LOCAL_MODEL_DIR):
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self.LOCAL_MODEL_DIR,
                    device_map="auto" if use_cuda else None,
                    trust_remote_code=True,
                )
                # ensure on CPU if requested
                if not use_cuda:
                    model.to("cpu")
                return model
            except Exception as e:
                # If loading from local folder fails, show a helpful message and fall back to HF download
                print(f"[Translator] Loading from local folder {self.LOCAL_MODEL_DIR} failed: {e}. Falling back to hub download.")

        # 3) Download weights from HF hub (may be large)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.HF_MODEL_NAME,
                device_map="auto" if use_cuda else None,
                trust_remote_code=True,
            )
            # Optionally save downloaded weights for future runs (comment/uncomment as desired)
            # saving huge weights may be undesired in constrained systems; we skip automatic save.
            return model
        except Exception as e:
            # final helpful error: explain options
            raise RuntimeError(
                f"Failed to load model weights for {self.HF_MODEL_NAME}. "
                "Possible reasons:\n"
                " - You're offline and the local folder doesn't contain model weights.\n"
                " - The model is large and the environment ran out of memory during load.\n"
                " - Quantized load was requested but bitsandbytes / CUDA isn't available or failed.\n\n"
                "Suggested fixes:\n"
                " 1) If you have a GPU and want lower memory use, install bitsandbytes & accelerate and enable 4-bit quantization:\n"
                "      pip install bitsandbytes accelerate\n"
                " 2) If you cannot load the full weights on the target machine, pre-download the model on another machine\n"
                "    (where you can install sentencepiece / bitsandbytes if needed) and copy the entire model folder\n"
                "    into ./local_translator_models/<model_name>/ so this machine can load from local files.\n"
                " 3) Use a smaller model (e.g., a 1.8B or 2.7B variant) if you only have CPU or limited RAM.\n\n"
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
        # run on model device
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.model.device)
        generate_params = dict(max_new_tokens=max_new_tokens, do_sample=False, temperature=0.2)
        generate_params.update(gen_kwargs)
        outputs = self.model.generate(**inputs, **generate_params)
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Vietnamese:" in decoded:
            return decoded.split("Vietnamese:", 1)[1].strip()
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
