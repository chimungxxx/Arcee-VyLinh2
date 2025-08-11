# local_translator.py
"""
Local translator loader (Sailor-0.5B only) with detailed server logging.
- Model: sail/Sailor-0.5B (hardcoded default)
- Tries to prefer fast tokenizer (avoid sentencepiece)
- Attempts 4-bit quantized load if CUDA + bitsandbytes present (reduces VRAM)
- Logs steps and decisions to stdout via logging module (server logs)
"""

import os
import shutil
import glob
import logging
from typing import Tuple, Union
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# BitsAndBytesConfig may not exist if bitsandbytes not installed
try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    BitsAndBytesConfig = None
    _HAS_BNB = False

# Configure logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Basic configuration - server collects stdout/stderr
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

MODEL_ID = "sail/Sailor-0.5B"  # <- fixed to Sailor 1.8B


def safe_auto_tokenizer(pretrained_or_local: str, use_fast_prefer: bool = True, **kwargs) -> AutoTokenizer:
    """
    Try fast tokenizer first (no sentencepiece). If that fails, try slow tokenizer and raise
    a helpful RuntimeError if sentencepiece is required.
    """
    last_exc = None
    if use_fast_prefer:
        try:
            logger.info("Attempting to load fast tokenizer for '%s' (use_fast=True)...", pretrained_or_local)
            tok = AutoTokenizer.from_pretrained(pretrained_or_local, use_fast=True, **kwargs)
            logger.info("Loaded fast tokenizer successfully for '%s'.", pretrained_or_local)
            return tok
        except Exception as e:
            last_exc = e
            logger.debug("Fast tokenizer load failed: %s", e)

    try:
        logger.info("Attempting to load slow tokenizer for '%s' (use_fast=False)...", pretrained_or_local)
        tok = AutoTokenizer.from_pretrained(pretrained_or_local, use_fast=False, **kwargs)
        logger.info("Loaded slow tokenizer successfully for '%s'.", pretrained_or_local)
        return tok
    except Exception as e:
        logger.error("Failed to load tokenizer for '%s'. Original error: %s", pretrained_or_local, e)
        msg = (
            f"Failed to load tokenizer for '{pretrained_or_local}'. Likely missing 'sentencepiece' for slow tokenizer.\n\n"
            "Options:\n"
            " 1) Install sentencepiece (conda recommended): `conda install -c conda-forge sentencepiece`\n"
            " 2) Pre-download tokenizer files (including tokenizer.json) on another machine and copy to the local model folder.\n"
            " 3) Use a model that ships a fast tokenizer.\n\n"
            f"Original error: {e}"
        )
        raise RuntimeError(msg) from last_exc


def model_weights_exist(local_dir: str) -> bool:
    """Return True if the local_dir contains known model weight files."""
    if not os.path.isdir(local_dir):
        return False
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
            logger.debug("Found weight files via pattern '%s' in %s: %s", pat, local_dir, matches[:3])
            return True
    # Check shard style e.g., pytorch_model-00001-of-00002.bin
    shard_matches = glob.glob(os.path.join(local_dir, "pytorch_model-*of-*"))
    if shard_matches:
        logger.debug("Found shard-style weight files in %s: %s", local_dir, shard_matches[:3])
        return True
    return False


class Translator:
    def __init__(
        self,
        hf_model_name: str = MODEL_ID,
        local_base: str = None,
        device: Union[str, None] = None,
        prefer_4bit: bool = True,
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

        logger.info("Translator init: model=%s, local_dir=%s, device=%s, prefer_4bit=%s",
                    self.HF_MODEL_NAME, self.LOCAL_MODEL_DIR, self.DEVICE, self.prefer_4bit)

        # Ensure tokenizer present locally (fast preferred)
        self.ensure_tokenizer_local()

        # Config: prefer local config if weights exist locally; otherwise ask hub
        try:
            if model_weights_exist(self.LOCAL_MODEL_DIR):
                logger.info("Local weights found in %s -> loading config from local dir", self.LOCAL_MODEL_DIR)
                self.config = AutoConfig.from_pretrained(self.LOCAL_MODEL_DIR)
            else:
                logger.info("No local weights found -> loading config from HF hub for %s", self.HF_MODEL_NAME)
                self.config = AutoConfig.from_pretrained(self.HF_MODEL_NAME)
        except Exception as e:
            logger.warning("Could not load config cleanly, attempting to fallback. Error: %s", e)
            self.config = AutoConfig.from_pretrained(self.HF_MODEL_NAME)

        self.model_type = getattr(self.config, "model_type", "").lower() or "causal"
        logger.info("Detected model_type='%s' (config.model_type=%s)", self.model_type, getattr(self.config, "model_type", None))

        # tokenizer loaded from local dir (safe_auto_tokenizer inside ensure_tokenizer_local saved it)
        self.tokenizer = safe_auto_tokenizer(self.LOCAL_MODEL_DIR, use_fast_prefer=True)

        # Load model (tries quantized -> local -> hub)
        self.model = self.load_model()
        logger.info("Model loaded, device: %s", next(self.model.parameters()).device if hasattr(self.model, "parameters") else "unknown")

    def ensure_tokenizer_local(self):
        """
        Ensure tokenizer saved in LOCAL_MODEL_DIR. If not present, download tokenizer only and save it there.
        """
        try:
            if os.path.isdir(self.LOCAL_MODEL_DIR) and os.listdir(self.LOCAL_MODEL_DIR):
                logger.info("Local model dir exists: %s. Checking tokenizer...", self.LOCAL_MODEL_DIR)
                try:
                    AutoTokenizer.from_pretrained(self.LOCAL_MODEL_DIR, use_fast=True)
                    logger.info("Fast tokenizer loaded from local dir successfully.")
                    return
                except Exception as e:
                    logger.info("Fast tokenizer not usable from local dir (will re-download tokenizer). Error: %s", e)
                    # We will re-download tokenizer below
            logger.info("Downloading tokenizer for '%s' (fast preferred). This will be saved to %s", self.HF_MODEL_NAME, self.LOCAL_MODEL_DIR)
            tok = safe_auto_tokenizer(self.HF_MODEL_NAME, use_fast_prefer=True)
            os.makedirs(self.LOCAL_MODEL_DIR, exist_ok=True)
            tok.save_pretrained(self.LOCAL_MODEL_DIR)
            logger.info("Tokenizer saved to local dir: %s", self.LOCAL_MODEL_DIR)
        except Exception as e:
            logger.error("Failed in ensure_tokenizer_local: %s", e)
            raise

    def load_model(self):
        """
        Load model with these fallbacks:
          1) If CUDA & prefer_4bit & bitsandbytes available: try quantized 4-bit load from HF (device_map='auto').
          2) If local weights present: load from local folder.
          3) Else: download from HF and load (non-quantized).
        """
        use_cuda = (self.DEVICE == "cuda")
        logger.info("Beginning model load. CUDA=%s, bitsandbytes_available=%s", use_cuda, _HAS_BNB)

        # 1) Try quantized 4-bit from hub if requested
        if use_cuda and self.prefer_4bit and _HAS_BNB:
            logger.info("Attempting 4-bit quantized load via bitsandbytes (this may take a bit)...")
            try:
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    quant_type="nf4"
                )
                logger.info("BitsAndBytesConfig prepared: %s", bnb_cfg)
                model = AutoModelForCausalLM.from_pretrained(
                    self.HF_MODEL_NAME,
                    quantization_config=bnb_cfg,
                    device_map="auto",
                    trust_remote_code=True,
                )
                logger.info("4-bit quantized load succeeded (from HF hub).")
                return model
            except Exception as e:
                logger.warning("4-bit quantized load failed: %s. Will try other fallbacks.", e)

        # 2) Try local weights if present
        if model_weights_exist(self.LOCAL_MODEL_DIR):
            logger.info("Detected local weight files in %s -- attempting to load from local folder.", self.LOCAL_MODEL_DIR)
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self.LOCAL_MODEL_DIR,
                    device_map="auto" if use_cuda else None,
                    trust_remote_code=True,
                )
                if not use_cuda:
                    logger.info("Moving model to CPU as CUDA not used.")
                    model.to("cpu")
                logger.info("Loaded model from local folder successfully.")
                return model
            except Exception as e:
                logger.warning("Loading from local folder failed: %s. Will attempt hub download.", e)

        # 3) Download (non-quantized) from HF
        logger.info("Attempting to download model weights from HF hub: %s (this can be large)", self.HF_MODEL_NAME)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.HF_MODEL_NAME,
                device_map="auto" if use_cuda else None,
                trust_remote_code=True,
            )
            logger.info("Downloaded and loaded model from HF hub successfully.")
            return model
        except Exception as e:
            logger.error("Failed to load model from HF hub: %s", e)
            raise RuntimeError(
                f"Failed to load model weights for {self.HF_MODEL_NAME}. "
                "Possible causes: offline environment, missing local weights, insufficient memory, or quantized load failed.\n\n"
                "Fixes:\n"
                " - If offline: pre-download model weights on another machine and copy the folder into ./local_translator_models/\n"
                " - If you have GPU and want lower memory usage: install bitsandbytes & accelerate and enable prefer_4bit=True\n"
                " - Use a smaller model if you only have CPU\n\n"
                f"Original error: {e}"
            ) from e

    def translate(self, text: str, max_new_tokens: int = 256, **gen_kwargs) -> str:
        """Translate Chinese -> Vietnamese using a causal prompt for Sailor (causal LM)."""
        if not text or not text.strip():
            return ""
        prompt = (
            "Translate the following Chinese text into fluent Vietnamese.\n"
            "Preserve named entities when appropriate.\n\n"
            f"Chinese: {text}\n\nVietnamese:"
        )
        logger.info("Generating translation for input length=%d", len(text))
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.model.device)
        generate_params = dict(max_new_tokens=max_new_tokens, do_sample=False, temperature=0.2)
        generate_params.update(gen_kwargs)
        outputs = self.model.generate(**inputs, **generate_params)
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Vietnamese:" in decoded:
            result = decoded.split("Vietnamese:", 1)[1].strip()
            logger.info("Generation complete (extracted after 'Vietnamese:').")
            return result
        if decoded.startswith(prompt):
            logger.info("Prompt was echoed; stripping prefix.")
            return decoded[len(prompt):].strip()
        logger.info("Generation complete.")
        return decoded.strip()


class TranslatorManager:
    def __init__(self, model_names=None, **kwargs):
        # Force single model usage (Sailor-0.5B) regardless of model_names parameter
        self.model_id = MODEL_ID
        logger.info("TranslatorManager init (forced model=%s)", self.model_id)
        self.translators = {}
        try:
            self.translators[self.model_id] = Translator(self.model_id, **kwargs)
        except Exception as e:
            logger.error("Error creating Translator for %s: %s", self.model_id, e)
            self.translators[self.model_id] = f"Error loading model: {e}"

    def translate_all(self, text: str):
        results = {}
        for name, translator in self.translators.items():
            if isinstance(translator, Translator):
                try:
                    results[name] = translator.translate(text)
                except Exception as e:
                    logger.error("Error during translate for %s: %s", name, e)
                    results[name] = f"Error during translate: {e}"
            else:
                results[name] = str(translator)
        return results
