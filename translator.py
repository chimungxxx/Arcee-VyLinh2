"""
 Translator utility that tries to avoid sentencepiece by preferring fast tokenizers.
- Downloads model+tokenizer once from HF Hub (if not already present) and saves into local folder.
- Loads model from the local folder afterwards.
- Uses AutoConfig to detect seq2seq vs causal models.
- If the tokenizer truly requires sentencepiece, shows a helpful message explaining options.
"""

import os
import shutil
from typing import Tuple, Union
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)

# Models that are usually seq2seq by config.model_type
SEQ2SEQ_TYPES = {"t5", "mt5", "mbart", "marian", "m2m_100", "fsmt", "pegasus"}


def safe_auto_tokenizer(pretrained_or_local: str, use_fast_prefer: bool = True, **kwargs) -> AutoTokenizer:
    """
    Try to load a fast tokenizer (tokenizer.json) first (which often avoids sentencepiece).
    If that fails, try the slow tokenizer (may need sentencepiece). If slow tokenizer fails,
    raise a helpful RuntimeError explaining options (install sentencepiece or pre-download tokenizer.json).
    """
    last_exc = None
    if use_fast_prefer:
        try:
            return AutoTokenizer.from_pretrained(pretrained_or_local, use_fast=True, **kwargs)
        except Exception as e:
            last_exc = e  # fall through to trying non-fast tokenizer

    try:
        return AutoTokenizer.from_pretrained(pretrained_or_local, use_fast=False, **kwargs)
    except Exception as e:
        # Provide actionable guidance rather than a long stacktrace
        msg = (
            "Failed to load tokenizer for '{}'.\n\n"
            "Likely the tokenizer requires the 'sentencepiece' package which is not installed in this environment.\n\n"
            "Options to resolve (pick one):\n"
            "1) Install sentencepiece (recommended if you can):\n"
            "   - conda:  conda install -c conda-forge sentencepiece\n"
            "   - pip:    pip install sentencepiece  (may fail on some Windows environments without build tools)\n\n"
            "2) Avoid installing sentencepiece by ensuring a FAST tokenizer is available:\n"
            "   - On a machine that can install sentencepiece or has full HF tooling, download the model and tokenizer files\n"
            "     and copy the model folder (including tokenizer.json or tokenizer files) into your target machine's\n"
            "     local models folder (e.g., ./local_translator_models/<model_name>/). After that, call\n"
            "     AutoTokenizer.from_pretrained(local_dir, use_fast=True) and it will not need sentencepiece.\n\n"
            "3) Use a different HF model that ships a fast tokenizer (tokenizer.json) and does not require sentencepiece.\n\n"
            "Original error: {}\n"
        ).format(pretrained_or_local, str(e))
        raise RuntimeError(msg) from last_exc


class Translator:
    """
    Loads a model from HF or local folder and provides translate(text) method.
    """

    def __init__(self, hf_model_name: str, local_base: str = None, device: Union[str, None] = None):
        """
        hf_model_name: HF repo id (e.g., "arcee-ai/Arcee-VyLinh") or absolute/local path
        local_base: parent folder for local caches (default: ./local_translator_models)
        device: "cpu", "cuda", or None (auto)
        """
        self.HF_MODEL_NAME = hf_model_name
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.local_base = local_base or os.path.join(self.script_dir, "local_translator_models")
        os.makedirs(self.local_base, exist_ok=True)
        self.LOCAL_MODEL_DIR = os.path.join(self.local_base, self.HF_MODEL_NAME.replace("/", "_"))

        if device is None:
            self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.DEVICE = torch.device(device)

        # Ensure model artifacts are on disk (download if necessary)
        self.ensure_model_local()

        # Load config and decide model_type
        try:
            self.config = AutoConfig.from_pretrained(self.LOCAL_MODEL_DIR)
        except Exception:
            # As fallback try to load from HF name (still OK)
            self.config = AutoConfig.from_pretrained(self.HF_MODEL_NAME)

        model_type = getattr(self.config, "model_type", "").lower()
        if model_type in SEQ2SEQ_TYPES:
            self.model_type = "seq2seq"
            self.tokenizer, self.model = self.load_seq2seq()
        else:
            # default to causal if not recognized
            self.model_type = "causal"
            self.tokenizer, self.model = self.load_causal()

    def ensure_model_local(self):
        """
        If local model dir exists and is usable, do nothing.
        Otherwise download tokenizer/model from HF Hub (public models) and save them locally.
        """
        # Quick check: if folder exists and has tokenizer files, try to load tokenizer from it
        if os.path.isdir(self.LOCAL_MODEL_DIR) and os.listdir(self.LOCAL_MODEL_DIR):
            try:
                # Try fast tokenizer first (no sentencepiece needed if tokenizers files present)
                AutoTokenizer.from_pretrained(self.LOCAL_MODEL_DIR, use_fast=True)
                return
            except Exception:
                # maybe partially downloaded or requires sentencepiece; remove and re-download
                shutil.rmtree(self.LOCAL_MODEL_DIR, ignore_errors=True)

        print(f"[Translator] Downloading model/tokenizer for '{self.HF_MODEL_NAME}' to '{self.LOCAL_MODEL_DIR}' ...")
        # Create parent dir then download
        os.makedirs(self.LOCAL_MODEL_DIR, exist_ok=True)

        # Tokenizer: try fast first using safe_auto_tokenizer; this will either return a tokenizer
        # or raise a helpful error if the model truly requires sentencepiece and it's not available.
        tokenizer = safe_auto_tokenizer(self.HF_MODEL_NAME, use_fast_prefer=True)

        # Decide which model class to use based on config
        try:
            cfg = AutoConfig.from_pretrained(self.HF_MODEL_NAME)
        except Exception:
            cfg = None

        # Attempt seq2seq load first if config suggests so; otherwise try causal as fallback
        model = None
        try:
            if cfg and getattr(cfg, "model_type", "").lower() in SEQ2SEQ_TYPES:
                model = AutoModelForSeq2SeqLM.from_pretrained(self.HF_MODEL_NAME)
            else:
                # Try seq2seq first (many translation models are seq2seq). If it fails, fallback to causal.
                try:
                    model = AutoModelForSeq2SeqLM.from_pretrained(self.HF_MODEL_NAME)
                except Exception:
                    model = AutoModelForCausalLM.from_pretrained(self.HF_MODEL_NAME)
        except Exception as e:
            # Re-raise with context (tokenizer loaded succeeded; model load failed)
            raise RuntimeError(f"Failed to download/load model weights for '{self.HF_MODEL_NAME}': {e}")

        # Save both tokenizer and model to the local folder for repeatable loads
        tokenizer.save_pretrained(self.LOCAL_MODEL_DIR)
        model.save_pretrained(self.LOCAL_MODEL_DIR)
        # Save the config back too
        try:
            cfg = model.config
            cfg.save_pretrained(self.LOCAL_MODEL_DIR)
        except Exception:
            pass

        print(f"[Translator] Saved model+tokenizer to {self.LOCAL_MODEL_DIR}")

    def load_seq2seq(self) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
        tokenizer = safe_auto_tokenizer(self.LOCAL_MODEL_DIR, use_fast_prefer=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.LOCAL_MODEL_DIR)
        model.to(self.DEVICE)
        return tokenizer, model

    def load_causal(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        tokenizer = safe_auto_tokenizer(self.LOCAL_MODEL_DIR, use_fast_prefer=True)
        model = AutoModelForCausalLM.from_pretrained(self.LOCAL_MODEL_DIR)
        model.to(self.DEVICE)
        return tokenizer, model

    def translate(self, text: str, max_length: int = 256, **generate_kwargs) -> str:
        """
        Translate the provided text. For seq2seq models we pass the text as-is.
        For causal models we construct a short instruction prompt.
        """
        if not text or not text.strip():
            return ""
        if self.model_type == "seq2seq":
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=1024
            ).to(self.DEVICE)
            gen_k = dict(max_length=max_length, num_beams=4, early_stopping=True)
            gen_k.update(generate_kwargs)
            outputs = self.model.generate(**inputs, **gen_k)
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            return decoded.strip()
        else:
            # causal model prompt (simple)
            prompt = (
                "Translate the following Chinese text into natural Vietnamese. "
                "Keep names as-is where appropriate.\n\n"
                f"Chinese: {text}\n\nVietnamese:"
            )
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.DEVICE)
            gen_k = dict(max_new_tokens=max_length, do_sample=False, temperature=0.2)
            gen_k.update(generate_kwargs)
            outputs = self.model.generate(**inputs, **gen_k)
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Vietnamese:" in decoded:
                return decoded.split("Vietnamese:", 1)[1].strip()
            return decoded.strip()


class TranslatorManager:
    """
    Manage multiple translators (useful for comparing models).
    """
    def __init__(self, model_names, local_base: str = None, device: Union[str, None] = None):
        self.translators = {}
        for name in model_names:
            try:
                self.translators[name] = Translator(name, local_base=local_base, device=device)
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
