import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

class Translator:
    def __init__(self, hf_model_name):
        self.HF_MODEL_NAME = hf_model_name
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.LOCAL_MODEL_DIR = os.path.join(
            self.script_dir,
            "local_translator_models",
            self.HF_MODEL_NAME.replace('/', '_')
        )
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.LOCAL_MODEL_DIR, exist_ok=True)
        self.ensure_model_local()

        self.config = AutoConfig.from_pretrained(self.LOCAL_MODEL_DIR)
        self.tokenizer, self.model = self.load_causal()

    def ensure_model_local(self):
        if os.path.exists(self.LOCAL_MODEL_DIR) and os.listdir(self.LOCAL_MODEL_DIR):
            try:
                AutoTokenizer.from_pretrained(self.LOCAL_MODEL_DIR)
                print("[DEBUG] Model already cached locally.")
                return
            except Exception:
                import shutil
                shutil.rmtree(self.LOCAL_MODEL_DIR, ignore_errors=True)

        print(f"[DEBUG] Downloading model: {self.HF_MODEL_NAME} ...")
        tokenizer = AutoTokenizer.from_pretrained(self.HF_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(self.HF_MODEL_NAME)
        tokenizer.save_pretrained(self.LOCAL_MODEL_DIR)
        model.save_pretrained(self.LOCAL_MODEL_DIR)
        print("[DEBUG] Model download completed.")

    def load_causal(self):
        print("[DEBUG] Loading model to device:", self.DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(self.LOCAL_MODEL_DIR)
        model = AutoModelForCausalLM.from_pretrained(self.LOCAL_MODEL_DIR).to(self.DEVICE)
        return tokenizer, model

    def translate(self, text, max_length=128):
        prompt = f"Translate the following Chinese text to Vietnamese:\n{text}\nVietnamese:"
        print(f"[DEBUG] Prompt: {prompt}")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.DEVICE)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=False
        )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"[DEBUG] Raw model output: {decoded}")

        # Extract translation
        if "Vietnamese:" in decoded:
            translation = decoded.split("Vietnamese:")[-1].strip()
        else:
            translation = decoded.strip()

        print(f"[DEBUG] Final translation: {translation}")
        return translation


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Chinese → Vietnamese Translator", layout="centered")
st.title("Chinese → Vietnamese Translator (Sailor-0.5B)")

translator = Translator("sail/Sailor-0.5B")

text_input = st.text_area("Enter Chinese text here:")
if st.button("Translate"):
    if text_input.strip():
        with st.spinner("Translating..."):
            result = translator.translate(text_input)
        st.subheader("Vietnamese Translation")
        st.write(result)
    else:
        st.warning("Please enter some Chinese text.")
