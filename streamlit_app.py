# app.py
import streamlit as st
from local_translator import Translator, TranslatorManager
import torch

st.set_page_config(page_title="Local Chinese→Vietnamese (Sailor-4B)", layout="wide")
st.title("Local Chinese → Vietnamese — sail/Sailor-4B")

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("**Model (Hugging Face id)**")
    model_id = st.text_input("Model id", value="sail/Sailor-4B")
    chinese_text = st.text_area("Chinese text to translate", height=260, placeholder="例如：今天天气很好，我们去公园吧。")
    max_tokens = st.number_input("Max new tokens (output length)", min_value=32, max_value=1024, value=256, step=16)
    run_btn = st.button("Translate")

with col2:
    st.markdown("## Settings")
    st.write(f"CUDA available: {torch.cuda.is_available()}")
    prefer_4bit = st.checkbox("Prefer 4-bit quantized load (requires bitsandbytes + CUDA)", value=True)
    st.markdown(
        "- If you have a GPU and `bitsandbytes` installed, the app will try to load the model in 4-bit which drastically reduces VRAM.\n"
        "- If not available, the app will fall back to a standard load (may be large)."
    )
    st.markdown("## Notes\n- Sailor models include a fast tokenizer (tokenizer.json) so `use_fast=True` typically avoids `sentencepiece`.\n")

# cache the Translator instance so repeated runs don't reload
@st.cache_resource
def get_translator(model_id, prefer_4bit):
    return Translator(model_id, prefer_4bit=prefer_4bit)

if run_btn:
    if not model_id.strip():
        st.error("Provide model id (e.g., sail/Sailor-4B)")
    elif not chinese_text.strip():
        st.error("Paste some Chinese text to translate.")
    else:
        try:
            with st.spinner("Loading model (first run may take time) and generating..."):
                translator = get_translator(model_id.strip(), prefer_4bit)
                out = translator.translate(chinese_text.strip(), max_new_tokens=int(max_tokens))
                st.subheader("Vietnamese translation")
                st.code(out)
                st.download_button("Download .txt", data=out.encode("utf-8"), file_name="translation_vi.txt")
        except Exception as e:
            st.error("Error: " + str(e))
            st.markdown(
                "<small>If the error mentions 'sentencepiece', either install it (conda recommended) or pre-download the tokenizer.json/tokenizer files from the HF repo and place them in ./local_translator_models/<model_id>/</small>",
                unsafe_allow_html=True
            )
