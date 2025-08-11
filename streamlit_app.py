# app.py
import streamlit as st
import torch
import logging
from local_translator import Translator, TranslatorManager, MODEL_ID

# Configure a Streamlit-friendly logger (server captures stdout/stderr)
logger = logging.getLogger("sail_translator_app")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

st.set_page_config(page_title="Sailor-0.5B Chinese→Vietnamese", layout="wide")
st.title("Local Chinese → Vietnamese — sail/Sailor-0.5B (fixed)")

st.markdown(f"**Model (fixed):** `{MODEL_ID}`")

col1, col2 = st.columns([3, 1])
with col1:
    chinese_text = st.text_area("Chinese text to translate", height=260, placeholder="例如：今天天气很好，我们去公园吧。")
    max_tokens = st.number_input("Max new tokens (output length)", min_value=32, max_value=1024, value=256, step=16)
    run_btn = st.button("Translate")

with col2:
    st.markdown("## Settings & Info")
    st.write(f"CUDA available: {torch.cuda.is_available()}")
    prefer_4bit = st.checkbox("Prefer 4-bit quantized load (requires bitsandbytes + CUDA)", value=True)
    st.markdown(
        "- The model is fixed to `sail/Sailor-0.5B` to reduce memory footprint compared to larger LLMs.\n"
        "- Logs about downloading/loading appear in the server logs (stdout/stderr). Check your cloud provider's log viewer.\n"
    )

@st.cache_resource
def get_translator_cached(prefer_4bit):
    logger.info("Creating Translator (cached) with prefer_4bit=%s", prefer_4bit)
    return Translator(MODEL_ID, prefer_4bit=prefer_4bit)

if run_btn:
    if not chinese_text.strip():
        st.error("Please enter Chinese text to translate.")
    else:
        try:
            with st.spinner("Loading model (may take time on first run) and translating..."):
                translator = get_translator_cached(prefer_4bit)
                result = translator.translate(chinese_text.strip(), max_new_tokens=int(max_tokens))
                st.subheader("Vietnamese translation")
                st.code(result)
                st.download_button("Download .txt", data=result.encode("utf-8"), file_name="translation_vi.txt")
        except Exception as e:
            # Both print to server log and show helpful message in UI
            logger.error("Error in app during translation: %s", e)
            st.error("Error: " + str(e))
            st.markdown(
                "<small>See server logs for full debug messages. If the error mentions 'weights' or 'offline', you may need to pre-download\n"
                "the model weights on another machine and copy the model folder to `./local_translator_models/sail_Sailor-0.5B/`.</small>",
                unsafe_allow_html=True,
            )
