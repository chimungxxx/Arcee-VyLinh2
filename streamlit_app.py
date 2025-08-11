# app.py
import streamlit as st
from local_translator import Translator, TranslatorManager
import os
import torch

st.set_page_config(page_title="Local Chinese→Vietnamese Translator", layout="wide")

st.title("Local Chinese → Vietnamese translator (no API token)")

col1, col2 = st.columns([2, 1])

with col1:
    model_input = st.text_input(
        "Hugging Face model id or local path",
        value="arcee-ai/Arcee-VyLinh",
        help="Examples: 'arcee-ai/Arcee-VyLinh' or a local folder path where model is already downloaded."
    )
    chinese_text = st.text_area("Chinese text to translate", height=240, placeholder="例如：今天天气很好，我们去公园吧。")
    max_length = st.number_input("Max tokens / output length", min_value=32, max_value=2048, value=256, step=16)
    device_select = st.selectbox("Device", options=["auto", "cuda", "cpu"], index=0)
    if device_select == "auto":
        device = None
    else:
        device = device_select

with col2:
    st.markdown("## Options")
    st.write(f"Local models base folder: `./local_translator_models`")
    use_manager = st.checkbox("Load as TranslatorManager (multiple models)", value=False)
    if use_manager:
        extra_models = st.text_area("Additional model ids (one per line)", height=120, placeholder="one per line")
    st.write("---")
    st.write("Info")
    st.write(f"torch device available: {torch.cuda.is_available()}")
    st.write(f"Current Python process device selection: {device or 'auto (cuda if available else cpu)'}")

translate_btn = st.button("Translate (local)")

if translate_btn:
    if not model_input.strip():
        st.error("Please provide a model id or local path.")
    elif not chinese_text.strip():
        st.error("Please paste some Chinese text to translate.")
    else:
        with st.spinner("Loading model and translating (this may take a while the first time)..."):
            try:
                if use_manager:
                    model_list = [model_input.strip()] + [m.strip() for m in extra_models.splitlines() if m.strip()]
                    manager = TranslatorManager(model_list)
                    results = manager.translate_all(chinese_text.strip())
                    for name, out in results.items():
                        st.subheader(name)
                        if out and out.startswith("Error"):
                            st.error(out)
                        else:
                            st.code(out)
                else:
                    translator = Translator(model_input.strip(), device=device)
                    result = translator.translate(chinese_text.strip(), max_length=int(max_length))
                    st.subheader("Translation")
                    st.code(result)
                    st.download_button("Download translation (.txt)", data=result.encode("utf-8"), file_name="translation_vi.txt")
            except Exception as e:
                st.exception(e)
