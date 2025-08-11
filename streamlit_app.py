# app.py
import streamlit as st
from local_translator import Translator, TranslatorManager
import torch
import os

st.set_page_config(page_title="Local Chinese → Vietnamese (no sentencepiece)", layout="wide")
st.title("Local Chinese → Vietnamese translator (tries to avoid sentencepiece)")

col1, col2 = st.columns([3, 1])

with col1:
    model_id = st.text_input("Hugging Face model id or local path", value="arcee-ai/Arcee-VyLinh")
    text_in = st.text_area("Chinese text to translate", height=260, placeholder="例如：今天天气很好，我们去公园吧。")
    max_len = st.number_input("Max output tokens", min_value=32, max_value=2048, value=256, step=16)
    run_translate = st.button("Translate (local)")

with col2:
    st.markdown("## Options & info")
    st.write(f"Local models folder: `./local_translator_models`")
    device_choice = st.selectbox("Device", options=["auto", "cuda", "cpu"], index=0)
    device = None if device_choice == "auto" else device_choice
    st.write("---")
    st.write(f"CUDA available: {torch.cuda.is_available()}")
    st.markdown(
        "### Notes\n"
        "- This app prefers the *fast* tokenizer (no `sentencepiece`) but may still require sentencepiece for some models.\n"
        "- If you can't install sentencepiece on your environment, pre-download the model + tokenizer on another machine\n"
        "  (where sentencepiece can be installed) and copy the folder into `./local_translator_models/<model_name>/`.\n"
        "- First-time model downloads will take time and disk space."
    )

# caching loaded translator so Streamlit doesn't reload every interaction
# key uses model_id + device to support multiple model choices
@st.cache_resource
def get_translator_cached(model_id, device):
    return Translator(model_id, device=device)

@st.cache_resource
def get_manager_cached(model_list, device):
    return TranslatorManager(model_list, device=device)

if run_translate:
    if not model_id.strip():
        st.error("Please provide a model id or local path.")
    elif not text_in.strip():
        st.error("Please provide Chinese text to translate.")
    else:
        try:
            with st.spinner("Loading model (if not cached) and translating — this may take a while on first run..."):
                if "\n" in model_id.strip():
                    # if user pasted multiple models (one per line), treat as manager
                    model_list = [m.strip() for m in model_id.splitlines() if m.strip()]
                    manager = get_manager_cached(tuple(model_list), device)
                    results = manager.translate_all(text_in.strip())
                    for name, out in results.items():
                        st.subheader(name)
                        if isinstance(out, str) and out.lower().startswith("error"):
                            st.error(out)
                        else:
                            st.code(out)
                else:
                    trans = get_translator_cached(model_id.strip(), device)
                    result = trans.translate(text_in.strip(), max_length=int(max_len))
                    st.subheader("Vietnamese translation")
                    st.code(result)
                    st.download_button("Download (.txt)", data=result.encode("utf-8"), file_name="translation_vi.txt")
        except Exception as e:
            st.error("Error: " + str(e))
            # show hint for advanced users
            st.markdown(
                "<small>Hint: if the error mentions 'sentencepiece', either install it (conda recommended) or pre-download the "
                "tokenizer.json/tokenizer files on another machine and copy them into the local model folder.</small>",
                unsafe_allow_html=True
            )
