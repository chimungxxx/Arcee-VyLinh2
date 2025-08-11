# app.py
import streamlit as st
import requests
from typing import Optional
import time
import hashlib

st.set_page_config(page_title="Chinese → Vietnamese — Arcee-VyLinh", page_icon="🌏", layout="centered")

MODEL_ID = "arcee-ai/Arcee-VyLinh"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

# Helper to get token from secrets or UI
def get_hf_token() -> Optional[str]:
    # Prefer streamlit secrets (recommended for deployment on streamlit.app)
    token = st.secrets.get("HF_API_TOKEN") if "HF_API_TOKEN" in st.secrets else None
    if token:
        return token
    # Fallback to typed-in token (not recommended for production)
    return st.session_state.get("hf_token_input")

# Call Hugging Face Inference API
def call_inference_api(prompt: str, token: str, max_new_tokens: int = 512, temperature: float = 0.2, top_p: float = 0.95):
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            # you can add other text-generation-inference params here
        },
        # "options": {"use_cache": False}  # optional
    }
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    # For text-generation models the response is usually a list of dicts with 'generated_text'
    data = resp.json()
    # Attempt to parse common response shapes:
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(data["error"])
    if isinstance(data, list):
        # expected: [{"generated_text": "..."}]
        first = data[0]
        if isinstance(first, dict) and "generated_text" in first:
            return first["generated_text"]
        # some endpoints return [{"generated_text": "...", ...}]
        # others may directly return text; handle that next:
    if isinstance(data, str):
        return data
    # fallback: stringify
    return str(data)

@st.cache_data(show_spinner=False)
def cached_translation(prompt_hash: str, prompt_text: str, token_hash: str, params_hash: str, result_text: str):
    # This function exists so the caching decorator persists results keyed by hash.
    # It returns the same result back (no heavy work here). The cache key is composite of hashes.
    return result_text

def make_prompt(chinese_text: str) -> str:
    # Instruction prompt to guide the model
    # Keep clear: ask for faithful, natural Vietnamese translation.
    return (
        "You are an expert translator. Translate the following Chinese text into natural, fluent Vietnamese.\n\n"
        "Requirements:\n"
        "- Preserve meaning and tone (formal/informal) where possible.\n"
        "- Keep named entities as-is unless a standard Vietnamese form exists.\n\n"
        "Translate now:\n\n"
        f"{chinese_text}\n\n"
        "Output only the Vietnamese translation (no extra commentary)."
    )

# UI
st.title("Chinese → Vietnamese translator — Arcee-VyLinh")
st.caption("Uses the model arcee-ai/Arcee-VyLinh via Hugging Face Inference API")

col1, col2 = st.columns([3, 1])
with col1:
    chinese_input = st.text_area("Enter Chinese text", height=200, placeholder="例如：今天天气很好，我们去公园吧。")
with col2:
    st.write("Options")
    max_tokens = st.number_input("Max new tokens", min_value=50, max_value=2048, value=256, step=50)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    top_p = st.slider("Top-p (nucleus)", min_value=0.1, max_value=1.0, value=0.95, step=0.01)

st.markdown("---")

# HF token input (only if not set in secrets)
if "HF_API_TOKEN" not in st.secrets:
    st.info("Hugging Face token not found in Streamlit Secrets. For secure deployment, add HF_API_TOKEN to your app's Secrets.")
    st.text_input("Or paste a Hugging Face API token (will be stored in session only):", type="password", key="hf_token_input")

submit = st.button("Translate")

if submit:
    token = get_hf_token()
    if not token:
        st.error("No Hugging Face API token provided. Add one in Streamlit Secrets (HF_API_TOKEN) or paste it in the box.")
    elif not chinese_input.strip():
        st.error("Please enter Chinese text to translate.")
    else:
        prompt = make_prompt(chinese_input.strip())
        # create simple cache keys to avoid re-calling on identical inputs
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()[:8]
        params_hash = hashlib.sha256(f"{max_tokens}-{temperature}-{top_p}".encode()).hexdigest()[:8]

        try:
            with st.spinner("Translating..."):
                # check cache first via cached_translation (decorated)
                cached_key = f"{prompt_hash}-{token_hash}-{params_hash}"
                # Try: if we already cached, quickly return it (cached_translation will return)
                # Otherwise call API and store in cache function call result.
                # Note: st.cache_data uses function signature to key, so we pass the keys.
                # We attempt to read cached result by calling API only if cache miss.
                # Simple approach: try to call call_inference_api directly (we rely on cache decorator on storing).
                result_text = call_inference_api(prompt, token, max_new_tokens=int(max_tokens), temperature=float(temperature), top_p=float(top_p))
                # store (or update) cache
                cached_translation(cached_key, prompt, token_hash, params_hash, result_text)
            st.success("Done")
            st.subheader("Vietnamese translation")
            st.code(result_text, language="text")
            st.download_button("Download translation (.txt)", data=result_text.encode("utf-8"), file_name="translation_vi.txt")
        except requests.exceptions.HTTPError as e:
            try:
                st.error(f"HTTP error: {e.response.status_code} — {e.response.text}")
            except Exception:
                st.error(f"HTTP error: {e}")
        except Exception as ex:
            st.error(f"Error: {ex}")
            st.caption("If you believe this is an error from the model or API, check your token & quota, or try again in a moment.")
