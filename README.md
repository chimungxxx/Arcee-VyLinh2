# Chinese to Vietnamese Translator

A Streamlit web application for translating Chinese text to Vietnamese using the `arcee-ai/Arcee-VyLinh` model.

## Features

- Real-time Chinese to Vietnamese translation
- Built with Streamlit for easy deployment
- Powered by the Arcee-VyLinh translation model

## Deployment

This app is configured for deployment on [Streamlit Community Cloud](https://streamlit.io/cloud).

### Quick Deploy

1. Fork this repository
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/)
3. Click "New app" and select this repository
4. Set the main file path to `app.py`
5. Click "Deploy!"

## Local Development

```bash
# Clone the repository
git clone <repository-url>

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Requirements

- Python 3.8+
- Streamlit
- Transformers
- Torch
- SentencePiece

## Model

This application uses the [`arcee-ai/Arcee-VyLinh`](https://huggingface.co/arcee-ai/Arcee-VyLinh) model from Hugging Face for translation.

## License

MIT License
```

Note: You'll also need a `requirements.txt` file with the necessary dependencies:

```txt
streamlit
transformers
torch
sentencepiece
```
