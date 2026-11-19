# 🌍 Arabic-to-Multiple-Languages Neural Translator

A complete project for translating Arabic text into **10 languages** using Helsinki-NLP OPUS-MT models, with both a **Streamlit GUI** and a **Python API**.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🌐 **10 Languages** | English, German, French, Spanish, Italian, Russian, Turkish, Polish, Hebrew, Greek |
| 🖥️ **Streamlit GUI** | Beautiful web interface with RTL Arabic support |
| ⚡ **Model Caching** | Models load once and stay in memory for fast reuse |
| 🔍 **Beam Search** | Adjustable beam width for quality vs. speed trade-off |
| 📦 **Batch Translation** | Translate multiple sentences in one call |
| 🧪 **Demo Dataset** | Built-in test sentences for quick evaluation |

---

## 📁 Project Structure

```
arabic_multilang_translator/
├── requirements.txt       # Python dependencies
├── translator_core.py     # Core translation engine (reusable API)
├── app.py                 # Streamlit GUI application
├── test_translator.py     # CLI test script
└── README.md              # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit GUI

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`.

### 3. Run CLI Tests

```bash
python test_translator.py
```

### 4. Use as a Python Library

```python
from translator_core import ArabicTranslator

translator = ArabicTranslator()

# Translate to English
result = translator.translate("مرحباً بالعالم", target_lang="en")
print(result)  # "Hello world"

# Translate to French
result = translator.translate("مرحباً بالعالم", target_lang="fr")
print(result)  # "Bonjour le monde"

# Batch translation
results = translator.translate_batch(
    ["السلام عليكم", "كيف حالك؟"],
    target_lang="de"
)
print(results)  # ["Grüß Gott", "Wie geht es dir?"]
```

---

## 🗣️ Supported Languages

| Code | Language | Model |
|------|----------|-------|
| `en` | 🇺🇸 English | `Helsinki-NLP/opus-mt-tc-big-ar-en` |
| `de` | 🇩🇪 German | `Helsinki-NLP/opus-mt-ar-de` |
| `fr` | 🇫🇷 French | `Helsinki-NLP/opus-mt-ar-fr` |
| `es` | 🇪🇸 Spanish | `Helsinki-NLP/opus-mt-ar-es` |
| `it` | 🇮🇹 Italian | `Helsinki-NLP/opus-mt-ar-it` |
| `ru` | 🇷🇺 Russian | `Helsinki-NLP/opus-mt-ar-ru` |
| `tr` | 🇹🇷 Turkish | `Helsinki-NLP/opus-mt-ar-tr` |
| `pl` | 🇵🇱 Polish | `Helsinki-NLP/opus-mt-ar-pl` |
| `he` | 🇮🇱 Hebrew | `Helsinki-NLP/opus-mt-ar-he` |
| `el` | 🇬🇷 Greek | `Helsinki-NLP/opus-mt-ar-el` |

---

## ⚙️ GUI Features

- **🎯 Language Selection**: Checkbox grid to choose any combination of target languages
- **🔍 Beam Search Slider**: Control translation quality (1–10 beams)
- **📏 Max Length Slider**: Limit output length
- **📚 Demo Sentences**: One-click test with pre-loaded Arabic sentences
- **📊 Real-time Stats**: Success count, timing, input length metrics
- **🗑️ Model Management**: Unload models from sidebar to free memory
- **🌙 RTL Support**: Full right-to-left Arabic text input

---

## 🧠 How It Works

1. **Model Loading**: The first time you translate to a language, its Helsinki-NLP OPUS-MT model is downloaded from Hugging Face and cached in GPU/CPU memory.
2. **Translation**: Arabic text is tokenized → encoded → passed through the transformer → decoded into the target language.
3. **Caching**: Loaded models stay in memory so subsequent translations to the same language are instant.

---

## 💡 Tips

- **GPU Recommended**: Models run much faster on CUDA. CPU works but is slower.
- **First Load**: The first translation to a new language downloads the model (~300MB–1GB). Be patient.
- **Memory**: Each loaded model uses ~300MB–1GB RAM. Unload unused models from the sidebar if needed.
- **Quality vs Speed**: Lower `num_beams` (e.g., 1–2) for faster translation; higher (e.g., 6–10) for better quality.

---

## 📜 License

This project uses models from the [Helsinki-NLP OPUS-MT](https://huggingface.co/Helsinki-NLP) project, released under open licenses. The code is provided as-is for educational purposes.

---

## 🙏 Acknowledgements

- [Helsinki-NLP](https://github.com/Helsinki-NLP/Opus-MT) for the OPUS-MT models
- [Hugging Face](https://huggingface.co/) for the Transformers library
- [Streamlit](https://streamlit.io/) for the GUI framework
