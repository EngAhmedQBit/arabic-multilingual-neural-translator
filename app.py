"""Streamlit UI. Save as ``app.py`` next to ``translator_core.py``."""

import time

import streamlit as st
from translator_core import ArabicTranslator, SUPPORTED_LANGUAGES

st.set_page_config(page_title="Arabic Neural Translator", page_icon="🌍", layout="wide")

st.markdown(
    """
    <style>
    .stTextArea textarea { direction: rtl; text-align: right; font-size: 1.15rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_translator():
    return ArabicTranslator()


def clear_input():
    st.session_state["input_text"] = ""
    st.session_state["translations"] = {}


def use_demo(text):
    st.session_state["input_text"] = text
    st.session_state["translations"] = {}


translator = get_translator()
if "translations" not in st.session_state:
    st.session_state["translations"] = {}

with st.sidebar:
    st.title("⚙️ Settings / الإعدادات")
    num_beams = st.slider("Beam search", min_value=1, max_value=5, value=1,
                          help="1 is fastest; higher values may improve quality but take longer.")
    max_length = st.slider("Max tokens", min_value=32, max_value=256, value=128, step=32,
                           help="Applies to both source token limit and maximum generated tokens.")
    st.info(f"Device: {translator.device.upper()}")
    st.caption("Only one language model is held in RAM at a time. Model files stay cached on disk.")
    if st.button("🗑️ Unload model", use_container_width=True):
        translator.unload_all()
        st.success("RAM cache cleared (downloaded model files remain on disk).")

st.title("🌍 مترجم العربية إلى لغات متعددة")
st.write("Translate Arabic into up to 10 languages using local Helsinki-NLP models.")
st.caption("First use of each language downloads its model from Hugging Face. This can take significant time and disk space.")

left, right = st.columns([1, 2])
with left:
    select_all = st.checkbox("🌐 Select all 10 languages", value=False)
with right:
    st.caption("Results appear as each language finishes; failed languages don't stop the others.")

selected = []
cols = st.columns(5)
for i, (code, info) in enumerate(SUPPORTED_LANGUAGES.items()):
    with cols[i % 5]:
        # Checkbox value tracks the select-all toggle; using a changing key avoids
        # stale widget state when select-all changes.
        if st.checkbox(info["name"], value=(select_all or code in ("en", "de", "fr")),
                       key=f"language_{code}_{int(select_all)}"):
            selected.append(code)

st.text_area("📝 أدخل النص العربي / Arabic input", height=140,
             placeholder="مثال: مرحبًا، أنا أدرس الذكاء الاصطناعي.", key="input_text")

c1, c2, _ = st.columns([1, 1, 3])
with c1:
    translate_clicked = st.button("🚀 Translate", type="primary", use_container_width=True)
with c2:
    st.button("🧹 Clear", on_click=clear_input, use_container_width=True)

if translate_clicked:
    text = st.session_state.get("input_text", "").strip()
    if not text:
        st.error("Please enter some Arabic text.")
    elif not selected:
        st.error("Please select at least one language.")
    else:
        st.session_state["translations"] = {}
        st.subheader("📤 Results / النتائج")
        progress = st.progress(0)
        status = st.empty()
        slots = {code: st.empty() for code in selected}
        total = len(selected)
        for index, code in enumerate(selected, start=1):
            name = SUPPORTED_LANGUAGES[code]["name"]
            status.info(f"⏳ {index}/{total}: Preparing model / translating into {name}. "
                        "First-time download may be slow...")
            slots[code].info(f"⏳ {name}: Loading / translating...")
            start = time.perf_counter()
            try:
                result = translator.translate(text, code, max_length=max_length,
                                              num_beams=num_beams)
                seconds = time.perf_counter() - start
                st.session_state["translations"][code] = {
                    "success": True, "text": result, "seconds": seconds
                }
                # st.write does not interpret the translated text as raw HTML.
                with slots[code].container():
                    st.markdown(f"**{name}**")
                    st.write(result)
                    st.caption(f"⏱️ {seconds:.2f}s (includes model loading if needed)")
            except Exception as exc:
                seconds = time.perf_counter() - start
                st.session_state["translations"][code] = {
                    "success": False, "text": str(exc), "seconds": seconds
                }
                slots[code].error(f"❌ {name}: {exc}")
            progress.progress(index / total)
        status.success("✅ Finished all selected languages (check per-language results).")
        success_count = sum(v["success"] for v in st.session_state["translations"].values())
        st.metric("Successful translations", f"{success_count}/{total}")

elif st.session_state["translations"]:
    st.subheader("📤 Previous results / النتائج السابقة")
    for code, data in st.session_state["translations"].items():
        name = SUPPORTED_LANGUAGES[code]["name"]
        if data["success"]:
            st.markdown(f"**{name}**")
            st.write(data["text"])
            st.caption(f"⏱️ {data['seconds']:.2f}s")
        else:
            st.error(f"❌ {name}: {data['text']}")

with st.expander("📚 Example sentences / جمل تجريبية"):
    samples = [
        "السلام عليكم",
        "كيف حالك اليوم؟",
        "أنا أحب البرمجة كثيراً.",
        "التعلم الآلي فرع مهم من الذكاء الاصطناعي.",
    ]
    for i, sample in enumerate(samples):
        st.button(sample, key=f"demo_{i}", on_click=use_demo, args=(sample,))

st.caption("Powered by Helsinki-NLP OPUS-MT via Hugging Face Transformers — local inference.")
