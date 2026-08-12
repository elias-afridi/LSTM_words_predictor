import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import re
import time
from typing import Tuple, Optional

# ────────────────────────────────────────────────────────────────────────────
# Page configuration — MUST be the first Streamlit command
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wordcraft — Neural Text Generator",
    page_icon="🪶",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────────────────────────────────
# Design system.
# Every color rule below is !important because Streamlit injects its own
# CSS-in-JS styles (auto-generated, high-specificity class names) that would
# otherwise silently win over plain rules — that was the root cause of the
# invisible/backgroundless look in the previous version.
# ────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');

    :root {
        --ink: #1c1c1f;
        --muted: #6b6f76;
        --accent: #4f46e5;
        --accent-soft: #eef0fd;
        --surface: #ffffff;
        --surface-alt: #f5f5f8;
        --border: #e2e2e8;
    }

    html, body {
        background-color: var(--surface-alt) !important;
    }

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}

    /* Force the light theme on every main container, regardless of the
       visitor's OS/browser dark-mode setting. */
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stMain"],
    [data-testid="stBottomBlockContainer"] {
        background-color: var(--surface-alt) !important;
    }

    /* Default text color everywhere unless overridden below */
    p, span, label, li, div, h1, h2, h3, h4, h5, h6 {
        color: var(--ink) !important;
    }

    .block-container {
        padding-top: 2.5rem;
        max-width: 760px;
    }

    /* ── Header ───────────────────────────────────────────────────────── */
    .app-title {
        font-family: 'Source Serif 4', serif;
        font-size: 2.4rem;
        font-weight: 600;
        color: var(--ink) !important;
        text-align: center;
        margin-bottom: 0.15rem;
        letter-spacing: -0.01em;
    }
    .app-subtitle {
        text-align: center;
        color: var(--muted) !important;
        font-size: 0.98rem;
        margin-bottom: 1rem;
        font-weight: 400;
    }
    .disclaimer {
        text-align: center;
        color: var(--muted) !important;
        font-size: 0.82rem;
        background: var(--accent-soft) !important;
        border: 1px solid #d8dafa;
        border-radius: 10px;
        padding: 8px 16px;
        margin: 0 auto 2rem;
        max-width: 520px;
    }
    .disclaimer b { color: var(--accent) !important; }

    /* ── Inputs ───────────────────────────────────────────────────────── */
    .stTextArea textarea {
        font-family: 'Source Serif 4', serif !important;
        font-size: 1.05rem;
        line-height: 1.6;
        border-radius: 14px;
        border: 1.5px solid var(--border) !important;
        background: var(--surface) !important;
        color: var(--ink) !important;
        padding: 16px 18px;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 4px var(--accent-soft);
    }
    .stTextArea textarea::placeholder {
        color: var(--muted) !important;
        opacity: 1 !important;
    }

    /* ── Buttons: Streamlit renders kind="primary" / kind="secondary" ──── */
    .stButton > button, div[data-testid="stDownloadButton"] > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.18s ease;
        width: 100%;
    }

    button[kind="primary"] {
        background: var(--ink) !important;
        border: none !important;
    }
    button[kind="primary"] p, button[kind="primary"] span {
        color: #ffffff !important;
    }
    button[kind="primary"]:hover {
        background: var(--accent) !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(79, 70, 229, 0.25);
    }

    button[kind="secondary"] {
        background: var(--surface) !important;
        border: 1.5px solid var(--border) !important;
    }
    button[kind="secondary"] p, button[kind="secondary"] span {
        color: var(--ink) !important;
    }
    button[kind="secondary"]:hover {
        border-color: var(--accent) !important;
    }
    button[kind="secondary"]:hover p, button[kind="secondary"]:hover span {
        color: var(--accent) !important;
    }

    /* ── Result card ──────────────────────────────────────────────────── */
    .result-card {
        background: var(--surface) !important;
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 28px 30px;
        margin: 1.4rem 0;
        font-family: 'Source Serif 4', serif;
        font-size: 1.15rem;
        line-height: 1.75;
        box-shadow: 0 2px 10px rgba(20, 20, 30, 0.04);
    }
    .result-card .seed-part { color: var(--muted) !important; }
    .result-card .gen-part { color: var(--ink) !important; font-weight: 500; }

    /* ── Stat pills ───────────────────────────────────────────────────── */
    .stat-row { display: flex; gap: 10px; margin: 1.2rem 0; }
    .stat-pill {
        flex: 1;
        background: var(--surface) !important;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px 8px;
        text-align: center;
    }
    .stat-num { font-size: 1.4rem; font-weight: 700; color: var(--accent) !important; }
    .stat-label { font-size: 0.72rem; color: var(--muted) !important; text-transform: uppercase; letter-spacing: 0.04em; margin-top: 2px; }

    /* ── Sidebar ──────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] * {
        color: var(--ink) !important;
    }

    .info-callout {
        background: var(--accent-soft) !important;
        color: #3b3fa0 !important;
        padding: 12px 16px;
        border-radius: 10px;
        font-size: 0.85rem;
        margin: 10px 0;
        border: 1px solid #d8dafa;
    }
    .info-callout, .info-callout * { color: #3b3fa0 !important; }

    .footer-note {
        text-align: center;
        color: var(--muted) !important;
        font-size: 0.8rem;
        padding: 28px 0 10px;
        border-top: 1px solid var(--border);
        margin-top: 2.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# Resource loading (cached)
# ────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources() -> Tuple[Optional[tf.keras.Model], Optional[pickle], Optional[int]]:
    """Load the trained model, tokenizer, and max_len with caching."""
    try:
        model = load_model('lstm_model.h5', compile=False)
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        with open('max_len.pickle', 'rb') as handle:
            max_len = pickle.load(handle)
        return model, tokenizer, max_len
    except FileNotFoundError as e:
        st.error(
            f"❌ File not found: {e}\n\n"
            "Make sure the following files exist in the current directory:\n"
            "- lstm_model.h5\n- tokenizer.pickle\n- max_len.pickle"
        )
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None


def clean_text(text: str) -> str:
    """Clean and preprocess the input text for tokenization."""
    text = text.lower().strip()
    text = re.sub(r'[^a-zA-Z0-9\s\.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def generate_text(
    model: tf.keras.Model,
    tokenizer: pickle,
    seed_text: str,
    max_len: int,
    num_words: int = 30,
    temperature: float = 1.0
) -> str:
    """Generate text using the LSTM model with temperature sampling."""
    seed_text = clean_text(seed_text)
    token_list = tokenizer.texts_to_sequences([seed_text])[0]

    if not token_list:
        return seed_text

    # Reverse lookup built once instead of scanning word_index every step
    index_to_word = {index: word for word, index in tokenizer.word_index.items()}

    generated_text = seed_text

    for _ in range(num_words):
        token_list_padded = tf.keras.preprocessing.sequence.pad_sequences(
            [token_list], maxlen=max_len - 1, padding='pre'
        )

        predicted_probs = model.predict(token_list_padded, verbose=0)[0]

        if temperature != 1.0:
            predicted_probs = np.log(predicted_probs + 1e-7) / temperature
            predicted_probs = np.exp(predicted_probs)
            predicted_probs = predicted_probs / np.sum(predicted_probs)

        predicted_word_index = int(np.argmax(predicted_probs))
        predicted_word = index_to_word.get(predicted_word_index)

        if predicted_word is None:
            break

        token_list.append(predicted_word_index)
        generated_text += " " + predicted_word

        if len(token_list) > max_len - 1:
            token_list = token_list[-(max_len - 1):]

    return generated_text


# ────────────────────────────────────────────────────────────────────────────
# App
# ────────────────────────────────────────────────────────────────────────────
EXAMPLES = [
    "The world is",
    "There is no",
    "It is better to",
    "If you want to",
    "I believe in",
    "The best way to",
]


def main():
    # Session state defaults
    if "seed_text" not in st.session_state:
        st.session_state.seed_text = EXAMPLES[0]
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    # IMPORTANT: any pending update to the seed text must be applied
    # *before* st.text_area(key="seed_text") is instantiated below.
    # Streamlit forbids writing to a widget's session_state key after
    # that widget has already been created in the same run — that was
    # why the example chips previously did nothing.
    if "pending_seed" in st.session_state:
        st.session_state.seed_text = st.session_state.pop("pending_seed")

    st.markdown('<div class="app-title">🪶 Wordcraft</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Continue any sentence with a neural language model, word by word</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="disclaimer">🎓 This is a <b>learning project</b> — the underlying model '
        'has roughly 72% accuracy and trained on a small dataset, so generated text may be repetitive, ungrammatical, or '
        'not fully coherent. Treat the output as a demo, not a finished product.</div>',
        unsafe_allow_html=True
    )

    with st.spinner("Loading model…"):
        model, tokenizer, max_len = load_resources()

    if model is None or tokenizer is None or max_len is None:
        st.warning("⚠️ Please make sure the model files are in the current directory.")
        st.markdown("""
        <div class="info-callout">
        <b>Required files</b><br>
        • lstm_model.h5 — the trained LSTM model<br>
        • tokenizer.pickle — the fitted tokenizer<br>
        • max_len.pickle — the maximum sequence length
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Generation")
        num_words = st.slider("Words to generate", 2, 20, 10, step=2)
        temperature = st.slider(
            "Temperature", 0.1, 2.0, 1.0, step=0.1,
            help="Lower = focused and predictable. Higher = varied and creative."
        )

        st.markdown("---")
        st.markdown("### 📊 Model")
        st.markdown(
            f'<div class="info-callout">Max sequence length: <b>{max_len}</b><br>'
            f'Vocabulary size: <b>{len(tokenizer.word_index) + 1:,}</b><br>'
            f'Reported accuracy: <b>~72%</b></div>',
            unsafe_allow_html=True
        )
        with st.expander("Model architecture"):
            model.summary(print_fn=lambda x: st.text(x))

        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown(
            "- Give it 5–10 words of context\n"
            "- Lower temperature for coherence\n"
            "- Higher temperature for variety"
        )

    # ── Seed input ───────────────────────────────────────────────────────
    st.markdown("#### Start a sentence")
    seed_text = st.text_area(
        "Seed text",
        key="seed_text",
        height=100,
        label_visibility="collapsed",
        placeholder="Type a few words to begin…"
    )

    st.caption("Try an example:")
    chip_cols = st.columns(3)
    for i, example in enumerate(EXAMPLES):
        with chip_cols[i % 3]:
            if st.button(example, key=f"chip_{i}", use_container_width=True):
                st.session_state.pending_seed = example
                st.rerun()

    st.write("")
    generate_clicked = st.button("Generate →", use_container_width=True, type="primary")

    # ── Generation ───────────────────────────────────────────────────────
    if generate_clicked:
        if not seed_text.strip():
            st.warning("⚠️ Please enter some seed text first.")
        else:
            with st.spinner("Writing…"):
                start_time = time.time()
                try:
                    generated_text = generate_text(
                        model, tokenizer, seed_text, max_len, num_words, temperature
                    )
                    elapsed = time.time() - start_time
                    st.session_state.last_result = {
                        "seed": seed_text.strip(),
                        "text": generated_text,
                        "elapsed": elapsed,
                    }
                except Exception as e:
                    st.error(f"❌ Error generating text: {e}")
                    st.session_state.last_result = None

    # ── Result ───────────────────────────────────────────────────────────
    result = st.session_state.last_result
    if result:
        seed_words = result["seed"].split()
        all_words = result["text"].split()
        new_words = all_words[len(seed_words):]

        st.markdown("#### Result")
        st.markdown(
            f'<div class="result-card">'
            f'<span class="seed-part">{result["seed"]}</span> '
            f'<span class="gen-part">{" ".join(new_words)}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-pill"><div class="stat-num">{len(seed_words)}</div><div class="stat-label">Seed words</div></div>
            <div class="stat-pill"><div class="stat-num">{len(new_words)}</div><div class="stat-label">New words</div></div>
            <div class="stat-pill"><div class="stat-num">{len(all_words)}</div><div class="stat-label">Total words</div></div>
            <div class="stat-pill"><div class="stat-num">{result["elapsed"]:.2f}s</div><div class="stat-label">Time</div></div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "💾 Download as .txt",
                data=result["text"],
                file_name=f"wordcraft_{int(time.time())}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col2:
            if st.button("🔄 Regenerate", use_container_width=True):
                with st.spinner("Writing…"):
                    start_time = time.time()
                    generated_text = generate_text(
                        model, tokenizer, result["seed"], max_len, num_words, temperature
                    )
                    st.session_state.last_result = {
                        "seed": result["seed"],
                        "text": generated_text,
                        "elapsed": time.time() - start_time,
                    }
                st.rerun()

    st.markdown(
        '<div class="footer-note">Built with Streamlit · TensorFlow · LSTM · Practice project, ~72% accuracy</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
