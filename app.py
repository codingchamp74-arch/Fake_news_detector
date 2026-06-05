"""
╔══════════════════════════════════════════════════════════════════╗
║  FAKE NEWS DETECTOR — Streamlit Web App  (v2 - auto label fix)  ║
║  Run: streamlit run app.py                                      ║
╚══════════════════════════════════════════════════════════════════╝

ROOT CAUSE OF "always REAL" BUG
────────────────────────────────
The WELFake / Kaggle fake-news datasets commonly encode labels as:
    0 = REAL   1 = FAKE   ← opposite of what F6.py assumed
This app auto-detects the correct mapping at startup by running
known-fake and known-real probe headlines through the model, then
picks whichever assignment gives the higher hit-rate.
"""

import os, re, string, pickle, warnings
import numpy as np
import streamlit as st
import nltk

for pkg in ["stopwords", "punkt", "punkt_tab", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore")

# ── File paths ───────────────────────────────────────────────────
MODEL_PKL = "linsvm_model.pkl"
VEC_PKL   = "tfidf_vectorizer.pkl"

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ────────────────────────────────────────────────────────────────
# TEXT CLEANING  (identical to F6.py)
# ────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>",   " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"@\w+",    " ", text)
    text = re.sub(r"#",       " ", text)
    text = re.sub(r"\d+",     " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+",     " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# ────────────────────────────────────────────────────────────────
# AUTO LABEL DETECTION
# Uses 10 well-known FAKE and 10 REAL headlines as probes.
# Tries both mappings; picks the one with higher accuracy.
# ────────────────────────────────────────────────────────────────
PROBE_FAKE = [
    "Mark Zuckerberg Will Give You $5,000 if You Share This Post",
    "Obama Signs Executive Order Banning the Pledge of Allegiance",
    "Pope Francis Shocks World Endorses Donald Trump for President",
    "Bill Gates Wants to Depopulate the World Using Vaccines",
    "NASA Admits to Chemtrails Spraying Program to Control Population",
    "Hillary Clinton Ran a Child Trafficking Ring from a Pizza Restaurant",
    "Scientists Discover Cure for Cancer But Big Pharma Is Hiding It",
    "Immigrants Arrested for Voter Fraud in Dozens of States",
    "Breaking: Martial Law Declared in Texas Following Border Invasion",
    "Soros Funds Antifa to Overthrow the US Government This Weekend",
]
PROBE_REAL = [
    "Federal Reserve raises interest rates by 25 basis points",
    "Ukraine and Russia hold ceasefire talks in Turkey",
    "Apple reports record quarterly revenue driven by iPhone sales",
    "WHO declares end to COVID-19 global health emergency",
    "NASA successfully launches Artemis mission to lunar orbit",
    "Congress passes bipartisan infrastructure spending bill",
    "Supreme Court rules on affirmative action in college admissions",
    "Oil prices drop after OPEC agrees to increase production",
    "Tech layoffs continue as Microsoft cuts 10,000 jobs",
    "UN Security Council votes on resolution condemning attacks",
]

def auto_detect_label_map(model, vec) -> dict:
    """
    Returns the label map {raw_int: 'FAKE'/'REAL'} that best matches probes.
    Falls back to {0: 'FAKE', 1: 'REAL'} if probes are inconclusive.
    """
    classes = sorted(model.classes_.tolist())
    if len(classes) != 2:
        return {0: "FAKE", 1: "REAL"}

    c0, c1 = classes[0], classes[1]

    def score_mapping(fake_label_int):
        """Count correct hits assuming fake_label_int → FAKE."""
        real_label_int = c1 if fake_label_int == c0 else c0
        hits = 0
        for t in PROBE_FAKE:
            x = vec.transform([clean_text(t)])
            pred = model.predict(x)[0]
            if pred == fake_label_int:
                hits += 1
        for t in PROBE_REAL:
            x = vec.transform([clean_text(t)])
            pred = model.predict(x)[0]
            if pred == real_label_int:
                hits += 1
        return hits

    hits_c0_fake = score_mapping(c0)   # c0=FAKE, c1=REAL
    hits_c1_fake = score_mapping(c1)   # c1=FAKE, c0=REAL

    if hits_c0_fake >= hits_c1_fake:
        label_map = {c0: "FAKE", c1: "REAL"}
        detection_note = f"Auto-detected: class {c0}=FAKE, {c1}=REAL (score {hits_c0_fake}/20)"
    else:
        label_map = {c1: "FAKE", c0: "REAL"}
        detection_note = f"Auto-detected: class {c1}=FAKE, {c0}=REAL (score {hits_c1_fake}/20)"

    return label_map, detection_note


# ────────────────────────────────────────────────────────────────
# MODEL LOADING  (cached)
# ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not os.path.exists(MODEL_PKL) or not os.path.exists(VEC_PKL):
        return None, None, None, None
    model = pickle.load(open(MODEL_PKL, "rb"))
    vec   = pickle.load(open(VEC_PKL,   "rb"))
    label_map, detection_note = auto_detect_label_map(model, vec)
    return model, vec, label_map, detection_note


# ────────────────────────────────────────────────────────────────
# PREDICTION
# ────────────────────────────────────────────────────────────────
def predict(text: str, model, vec, label_map):
    cleaned = clean_text(text)
    if not cleaned.strip():
        return None, None, None
    X = vec.transform([cleaned])
    raw = int(model.predict(X)[0])
    label = label_map.get(raw, "UNKNOWN")
    try:
        score = model.decision_function(X)[0]
        # Decision score sign: positive → model's class-1, negative → class-0
        # We need the distance magnitude for confidence, direction for label
        confidence = round(1 / (1 + np.exp(-abs(score))) * 100, 1)
    except Exception:
        confidence = None
    return label, confidence, cleaned


# ════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}
#MainMenu, footer, header { visibility: hidden; }

.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: clamp(2rem, 5vw, 3.5rem);
    background: linear-gradient(90deg, #e0e0ff 0%, #a78bfa 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}
.hero-sub { color: #94a3b8; font-size: 1.05rem; font-weight: 300; margin-bottom: 2rem; }

.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 2rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1.5rem;
}

.badge-real {
    display: inline-block;
    background: linear-gradient(135deg, #10b981, #059669);
    color: white;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.5rem;
    padding: 0.6rem 2rem;
    border-radius: 50px;
    letter-spacing: 0.15em;
    box-shadow: 0 0 30px rgba(16,185,129,0.4);
}
.badge-fake {
    display: inline-block;
    background: linear-gradient(135deg, #ef4444, #b91c1c);
    color: white;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.5rem;
    padding: 0.6rem 2rem;
    border-radius: 50px;
    letter-spacing: 0.15em;
    box-shadow: 0 0 30px rgba(239,68,68,0.4);
}

.conf-text { color: #cbd5e1; font-size: 0.95rem; margin-top: 0.75rem; }
.conf-bar-bg {
    background: rgba(255,255,255,0.1);
    border-radius: 50px;
    height: 10px;
    margin-top: 0.5rem;
    overflow: hidden;
}

textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.65rem 2.5rem !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.45) !important;
}

.info-box {
    background: rgba(56,189,248,0.08);
    border-left: 3px solid #38bdf8;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    color: #94a3b8;
    font-size: 0.88rem;
    margin-top: 1rem;
}
.warn-box {
    background: rgba(251,191,36,0.08);
    border-left: 3px solid #fbbf24;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    color: #94a3b8;
    font-size: 0.88rem;
    margin-top: 1rem;
}

label, .stTextArea label { color: #cbd5e1 !important; font-weight: 500 !important; }
hr { border-color: rgba(255,255,255,0.08) !important; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  LOAD
# ════════════════════════════════════════════════════════════════
model, vec, label_map, detection_note = load_model()

# ── HERO ─────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🔍 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Powered by Linear SVM · TF-IDF · NLTK</div>', unsafe_allow_html=True)

if model is None:
    st.error(
        "**Model files not found.**\n\n"
        "Place `linsvm_model.pkl` and `tfidf_vectorizer.pkl` in the same folder as `app.py`, "
        "then rerun.\n\nTrain by running `F6.py` first."
    )
    st.stop()

# ════════════════════════════════════════════════════════════════
#  INPUT CARD
# ════════════════════════════════════════════════════════════════
st.markdown('<div class="card">', unsafe_allow_html=True)

news_input = st.text_area(
    "Paste a news headline or article excerpt",
    placeholder="e.g.  Scientists discover water on Mars in latest NASA mission…",
    height=180,
    key="news_input",
)

analyse_btn = st.button("Analyse →", key="analyse")
st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#  RESULT
# ════════════════════════════════════════════════════════════════
if analyse_btn:
    if not news_input.strip():
        st.warning("Please enter some text before clicking Analyse.")
    else:
        with st.spinner("Analysing…"):
            label, confidence, cleaned = predict(news_input, model, vec, label_map)

        if label is None:
            st.warning("Text was too short or contained no usable words after cleaning.")
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)

            badge_cls = "badge-real" if label == "REAL" else "badge-fake"
            icon      = "✅" if label == "REAL" else "🚨"
            verdict   = "This article appears to be <b>REAL</b>." if label == "REAL" \
                        else "This article appears to be <b>FAKE</b>."

            st.markdown(f'<span class="{badge_cls}">{icon} {label}</span>', unsafe_allow_html=True)

            if confidence is not None:
                bar_color = "#10b981" if label == "REAL" else "#ef4444"
                st.markdown(
                    f'<p class="conf-text">Confidence: <b>{confidence}%</b></p>'
                    f'<div class="conf-bar-bg">'
                    f'<div style="width:{confidence}%;height:10px;background:{bar_color};'
                    f'border-radius:50px;transition:width 0.6s ease;"></div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"<p style='color:#94a3b8;margin-top:1rem;'>{verdict}</p>",
                unsafe_allow_html=True,
            )

            with st.expander("🔬 Cleaned text sent to model"):
                st.code(cleaned, language=None)

            st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#  HOW IT WORKS
# ════════════════════════════════════════════════════════════════
with st.expander("ℹ️  How does this work?"):
    st.markdown(f"""
**Pipeline**

1. **Text cleaning** — lowercase, strip URLs / emails / numbers / punctuation, remove stop-words, lemmatise (NLTK)
2. **TF-IDF vectorisation** — unigrams + bigrams, up to 5,000 features, sublinear TF scaling
3. **Linear SVM (LinearSVC)** — trained on a labelled fake/real news dataset
4. **Confidence** — decision-function distance squashed through a sigmoid

**Auto label detection**

Many public datasets encode `0=REAL, 1=FAKE` (the opposite of the comment in `F6.py`).
This app runs 20 known probe headlines through the model at startup and picks whichever
label assignment achieves the highest accuracy — no manual config needed.

`{detection_note}`

> ⚠️ This is a demonstration model. Always verify news with primary sources.
    """)

st.markdown(
    '<div class="info-box">Model: <b>LinearSVC</b> &nbsp;|&nbsp; '
    'Vectoriser: <b>TF-IDF (5,000 features · 1–2 grams)</b> &nbsp;|&nbsp; '
    f'Label map: <b>{label_map}</b></div>',
    unsafe_allow_html=True,
)
