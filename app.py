"""
TruthLens — Fake News Detector with GenAI Explanations

Streamlit app: paste a news article, get an ML-based Real/Fake prediction
with confidence, plus a GenAI-generated plain-English explanation of the
reasoning. The ML model makes the call; the LLM explains it — it does not
re-decide real/fake itself.
"""
import os

import joblib
import streamlit as st
from groq import Groq

from src.preprocess import clean_text
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="TruthLens", page_icon="🔍", layout="centered")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


@st.cache_resource
def load_model():
    model_path = os.path.join(MODELS_DIR, "model.joblib")
    vec_path = os.path.join(MODELS_DIR, "vectorizer.joblib")
    if not (os.path.exists(model_path) and os.path.exists(vec_path)):
        return None, None
    return joblib.load(model_path), joblib.load(vec_path)


def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def explain_with_genai(client, article_text, label, confidence):
    prompt = f"""You are a media literacy assistant. A machine learning model has
analyzed a news article and classified it as "{label}" with {confidence:.0%} confidence.

Your job is NOT to re-decide whether it's real or fake — the ML model already made that
call. Instead, explain WHY the article might look that way, in plain, neutral language.
Cover:
1. Any emotionally loaded or sensational language you notice
2. Any claims that seem unverified or would need fact-checking
3. Overall tone/style observations (e.g. use of ALL CAPS, excessive punctuation, vague sourcing)
4. One or two concrete things a reader should fact-check before sharing this

Keep it concise (150-200 words), neutral in tone, and educational rather than alarmist.

Article:
{article_text[:3000]}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=400,
    )
    return response.choices[0].message.content


# --- UI ---
st.title("🔍 TruthLens")
st.caption("ML model predicts real vs. fake — GenAI explains the reasoning.")

model, vectorizer = load_model()

if model is None:
    st.warning(
        "No trained model found yet. Run `python src/train_model.py` from the "
        "project root first (see README.md for dataset setup)."
    )
    st.stop()

groq_client = get_groq_client()
if groq_client is None:
    st.info(
        "GROQ_API_KEY not set — predictions will still work, but the GenAI "
        "explanation will be skipped. Set the environment variable and restart "
        "to enable it (see README.md)."
    )

article_text = st.text_area(
    "Paste a news article (or a snippet) below:",
    height=250,
    placeholder="Paste article text here...",
)

analyze = st.button("Analyze", type="primary")

if analyze:
    if not article_text.strip():
        st.error("Please paste some article text first.")
    else:
        cleaned = clean_text(article_text)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        label = "Real" if pred == 1 else "Fake"
        confidence = proba[pred]

        col1, col2 = st.columns(2)
        with col1:
            if label == "Real":
                st.success(f"Prediction: **{label}**")
            else:
                st.error(f"Prediction: **{label}**")
        with col2:
            st.metric("Confidence", f"{confidence:.1%}")

        st.progress(float(confidence))

        st.divider()
        st.subheader("🤖 GenAI Explanation")

        if groq_client:
            with st.spinner("Generating explanation..."):
                try:
                    explanation = explain_with_genai(groq_client, article_text, label, confidence)
                    st.write(explanation)
                except Exception as e:
                    st.error(f"Could not generate explanation: {e}")
        else:
            st.caption("Set GROQ_API_KEY to enable this feature.")

        st.divider()
        st.caption(
            "⚠️ This tool is an educational aid, not a fact-checking authority. "
            "Always verify important claims through trusted primary sources."
        )

st.sidebar.header("About")
st.sidebar.write(
    "TruthLens combines a TF-IDF + Logistic Regression classifier "
    "(trained on labeled news data) with an LLM explanation layer. "
    "The ML model decides real vs. fake; the GenAI layer explains why."
)
st.sidebar.write("Built for the Data Science & ML with GenAI final project.")
