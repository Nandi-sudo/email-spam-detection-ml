from __future__ import annotations

import json
from pathlib import Path

import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "spam_classifier.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

st.set_page_config(page_title="Email Spam Detector", page_icon="📧", layout="centered")
st.title("📧 Email Spam Detector")
st.write("Check whether an email looks like **spam** or **ham** (legitimate email).")
st.caption("Demo machine-learning classifier trained on the Apache SpamAssassin Public Corpus.")

if not MODEL_PATH.exists():
    st.info("The model has not been trained yet. Training downloads the SpamAssassin sample corpus.")
    if st.button("Train model", type="primary"):
        with st.spinner("Downloading emails and training the model..."):
            try:
                from train import train_model

                result = train_model()
                st.success(f"Training complete — test accuracy: {result['accuracy']:.1%}")
                st.rerun()
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                st.caption("Check your internet connection and try again.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
threshold = st.slider(
    "Spam classification threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
    help="A message is labeled spam when its predicted spam probability is at least this value.",
)

with st.form("email_form"):
    subject = st.text_input("Email subject", placeholder="e.g. Your account update")
    body = st.text_area("Email content", height=220, placeholder="Paste the email text here...")
    submitted = st.form_submit_button("Check email", type="primary")

if submitted:
    message = "\n".join(part for part in (subject.strip(), body.strip()) if part)
    if not message:
        st.warning("Enter an email subject or some email content first.")
    else:
        probabilities = model.predict_proba([message])[0]
        classes = list(model.classes_)
        spam_probability = float(probabilities[classes.index("spam")])
        if spam_probability >= threshold:
            st.error(f"🚨 Likely SPAM — {spam_probability:.1%} spam probability")
        else:
            st.success(f"✅ Likely HAM — {1 - spam_probability:.1%} ham probability")
        st.progress(spam_probability, text=f"Spam probability: {spam_probability:.1%}")
        st.caption("This is a machine-learning estimate, not a guarantee. Avoid pasting sensitive email content into demos.")

if METRICS_PATH.exists():
    with METRICS_PATH.open(encoding="utf-8") as file:
        metrics = json.load(file)
    st.divider()
    st.subheader("Model information")
    first, second = st.columns(2)
    first.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
    second.metric("Emails used", f"{metrics['total_messages']:,}")
    st.caption(f"Dataset: {metrics['dataset']} · Trained: {metrics['trained_at_utc']}")
