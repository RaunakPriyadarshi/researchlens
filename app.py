import re
from pathlib import Path

import joblib
import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(
    page_title="ResearchLens | Abstract Classifier",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f6f8fc; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px; }
.hero {
  padding: 2rem 2.2rem; border-radius: 24px; color: white;
  background: linear-gradient(120deg, #111b3c 0%, #244d83 60%, #3d8bba 100%);
  box-shadow: 0 16px 42px rgba(21, 46, 91, .18); margin-bottom: 1.4rem;
}
.hero h1 { font-family: 'Manrope', sans-serif; font-size: 2.3rem; line-height: 1.15; margin: .25rem 0 .6rem; color: white; }
.hero p { color: #e0eafa; font-size: 1.02rem; margin-bottom: .25rem; }
.eyebrow { text-transform: uppercase; letter-spacing: .16em; font-size: .72rem; font-weight: 700; color: #b9d9ff; }
.panel {
  background: white; border: 1px solid #e4eaf3; border-radius: 18px;
  padding: 1.25rem 1.35rem; box-shadow: 0 5px 18px rgba(23, 42, 77, .04);
}
.result {
  border-radius: 18px; padding: 1.35rem; background: #eaf7f1;
  border: 1px solid #c9eadb; color: #14583e;
}
.small-muted { color: #62708a; font-size: .88rem; }
div[data-testid="stMetric"] { background: white; border: 1px solid #e4eaf3; padding: 1rem; border-radius: 15px; }
.stButton > button, .stDownloadButton > button { border-radius: 10px; min-height: 2.8rem; font-weight: 650; }
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Keep the preprocessing aligned with the original Colab notebook.
@st.cache_resource
def prepare_nlp():
    for package in ("stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.download(package, quiet=True)
        except Exception:
            pass
    try:
        stop = set(stopwords.words("english"))
    except LookupError as exc:
        raise RuntimeError(
            "NLTK stopwords could not be downloaded. Check the app's internet access "
            "or deploy the NLTK data with the app."
        ) from exc
    stop |= {
        "study", "paper", "article", "research", "result", "results", "show", "using", "used", "use",
        "based", "also", "however", "within", "among", "may", "can", "et", "al", "propose", "proposed",
        "present", "approach", "analysis", "finding", "findings", "purpose", "methods", "method",
        "conclusion", "conclusions", "background", "objective", "objectives", "abstract",
        "motivation", "availability", "implementation", "design", "setting", "patients", "participants",
        "introduction", "aim", "aims", "novel", "new", "one", "two", "three", "different", "several"
    }
    return stop, WordNetLemmatizer()

def strip_copyright(text):
    text = re.sub(r"©.*$", "", text)
    text = re.sub(r"(?i)\bcopyright\b.*$", "", text)
    text = re.sub(r"(?i)\ball rights reserved\.?", "", text)
    return text.strip()

def clean_text(text, stop, lemma):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [
        lemma.lemmatize(token)
        for token in text.split()
        if len(token) > 2 and token not in stop
    ]
    return " ".join(tokens)

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "best_model.joblib"
    if not model_path.exists():
        return None
    return joblib.load(model_path)

stop, lemmatizer = prepare_nlp()
model = load_model()

with st.sidebar:
    st.markdown("## 📚 ResearchLens")
    st.caption("Research abstract classification")
    st.divider()
    page = st.radio(
        "Workspace",
        ["Single abstract", "Batch CSV", "About the model"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("**Prediction classes**")
    st.markdown("🔬 Science")
    st.markdown("💼 Business")
    st.markdown("👥 Social Science")
    st.divider()
    st.caption("Built with Python · scikit-learn · TF-IDF · Streamlit")

st.markdown("""
<div class="hero">
  <div class="eyebrow">NLP · MACHINE LEARNING</div>
  <h1>ResearchLens</h1>
  <p>Turn research abstracts into useful subject-area insights.</p>
  <p>Classify an individual abstract or process a complete CSV file in seconds.</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.warning(
        "The trained model file is missing. In Google Colab, download "
        "`best_model.joblib` from the Files panel, then place it beside `app.py` "
        "in this project folder. See README.md for the exact steps."
    )
    st.stop()

if page == "Single abstract":
    left, right = st.columns([1.25, 0.75], gap="large")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Classify a research abstract")
        st.write("Paste the abstract below. The model will predict one of three subject areas.")
        abstract = st.text_area(
            "Research abstract",
            height=260,
            placeholder="Paste the full research abstract here…",
            help="For best results, use a complete abstract rather than a title or a few keywords.",
        )
        submitted = st.button("✨ Classify abstract", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("How it works")
        st.markdown("**01 · Clean**")
        st.caption("Normalize text, remove noise and apply the same preprocessing as training.")
        st.markdown("**02 · Vectorize**")
        st.caption("TF-IDF turns words and two-word phrases into numerical features.")
        st.markdown("**03 · Predict**")
        st.caption("The saved classifier chooses Science, Business or Social Science.")
        st.info("Predictions are estimates. Review the abstract and result before relying on the label.")
        st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        if len(abstract.strip()) < 30:
            st.error("Please enter a longer abstract (at least 30 characters; a complete abstract is recommended).")
        else:
            cleaned = clean_text(strip_copyright(abstract), stop, lemmatizer)
            if not cleaned.strip():
                st.error("No usable text remained after preprocessing. Try another abstract.")
            else:
                pred = model.predict([cleaned])[0]
                st.markdown("---")
                st.markdown('<div class="result">', unsafe_allow_html=True)
                st.caption("PREDICTED SUBJECT AREA")
                st.markdown(f"## {pred}")
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba([cleaned])[0]
                    classes = model.classes_
                    confidence = float(probabilities.max())
                    st.progress(confidence, text=f"Highest model probability: {confidence:.1%}")
                    st.caption("Model probability is not a guarantee that the prediction is correct.")
                st.markdown("</div>", unsafe_allow_html=True)

elif page == "Batch CSV":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Classify multiple abstracts")
    st.write("Upload a CSV with a column named `Abstract` or `text`. Other columns are retained.")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded, encoding="utf-8-sig")
            text_col = next(
                (col for col in data.columns if str(col).strip().lower() in ("abstract", "text")),
                None,
            )
            if text_col is None:
                st.error(f"No Abstract/text column found. Available columns: {', '.join(map(str, data.columns))}")
            elif data.empty:
                st.warning("The uploaded CSV contains no rows.")
            else:
                st.write(f"Loaded **{len(data):,}** rows.")
                with st.spinner("Classifying abstracts…"):
                    cleaned = data[text_col].fillna("").astype(str).apply(
                        lambda item: clean_text(strip_copyright(item), stop, lemmatizer)
                    )
                    data["predicted_label"] = model.predict(cleaned)
                    if hasattr(model, "predict_proba"):
                        data["confidence"] = model.predict_proba(cleaned).max(axis=1).round(3)
                counts = data["predicted_label"].value_counts()
                metric_cols = st.columns(3)
                for idx, label in enumerate(["Science", "Business", "Social Science"]):
                    metric_cols[idx].metric(label, int(counts.get(label, 0)))
                st.subheader("Preview")
                st.dataframe(data.head(100), use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download predictions CSV",
                    data=data.to_csv(index=False).encode("utf-8-sig"),
                    file_name="research_paper_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        except Exception as exc:
            st.error(f"Could not process this CSV: {exc}")

else:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("About this project")
    st.write(
        "ResearchLens is a supervised multiclass text-classification project. "
        "The original notebook collects Scopus abstracts for Science, Business and Social Science, "
        "cleans and balances the labelled data, and compares six machine-learning classifiers."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Classes", "3")
    c2.metric("Training approach", "TF-IDF")
    c3.metric("Selected model", "Logistic Regression")
    st.markdown("#### Pipeline")
    st.markdown("1. Text cleaning and lemmatization\n2. TF-IDF features using unigrams and bigrams\n3. Logistic Regression classification\n4. Predicted label and optional class probability")
    st.markdown("#### Notes")
    st.write(
        "The notebook's saved model was trained on the labelled dataset. This website uses that saved model; "
        "it does not retrain the six classifiers when a visitor opens the page."
    )
    st.write(
        "The notebook recorded 83.57% accuracy for Logistic Regression on its held-out test set. "
        "This is a past evaluation result, not a guarantee of accuracy on new papers."
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("ResearchLens · Research Paper Classification · Predictions should be checked against the original paper.")
