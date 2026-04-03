import streamlit as st
import requests
import pandas as pd
from streamlit_lottie import st_lottie
from transformers import pipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Fake News Detector System",
    page_icon="🧠",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: #000;
    color: white;
}

.title {
    font-size: 50px;
    text-align: center;
    font-weight: bold;
    color: white;
    text-shadow: 0 0 20px #00ffff;
}

.card {
    background: rgba(255,255,255,0.08);
    padding: 30px;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 40px rgba(0,0,0,0.6);
    margin-top: 20px;
}

.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background: rgba(255,255,255,0.05);
    text-align: center;
    padding: 10px;
    font-size: 16px;
    color: white;
    backdrop-filter: blur(8px);
}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="mrm8488/bert-tiny-finetuned-fake-news-detection"
    )

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# ---------------- TITLE ----------------
st.markdown(
    '<div class="title">🧠 AI Fake News Detector System</div>',
    unsafe_allow_html=True
)

# ---------------- LOTTIE ----------------
def load_lottie(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        return None

brain = load_lottie(
    "https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json"
)

if brain:
    st_lottie(brain, height=250)

# ---------------- USER INPUT ----------------
st.write("### Enter News Text")
news = st.text_area("Paste news headline or article")

# ---------------- DETECTION ----------------
if st.button("🚀 Detect News"):

    if not news.strip():
        st.warning("Please enter some news text first.")
    else:
        try:
            result = model(news)[0]

            label = result["label"]
            score = result["score"]

            st.markdown('<div class="card">', unsafe_allow_html=True)

            if label.upper() == "FAKE":
                st.error(f"❌ Fake News (Confidence: {score:.2f})")
            else:
                st.success(f"✅ Real News (Confidence: {score:.2f})")

            st.markdown('</div>', unsafe_allow_html=True)

            # -------- CONFIDENCE GRAPH --------
            chart_data = pd.DataFrame({
                "Result": ["Prediction", "Opposite"],
                "Confidence": [score, 1 - score]
            })

            st.write("### 📈 AI Confidence Score")
            st.bar_chart(chart_data.set_index("Result"))

            # -------- AI EXPLANATION --------
            st.write("### 🤖 AI Explanation")

            news_lower = news.lower()

            if "breaking" in news_lower or "shocking" in news_lower:
                st.warning(
                    "This news contains sensational words often used in fake news."
                )
            elif len(news) < 50:
                st.warning(
                    "Very short news articles may lack reliable context."
                )
            else:
                st.success(
                    "The language pattern matches authentic journalism style."
                )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------- ANALYTICS DASHBOARD ----------------
st.write("## 📊 Dataset Analytics")

try:
    data = pd.read_csv("news.csv")

    col1, col2 = st.columns(2)

    with col1:
        if "label" in data.columns:
            st.write("### Fake vs Real Distribution")
            st.bar_chart(data["label"].value_counts())

    with col2:
        if "subject" in data.columns:
            st.write("### Top News Categories")
            st.bar_chart(data["subject"].value_counts().head(10))

except Exception:
    st.info("Add 'news.csv' dataset file to show analytics")

# ---------------- LIVE NEWS API ----------------
st.write("## 🌍 Live News Headlines")

API_KEY = "YOUR_NEWS_API_KEY"

try:
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={API_KEY}"
    response = requests.get(url, timeout=10)
    news_data = response.json()

    if "articles" in news_data:
        for article in news_data["articles"][:5]:
            st.write("###", article.get("title", "No title"))
            st.write(article.get("description", "No description"))

except Exception:
    st.info("Live news unavailable right now")

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
🧠 AI Fake News Detector System | Made by <b>Harshit Sharma</b>
</div>
""", unsafe_allow_html=True)
