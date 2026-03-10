import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
from transformers import pipeline
import matplotlib
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Fake News Detector System", page_icon="🧠", layout="wide")

# ---------------- BERT MODEL ----------------

@st.cache_resource
def load_model():
    classifier = pipeline("text-classification",
                          model="mrm8488/bert-tiny-finetuned-fake-news-detection")
    return classifier

model = load_model()

# ---------------- 3D BACKGROUND ----------------

st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
background:#000;
color:white;
}

#bg{
position:fixed;
top:0;
left:0;
width:100%;
height:100%;
z-index:-1;
}

.title{
font-size:50px;
text-align:center;
font-weight:bold;
color:white;
text-shadow:0 0 20px #00ffff;
}

.card{
background:rgba(255,255,255,0.08);
padding:30px;
border-radius:20px;
backdrop-filter:blur(12px);
box-shadow:0 10px 40px rgba(0,0,0,0.6);
margin-top:20px;
}

</style>

<canvas id="bg"></canvas>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>

<script>

const canvas=document.getElementById('bg');

const scene=new THREE.Scene();

const camera=new THREE.PerspectiveCamera(
75,
window.innerWidth/window.innerHeight,
0.1,
1000
);

const renderer=new THREE.WebGLRenderer({canvas:canvas});

renderer.setSize(window.innerWidth,window.innerHeight);

const geometry=new THREE.BufferGeometry();

const vertices=[];

for(let i=0;i<6000;i++){

vertices.push(
THREE.MathUtils.randFloatSpread(2000),
THREE.MathUtils.randFloatSpread(2000),
THREE.MathUtils.randFloatSpread(2000)
);

}

geometry.setAttribute(
'position',
new THREE.Float32BufferAttribute(vertices,3)
);

const material=new THREE.PointsMaterial({
color:0x00ffff,
size:2
});

const particles=new THREE.Points(geometry,material);

scene.add(particles);

camera.position.z=400;

function animate(){

requestAnimationFrame(animate);

particles.rotation.x+=0.0005;
particles.rotation.y+=0.001;

renderer.render(scene,camera);

}

animate();

</script>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------

st.markdown('<div class="title">🧠 AI Fake News Detector System</div>', unsafe_allow_html=True)

# ---------------- LOTTIE BRAIN ----------------

def load_lottie(url):
    r=requests.get(url)
    return r.json()

brain=load_lottie("https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json")

st_lottie(brain,height=250)

# ---------------- USER INPUT ----------------

st.write("### Enter News Text")

news = st.text_area("Paste news headline or article")

# ---------------- DETECTION ----------------

if st.button("🚀 Detect News"):

    result = model(news)[0]

    label = result['label']
    score = result['score']

    st.markdown('<div class="card">', unsafe_allow_html=True)

    if label=="FAKE":

        st.error(f"❌ Fake News (Confidence {score:.2f})")

    else:

        st.success(f"✅ Real News (Confidence {score:.2f})")

    st.markdown('</div>', unsafe_allow_html=True)

    # -------- CONFIDENCE GRAPH --------

    labels=["Fake","Real"]
    values=[score,1-score]

    fig,ax=plt.subplots()

    ax.bar(labels,values)

    ax.set_title("AI Confidence Score")

    st.pyplot(fig)

    # -------- AI EXPLANATION --------

    st.write("### 🤖 AI Explanation")

    if "breaking" in news.lower() or "shocking" in news.lower():

        st.warning("This news contains sensational words often used in fake news.")

    elif len(news)<50:

        st.warning("Very short news articles may lack reliable context.")

    else:

        st.success("The language pattern matches authentic journalism style.")

# ---------------- ANALYTICS DASHBOARD ----------------

st.write("## 📊 Dataset Analytics")

try:

    data=pd.read_csv("news.csv")

    col1,col2=st.columns(2)

    with col1:

        fig,ax=plt.subplots()

        data['label'].value_counts().plot(kind="bar",ax=ax)

        ax.set_title("Fake vs Real Distribution")

        st.pyplot(fig)

    with col2:

        if 'subject' in data.columns:

            fig,ax=plt.subplots()

            data['subject'].value_counts().head(10).plot(kind="bar",ax=ax)

            ax.set_title("Top News Categories")

            st.pyplot(fig)

except:

    st.info("Add dataset file to show analytics")

# ---------------- LIVE NEWS API ----------------

st.write("## 🌍 Live News Headlines")

API_KEY="c07a8025d65443e28180d3b20539838b"

try:

    url=f"https://newsapi.org/v2/top-headlines?country=us&apiKey={API_KEY}"

    news_data=requests.get(url).json()

    for article in news_data["articles"][:5]:

        st.write("###",article["title"])

        st.write(article["description"])

except:

    st.info("c07a8025d65443e28180d3b20539838b")
# ---------------- FOOTER ----------------

st.markdown("""
<style>
.footer{
position:fixed;
left:0;
bottom:0;
width:100%;
background:rgba(255,255,255,0.05);
text-align:center;
padding:10px;
font-size:16px;
color:white;
backdrop-filter:blur(8px);
}
</style>

<div class="footer">
🧠 AI Fake News Detector System| Made by <b>Harshit Sharma</b>
</div>

""", unsafe_allow_html=True)
