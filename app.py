import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
from newsapi import NewsApiClient
import time

# --- Page Config ---
st.set_page_config(
    page_title="🧠 Fake News Detector",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Floating Icon Background with Animation ---
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #f4f6fc, #e3e9f7);
        overflow-x: hidden;
    }
    .icon-float {
        position: fixed;
        width: 40px;
        height: 40px;
        z-index: 0;
        opacity: 0.08;
        animation: floatAnim 15s infinite ease-in-out;
    }
    .icon1 { top: 10%; left: 5%; animation-delay: 0s; }
    .icon2 { top: 25%; left: 85%; animation-delay: 3s; }
    .icon3 { top: 60%; left: 50%; animation-delay: 6s; }
    .icon4 { top: 80%; left: 10%; animation-delay: 9s; }
    .icon5 { top: 30%; left: 40%; animation-delay: 12s; }

    @keyframes floatAnim {
        0% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(360deg); }
        100% { transform: translateY(0px) rotate(720deg); }
    }
    </style>
    <img src="https://cdn-icons-png.flaticon.com/512/1077/1077012.png" class="icon-float icon1">
    <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" class="icon-float icon2">
    <img src="https://cdn-icons-png.flaticon.com/512/906/906361.png" class="icon-float icon3">
    <img src="https://cdn-icons-png.flaticon.com/512/1828/1828817.png" class="icon-float icon4">
    <img src="https://cdn-icons-png.flaticon.com/512/919/919836.png" class="icon-float icon5">
""", unsafe_allow_html=True)

# --- Custom UI Enhancements ---
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            animation: fadeIn 1s ease-in-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(40px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes glow {
            0%, 100% { box-shadow: 0 0 10px rgba(0, 255, 255, 0.3); }
            50% { box-shadow: 0 0 30px rgba(0, 255, 255, 0.6); }
        }

        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }

        .headline-card {
            background: #0b131f;
            color: #fff;
            border-left: 6px solid;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 1.6rem;
            transition: transform 0.3s ease;
            animation: bounce 2s infinite ease-in-out;
        }

        .stTextArea textarea {
            background: #0b131f !important;
            color: #f5f5f5 !important;
            font-size: 16px;
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid #00c6ff;
        }

        .stButton button {
            background: linear-gradient(to right, #00c6ff, #0072ff);
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 0.6rem 1.4rem;
            border: none;
            transition: transform 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }

        .stButton button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 25px #00c6ff;
        }

        .stRadio > div {
            background-color: #0e1117 !important;
            border-radius: 10px;
            padding: 1rem;
            color: white;
        }

        .stRadio label {
            color: white !important;
        }

        h1, h2, h3 {
            color: #14213d;
            text-shadow: 1px 1px 3px #c3d1ff;
        }

        a {
            color: #00aaff;
        }

        a:hover {
            text-decoration: underline;
        }

        .stMarkdown {
            animation: fadeIn 0.6s ease-in-out;
        }
    </style>
""", unsafe_allow_html=True)
# --- Load Detection Model ---
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    return tokenizer, model

tokenizer, model = load_model()

# --- Load Generator Model ---
@st.cache_resource
def load_generator():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    return tokenizer, model

gen_tokenizer, gen_model = load_generator()

def classify_news(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    conf = probs[0][pred].item()
    label = "Real" if pred == 1 else "Fake"
    return label, conf

def generate_fake_news(prompt, max_length=300):
    inputs = gen_tokenizer(prompt, return_tensors="pt")
    outputs = gen_model.generate(**inputs, max_length=max_length, do_sample=True, top_k=50, top_p=0.95)
    return gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- News API ---
try:
    newsapi = NewsApiClient(api_key="25c89f7110de48898f6a8839a0016c8c")
except Exception as e:
    st.error(f"❌ NewsAPI Error: {e}")
    st.stop()

# --- Sidebar Info ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2920/2920254.png", width=90)
    st.title("🧠 About the App")
    st.info("Check if news is Fake or Real using BERT + GPT-2.")
    st.markdown("🌍 Real-time headlines")
    st.markdown("🧬 Fake news generation")
    st.caption("✨ Built with Streamlit & HuggingFace")

# --- Main Section ---
st.title("🧠 Fake News Detector")
st.subheader("⚖️ Instantly verify if a news article is **Real** or **Fake**")

mode = st.radio(
    "🧪 Choose a Detection Mode",
    ["📝 Manual News Check", "🌐 Real-Time Headlines", "🧬 Generate Fake News"],
    horizontal=True
)

# --- Manual Detection ---
if mode == "📝 Manual News Check":
    st.markdown("### ✍️ Paste your News or Headline")
    text = st.text_area("Your news article goes here...", height=160)

    if st.button("🚀 Detect"):
        if text.strip():
            with st.spinner("🔎 Analyzing..."):
                time.sleep(1)
                label, confidence = classify_news(text)
            color = "limegreen" if label == "Real" else "crimson"
            icon = "✅" if label == "Real" else "❌"
            st.markdown(f"""
                <div class="headline-card" style="border-left-color: {color};">
                    <h3>{icon} Result: <span style='color:{color}'>{label}</span></h3>
                    <p><b>Confidence Score:</b> {confidence:.2f}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter content to check.")

# --- Real-Time Headlines ---
elif mode == "🌐 Real-Time Headlines":
    st.markdown("### 📰 Top Headlines (Live from NewsAPI.org)")
    try:
        articles = newsapi.get_top_headlines(language="en", page_size=6)
        for article in articles['articles']:
            title = article.get("title", "")
            description = article.get("description", "")
            url = article.get("url", "#")
            content = f"{title} {description}"
            label, confidence = classify_news(content)
            color = "limegreen" if label == "Real" else "crimson"
            st.markdown(f"""
                <div class="headline-card" style="border-left-color:{color};">
                    <h4><a href="{url}" target="_blank">📰 {title}</a></h4>
                    <p>{description}</p>
                    <b style="color:{color}">Prediction: {label} ({confidence:.2f})</b>
                </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Error fetching headlines: {e}")

# --- Fake News Generator ---
elif mode == "🧬 Generate Fake News":
    st.markdown("### 🧪 Enter a Topic to Generate Fake News")
    prompt = st.text_area("Topic or headline idea...", height=120)

    if st.button("🎭 Generate"):
        if prompt.strip():
            with st.spinner("🌀 Generating fake news..."):
                time.sleep(1)
                fake_news = generate_fake_news(prompt)
                label, confidence = classify_news(fake_news)
            color = "crimson" if label == "Fake" else "limegreen"
            st.markdown(f"""
                <div class="headline-card" style="border-left-color:{color};">
                    <h4>🧬 Generated Article</h4>
                    <p>{fake_news}</p>
                    <b style="color:{color}">Prediction: {label} ({confidence:.2f})</b>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter a topic or headline to generate.")




