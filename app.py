"""
VibeCaption — app.py
======================
Streamlit front-end. This file only handles UI + user interaction.
All model logic (BLIP loading, caption generation) lives in model_utils.py.
"""

import io
import time
from datetime import datetime

import streamlit as st
from PIL import Image

# CNN+LSTM वाले फंक्शन्स भी इम्पोर्ट कर लिए हैं
from model_utils import (
    load_blip, generate_caption_blip,
    load_cnn_lstm_assets, generate_caption_cnn_lstm
)

# ------------------------------------------------------------------
# PAGE CONFIG & CUSTOM STYLING (Same as your original code)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="VibeCaption",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    .main { background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }
    .hero { padding: 2.2rem 2rem; border-radius: 20px; background: linear-gradient(120deg, #6a3fd1 0%, #b93fd1 50%, #d13f8f 100%); margin-bottom: 1.8rem; box-shadow: 0 10px 30px rgba(105, 63, 209, 0.35); }
    .hero h1 { font-family: 'Poppins', sans-serif; font-size: 2.6rem; font-weight: 700; color: #ffffff; margin-bottom: 0.3rem; }
    .hero p { font-size: 1.05rem; color: rgba(255,255,255,0.88); margin: 0; }
    .upload-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.09); border-radius: 18px; padding: 1.6rem; margin-bottom: 1.2rem; }
    .caption-box { background: linear-gradient(120deg, rgba(106,63,209,0.18), rgba(209,63,143,0.18)); border-left: 4px solid #b93fd1; border-radius: 12px; padding: 1.2rem 1.4rem; font-family: 'Poppins', sans-serif; font-size: 1.15rem; font-weight: 500; color: #f4eefe; margin-top: 0.8rem; }
    .history-item { background: rgba(255,255,255,0.03); border-radius: 10px; padding: 0.7rem 1rem; margin-bottom: 0.5rem; font-size: 0.9rem; color: rgba(255,255,255,0.85); border: 1px solid rgba(255,255,255,0.06); }
    .stButton>button { background: linear-gradient(120deg, #6a3fd1, #d13f8f); color: white; border: none; border-radius: 10px; padding: 0.55rem 1.6rem; font-weight: 600; transition: transform 0.15s ease; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(209, 63, 143, 0.4); }
    .footer-note { text-align: center; color: rgba(255,255,255,0.4); font-size: 0.82rem; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.08); }
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎨 VibeCaption")
    st.caption("CNN + LSTM theory, BLIP-powered live demo")
    st.markdown("---")
    
    # मॉडल चुनने का ऑप्शन यहाँ ऐड किया है
    st.markdown("#### Choose Model")
    model_choice = st.selectbox(
        "Select the backend model",
        options=["BLIP (Pre-trained)", "CNN + LSTM (Custom Trained)"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("#### Caption style (BLIP only)")
    style_choice = st.radio(
        "Choose a tone for the caption",
        options=["Natural description", "Short & punchy", "Detailed"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("#### Recent captions")
    if st.session_state.history:
        for item in reversed(st.session_state.history[-5:]):
            st.markdown(f"<div class='history-item'>🕒 {item['time']}<br><b>{item['model']}</b>: {item['caption']}</div>", unsafe_allow_html=True)
    else:
        st.caption("Your caption history will show up here.")

st.markdown("""
<div class="hero">
    <h1>VibeCaption 🎨📸</h1>
    <p>Upload any image and watch an AI describe it in natural language — powered by a vision-language transformer.</p>
</div>
""", unsafe_allow_html=True)

STYLE_PREFIX_MAP = {
    "Natural description": "",
    "Short & punchy": "a photo of",
    "Detailed": "a detailed photo showing",
}

# ------------------------------------------------------------------
# MAIN LAYOUT
# ------------------------------------------------------------------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown("#### 📤 Upload an image")
    uploaded_file = st.file_uploader("Drop a JPG or PNG file here", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.caption("Or try one of your own photos — pets, travel shots, food, anything works.")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown("#### ✨ Generated caption")

    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
        st.image(image, use_container_width=True)

        generate_clicked = st.button("Generate caption", type="primary")
        caption = ""

        if generate_clicked:
            if model_choice == "BLIP (Pre-trained)":
                with st.spinner("Looking at your image using BLIP..."):
                    processor, blip_model, device = load_blip()
                    prefix = STYLE_PREFIX_MAP[style_choice]
                    caption = generate_caption_blip(image, processor, blip_model, device, style_prefix=prefix)
            
            else:
                with st.spinner("Looking at your image using CNN+LSTM..."):
                    try:
                        # कस्टम मॉडल फाइल्स को models/ फोल्डर से लोड करना
                        c_model, f_extractor, tokenizer, config = load_cnn_lstm_assets(
                            "models/caption_model.keras",
                            "models/feature_extractor.keras",
                            "models/tokenizer.pkl",
                            "models/config.pkl"
                        )
                        caption = generate_caption_cnn_lstm(image, c_model, f_extractor, tokenizer, config)
                   except Exception as e:
                        st.error(f"असली एरर यह है: {e}")
                        caption = None

            if caption:
                st.markdown(f'<div class="caption-box">"{caption}"</div>', unsafe_allow_html=True)
                st.session_state.history.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "caption": caption,
                    "model": "BLIP" if "BLIP" in model_choice else "CNN+LSTM"
                })

    else:
        st.info("Upload an image on the left to generate a caption.")

    st.markdown('</div>', unsafe_allow_html=True)
