import base64
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---------------------------------------------------------------- page setup
st.set_page_config(page_title="Hurricane Damage Detector", page_icon="🌀", layout="centered")

HERE = Path(__file__).parent
MODEL_PATH = HERE / "hurricane_resnet18.pth"   # <- your model file
CLASS_NAMES = ["Damage", "No damage"]           # index 0 and 1, same as training
BG_PATH = HERE / "background.jpg"

# Same preprocessing as test_tf in training (no noise at inference)
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)               # architecture only
    model.fc = nn.Linear(model.fc.in_features, 2)       # same head as training
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


@st.cache_data
def background_url():
    """Embed the background photo as base64 so the CSS can use it."""
    if not BG_PATH.exists():
        return "none"
    b64 = base64.b64encode(BG_PATH.read_bytes()).decode()
    return f'url("data:image/jpeg;base64,{b64}")'


# ---------------------------------------------------------------- styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap');

/* full-page hurricane background with a dark gradient on top */
[data-testid="stAppViewContainer"] {
  background-color: #050b14;
  background-image:
    linear-gradient(180deg, rgba(5,11,20,.50) 0%, rgba(5,11,20,.80) 50%, rgba(5,11,20,.95) 100%),
    __BG__;
  background-size: cover;
  background-position: center;
  background-attachment: fixed;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppDeployButton"], [data-testid="stDecoration"], footer { display: none; }
/* hide the top-right toolbar (GitHub / Fork icons on Streamlit Cloud) */
[data-testid="stToolbar"], [data-testid="stToolbarActions"], .stToolbarActions { display: none !important; }

.stApp { font-family: 'Inter', sans-serif; color: #e8f1fa; }
.block-container { max-width: 780px; padding-top: 4.5rem; padding-bottom: 3rem; }

/* hero */
.hero { text-align: center; margin-bottom: 2.2rem; }
.badge {
  display: inline-block; padding: .35rem .95rem; border-radius: 999px;
  background: rgba(90,180,255,.12); border: 1px solid rgba(120,195,255,.35);
  color: #9fd4ff; font-size: .75rem; font-weight: 600;
  letter-spacing: .12em; text-transform: uppercase;
}
.title {
  font-family: 'Space Grotesk', sans-serif; font-weight: 700;
  font-size: clamp(2.2rem, 6vw, 3.4rem); line-height: 1.05; margin: .9rem 0 .7rem;
  background: linear-gradient(90deg, #ffffff, #9fd4ff 45%, #3a8dff);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.subtitle { color: rgba(232,241,250,.75); font-size: 1.05rem; max-width: 540px; margin: 0 auto; }

/* upload box as a frosted-glass card */
[data-testid="stFileUploaderDropzone"] {
  background: rgba(255,255,255,.06);
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  border: 1.5px dashed rgba(140,200,255,.45); border-radius: 18px;
  padding: 1.8rem 1.5rem; transition: all .25s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: #5ab4ff; background: rgba(90,180,255,.08);
}
[data-testid="stFileUploaderDropzone"] button {
  background: linear-gradient(135deg, #5ab4ff, #2f6fe0);
  color: #fff; border: none; border-radius: 10px; font-weight: 600;
}
[data-testid="stFileUploaderDropzone"] small { color: rgba(232,241,250,.6); }

/* three little "how it works" steps */
.steps { display: flex; gap: .8rem; margin-top: 1.4rem; }
.step {
  flex: 1; padding: 1rem; border-radius: 14px; text-align: center;
  background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.1);
  backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
}
.step-num { font-family: 'Space Grotesk', sans-serif; color: #5ab4ff; font-weight: 700; font-size: 1.1rem; }
.step-text { color: rgba(232,241,250,.75); font-size: .85rem; margin-top: .2rem; }

/* uploaded image */
[data-testid="stImage"] img { border-radius: 16px; box-shadow: 0 12px 40px rgba(0,0,0,.5); }

/* result card */
.result {
  padding: 1.6rem; border-radius: 18px;
  background: rgba(8,16,28,.70);
  backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(255,255,255,.12);
}
.result.damage { border-color: rgba(255,92,92,.55); box-shadow: 0 0 40px rgba(255,92,92,.22); }
.result.safe { border-color: rgba(61,220,132,.45); box-shadow: 0 0 40px rgba(61,220,132,.15); }
.result-icon { font-size: 2.4rem; }
.result-label { font-family: 'Space Grotesk', sans-serif; font-size: 1.7rem; font-weight: 700; margin-top: .3rem; }
.result.damage .result-label { color: #ff6b6b; }
.result.safe .result-label { color: #3ddc84; }
.result-conf { color: rgba(232,241,250,.7); margin-bottom: 1.3rem; }
.meter { height: 10px; border-radius: 999px; background: rgba(255,255,255,.1); overflow: hidden; }
.meter-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #9fd4ff, #ffb347, #ff5c5c); }
.meter-caption { display: flex; justify-content: space-between; font-size: .85rem; color: rgba(232,241,250,.7); margin-top: .5rem; }

.footer { text-align: center; color: rgba(232,241,250,.45); font-size: .8rem; margin-top: 3rem; }
</style>
"""
st.markdown(CSS.replace("__BG__", background_url()), unsafe_allow_html=True)

# ---------------------------------------------------------------- content
model = load_model()

st.markdown(
    '<div class="hero">'
    '<span class="badge">🌀 AI damage assessment</span>'
    '<div class="title">Hurricane Damage Detector</div>'
    '<p class="subtitle">Upload a satellite image taken after a hurricane and a deep-learning '
    'model tells you whether the area shows damage.</p>'
    '</div>',
    unsafe_allow_html=True,
)

file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"],
                        label_visibility="collapsed")

if file is None:
    st.markdown(
        '<div class="steps">'
        '<div class="step"><div class="step-num">01</div><div class="step-text">Upload an image</div></div>'
        '<div class="step"><div class="step-num">02</div><div class="step-text">The model analyzes it</div></div>'
        '<div class="step"><div class="step-num">03</div><div class="step-text">Get the verdict</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    img = Image.open(file).convert("RGB")

    with st.spinner("Analyzing image…"):
        x = preprocess(img).unsqueeze(0)                 # shape (1, 3, 224, 224)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]

    pred = int(probs.argmax())
    conf = float(probs[pred])
    p_damage = float(probs[0])                           # class 0 = damage

    if pred == 0:
        kind, icon, label = "damage", "⚠️", "Damage detected"
    else:
        kind, icon, label = "safe", "🏠", "No damage detected"

    col_img, col_result = st.columns([1.1, 1], gap="large")
    with col_img:
        st.image(img, width="stretch")
    with col_result:
        st.markdown(
            f'<div class="result {kind}">'
            f'<div class="result-icon">{icon}</div>'
            f'<div class="result-label">{label}</div>'
            f'<div class="result-conf">{conf:.1%} confidence</div>'
            f'<div class="meter"><div class="meter-fill" style="width:{p_damage * 100:.1f}%"></div></div>'
            f'<div class="meter-caption"><span>Damage probability</span><span>{p_damage:.1%}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="footer">Fine-tuned ResNet-18 · for research use, not an official damage assessment</div>',
    unsafe_allow_html=True,
)
