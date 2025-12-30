import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import pandas as pd
import plotly.express as px
from gtts import gTTS
import tempfile
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide"
)

# ===============================
# LOAD MODEL (SAFE)
# ===============================
@st.cache_resource
def load_model():
    if not os.path.exists("best_model.h5"):
        st.error("❌ best_model.h5 not found")
        st.stop()
    return tf.keras.models.load_model("best_model.h5", compile=False)

model = load_model()

# ===============================
# 43 CLASS NAMES (EXACT)
# ===============================
CLASSES = {
0:'Speed limit (20km/h)',1:'Speed limit (30km/h)',2:'Speed limit (50km/h)',
3:'Speed limit (60km/h)',4:'Speed limit (70km/h)',5:'Speed limit (80km/h)',
6:'End of speed limit (80km/h)',7:'Speed limit (100km/h)',8:'Speed limit (120km/h)',
9:'No passing',10:'No passing veh over 3.5 tons',11:'Right-of-way at intersection',
12:'Priority road',13:'Yield',14:'Stop',15:'No vehicles',
16:'Veh > 3.5 tons prohibited',17:'No entry',18:'General caution',
19:'Dangerous curve left',20:'Dangerous curve right',21:'Double curve',
22:'Bumpy road',23:'Slippery road',24:'Road narrows on the right',
25:'Road work',26:'Traffic signals',27:'Pedestrians',28:'Children crossing',
29:'Bicycles crossing',30:'Beware of ice/snow',31:'Wild animals crossing',
32:'End speed + passing limits',33:'Turn right ahead',34:'Turn left ahead',
35:'Ahead only',36:'Go straight or right',37:'Go straight or left',
38:'Keep right',39:'Keep left',40:'Roundabout mandatory',
41:'End of no passing',42:'End no passing veh > 3.5 tons'
}

# ===============================
# MULTI-LANGUAGE MEANINGS
# ===============================
MEANINGS = {
    k: {
        "en": v,
        "ta": "போக்குவரத்து அடையாளம் : " + v,
        "hi": "यातायात संकेत : " + v
    } for k, v in CLASSES.items()
}

# ===============================
# SMART DIGIT-FOCUSED CROP
# ===============================
def smart_crop(img):
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        return img[y:y+h, x:x+w]
    return img

# ===============================
# PREPROCESS
# ===============================
def preprocess(img):
    img = cv2.resize(img, (32,32))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# ===============================
# GRAD-CAM (WORKING)
# ===============================
def gradcam(img_array, model):
    last_conv = [l for l in model.layers if "conv" in l.name][-1]
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

# ===============================
# VOICE (SAFE)
# ===============================
def speak(text, lang):
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        tts.save(f.name)
        return f.name

# ===============================
# UI
# ===============================
st.title("🚦 Traffic Sign Recognition System")

uploaded = st.file_uploader("Upload Traffic Sign Image", type=["jpg","png","jpeg"])

lang = st.selectbox("Voice Language", ["English","Tamil","Hindi"])
lang_code = {"English":"en","Tamil":"ta","Hindi":"hi"}[lang]

top_k = st.slider("Top-K Predictions", 1, 5, 3)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    crop = smart_crop(img)
    st.image([img, crop], caption=["Original", "Cropped"], width=250)

    arr = preprocess(crop)
    preds = model.predict(arr)[0]

    top_idx = np.argsort(preds)[::-1][:top_k]
    results = [(i, CLASSES[i], preds[i]*100) for i in top_idx]

    df = pd.DataFrame(results, columns=["Class ID","Sign","Confidence (%)"])
    st.subheader("🔍 Predictions")
    st.dataframe(df)

    fig = px.bar(df, x="Confidence (%)", y="Sign", orientation="h")
    st.plotly_chart(fig, use_container_width=True)

    # Meaning + Voice
    best = top_idx[0]
    meaning = MEANINGS[best][lang_code]
    st.success(meaning)

    audio = speak(meaning, lang_code)
    st.audio(audio)

    # Grad-CAM
    heatmap = gradcam(arr, model)
    heatmap = cv2.resize(heatmap, crop.shape[:2][::-1])
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(crop, 0.6, heatmap, 0.4, 0)
    st.subheader("🧠 Grad-CAM")
    st.image(overlay, use_container_width=True)
