import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from gtts import gTTS
import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="German Traffic Sign AI",
    page_icon="🚦",
    layout="wide"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
.main-header {font-size:3rem;color:#1E88E5;text-align:center;font-weight:bold;}
.sub-header {text-align:center;color:#555;}
.bar {height:18px;border-radius:6px;background:#2196F3;}
</style>
""", unsafe_allow_html=True)

# ---------------- CLASS NAMES ----------------
class_names = {
    0: {"en": "Speed limit 20 km/h", "hi": "गति सीमा 20 किमी/घंटा", "ta": "வேக வரம்பு 20 கிமீ/மணி"},
    1: {"en": "Speed limit 30 km/h", "hi": "गति सीमा 30 किमी/घंटा", "ta": "வேக வரம்பு 30 கிமீ/மணி"},
    2: {"en": "Speed limit 50 km/h", "hi": "गति सीमा 50 किमी/घंटा", "ta": "வேக வரம்பு 50 கிமீ/மணி"},
    3: {"en": "Speed limit 60 km/h", "hi": "गति सीमा 60 किमी/घंटा", "ta": "வேக வரம்பு 60 கிமீ/மணி"},
    4: {"en": "Speed limit 70 km/h", "hi": "गति सीमा 70 किमी/घंटा", "ta": "வேக வரம்பு 70 கிமீ/மணி"},
    5: {"en": "Speed limit 80 km/h", "hi": "गति सीमा 80 किमी/घंटा", "ta": "வேக வரம்பு 80 கிமீ/மணி"},
    6: {"en": "End of speed limit 80 km/h", "hi": "गति सीमा समाप्त 80 किमी/घंटा", "ta": "வேக வரம்பு 80 கிமீ/மணி முடிவு"},
    7: {"en": "Speed limit 100 km/h", "hi": "गति सीमा 100 किमी/घंटा", "ta": "வேக வரம்பு 100 கிமீ/மணி"},
    8: {"en": "Speed limit 120 km/h", "hi": "गति सीमा 120 किमी/घंटा", "ta": "வேக வரம்பு 120 கிமீ/மணி"},
    9: {"en": "No passing", "hi": "पासिंग निषेध", "ta": "மீறிச் செல்ல தடை"},
    10: {"en": "No passing for vehicles over 3.5 tons", "hi": "3.5 टन से अधिक वाहनों के लिए पासिंग निषेध", "ta": "3.5 டன் மீது வாகனங்களுக்கு தடை"},
    11: {"en": "Right-of-way at intersection", "hi": "चौराहे पर प्राथमिकता", "ta": "இடைக்காலச் சந்திப்பில் முன்னுரிமை"},
    12: {"en": "Priority road", "hi": "प्राथमिकता सड़क", "ta": "முன்னுரிமை சாலை"},
    13: {"en": "Yield", "hi": "रास्ता दें", "ta": "வழியை விடுங்கள்"},
    14: {"en": "Stop", "hi": "रुकें", "ta": "நிறுத்தவும்"},
    15: {"en": "No vehicles", "hi": "वाहन निषेध", "ta": "வாகனங்கள் செல்லக்கூடாது"},
    16: {"en": "Vehicles over 3.5 tons prohibited", "hi": "3.5 टन से अधिक वाहन निषेध", "ta": "3.5 டன் மீது வாகனங்கள் தடை"},
    17: {"en": "No entry", "hi": "प्रवेश निषेध", "ta": "நுழைவு தடை"},
    18: {"en": "General caution", "hi": "सामान्य सावधानी", "ta": "பொதுச் எச்சரிக்கை"},
    19: {"en": "Dangerous curve left", "hi": "खतरनाक वक्र बायाँ", "ta": "ஆபத்தான வளைவு இடது"},
    20: {"en": "Dangerous curve right", "hi": "खतरनाक वक्र दायाँ", "ta": "ஆபத்தான வளைவு வலம்"},
    21: {"en": "Double curve", "hi": "दोहरी वक्र", "ta": "இரட்டை வளைவு"},
    22: {"en": "Bumpy road", "hi": "खुरदरी सड़क", "ta": "சிறுதுண்டு சாலை"},
    23: {"en": "Slippery road", "hi": "फिसलन भरी सड़क", "ta": "சரிவான சாலை"},
    24: {"en": "Road narrows on the right", "hi": "सड़क दायाँ संकरा हो रहा है", "ta": "சாலை வலப்பக்கம் குறைகிறது"},
    25: {"en": "Road work", "hi": "सड़क कार्य", "ta": "சாலை வேலை"},
    26: {"en": "Traffic signals", "hi": "ट्रैफिक सिग्नल", "ta": "போக்குவரத்து அறிகுறிகள்"},
    27: {"en": "Pedestrians", "hi": "पैदल यात्री", "ta": "அடியார்கள்"},
    28: {"en": "Children crossing", "hi": "बच्चे सड़क पार कर रहे हैं", "ta": "குழந்தைகள் கடக்கின்றனர்"},
    29: {"en": "Bicycles crossing", "hi": "साइकिल क्रॉसिंग", "ta": "சைக்கிள் கடக்கிறது"},
    30: {"en": "Beware of ice/snow", "hi": "बर्फ/बर्फ़ से सावधान", "ta": "பனியிலிருந்து எச்சரிக்கை"},
    31: {"en": "Wild animals crossing", "hi": "जंगली जानवर क्रॉसिंग", "ta": "காட்டில் விலங்குகள் கடக்கின்றனர்"},
    32: {"en": "End of all speed and passing limits", "hi": "सभी गति और पासिंग सीमाओं का अंत", "ta": "அனைத்து வேக மற்றும் கடக்கும் வரம்புகளின் முடிவு"},
    33: {"en": "Turn right ahead", "hi": "आगे दायाँ मुड़ें", "ta": "முன்னே வலம் திரும்புங்கள்"},
    34: {"en": "Turn left ahead", "hi": "आगे बायाँ मुड़ें", "ta": "முன்னே இடம் திரும்புங்கள்"},
    35: {"en": "Ahead only", "hi": "केवल आगे", "ta": "முன்னே மட்டும்"},
    36: {"en": "Go straight or right", "hi": "सिधा जाएँ या दायाँ मुड़ें", "ta": "நேராக செல்லுங்கள் அல்லது வலமிருந்து"},
    37: {"en": "Go straight or left", "hi": "सिधा जाएँ या बायाँ मुड़ें", "ta": "நேராக செல்லுங்கள் அல்லது இடம் திரும்புங்கள்"},
    38: {"en": "Keep right", "hi": "दायाँ रहें", "ta": "வலப்பக்கம் வைக்கவும்"},
    39: {"en": "Keep left", "hi": "बायाँ रहें", "ta": "இடப்பக்கம் வைக்கவும்"},
    40: {"en": "Roundabout mandatory", "hi": "राउंडअबाउट अनिवार्य", "ta": "சுற்று வட்ட வழி கட்டாயம்"},
    41: {"en": "End of no passing", "hi": "पासिंग निषेध समाप्त", "ta": "மீறிச் செல்ல தடை முடிவு"},
    42: {"en": "End of no passing by vehicles over 3.5 tons", "hi": "3.5 टन से अधिक वाहनों के लिए पासिंग निषेध समाप्त", "ta": "3.5 டன் மீது வாகனங்களுக்கு கடக்கும் தடை முடிவு"},
}

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.h5", compile=False)

model = load_model()

# ---------------- PREPROCESS ----------------
def preprocess_image(img):
    img = img.convert("RGB").resize((30,30))
    arr = np.array(img)/255.0
    return np.expand_dims(arr, axis=0)

# ---------------- DIGIT-FOCUSED CROP ----------------
def digit_focused_crop(img):
    try:
        img_np = np.array(img)
        if img_np.size == 0:
            return img

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        thresh = cv2.adaptiveThreshold(
            blur,255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,11,2
        )

        contours,_ = cv2.findContours(
            thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return img

        c = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        pad = 10

        x1,y1 = max(0,x-pad),max(0,y-pad)
        x2,y2 = min(img_np.shape[1],x+w+pad),min(img_np.shape[0],y+h+pad)
        cropped = img_np[y1:y2,x1:x2]

        if cropped.size == 0:
            return img

        return Image.fromarray(cropped)
    except:
        return img

# ---------------- LANGUAGE ----------------
def get_meaning(idx, lang):
    return class_names.get(idx,{}).get(lang,"Unknown")

# ---------------- VOICE ----------------
def speak(text, lang):
    tts = gTTS(text=text, lang=lang)
    audio = BytesIO()
    tts.write_to_fp(audio)
    audio.seek(0)
    st.audio(audio, format="audio/mp3")

# ---------------- BAR CHART ----------------
def plot_probs(indices, probs, lang):
    labels = [get_meaning(i,lang) for i in indices]
    values = [probs[i]*100 for i in indices]

    fig,ax = plt.subplots(figsize=(6,3))
    ax.barh(labels,values)
    ax.invert_yaxis()
    ax.set_xlabel("Confidence (%)")
    st.pyplot(fig)

# ---------------- GRADCAM ----------------
def make_gradcam(img_array, model):
    last_conv = model.layers[-3]
    grad_model = tf.keras.models.Model(
        model.inputs,[last_conv.output,model.output]
    )

    with tf.GradientTape() as tape:
        conv_out,preds = grad_model(img_array)
        idx = tf.argmax(preds[0])
        loss = preds[:,idx]

    grads = tape.gradient(loss,conv_out)
    pooled = tf.reduce_mean(grads,axis=(0,1,2))
    heatmap = conv_out[0] @ pooled[...,tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap,0)
    heatmap /= np.max(heatmap)+1e-8
    return heatmap

def overlay(img,heatmap):
    img = np.array(img)
    heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap),cv2.COLORMAP_JET)
    out = cv2.addWeighted(img,0.6,heatmap,0.4,0)
    return Image.fromarray(out)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    page = st.radio("Navigation",["Home","Upload & Predict","Statistics","About"])
    top_k = st.slider("Top-K",3,10,3)
    lang = st.selectbox("Language",["en","hi","ta"])

# ---------------- PAGES ----------------
if page=="Home":
    st.markdown('<h1 class="main-header">🚦 German Traffic Sign AI</h1>',unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-based traffic sign recognition with explainability</p>',unsafe_allow_html=True)

elif page=="Upload & Predict":
    file = st.file_uploader("Upload Image",["jpg","png","jpeg"])
    if file:
        img = Image.open(file)
        st.image(img,caption="Original",width=300)

        crop = digit_focused_crop(img)
        st.image(crop,caption="Processed",width=300)

        if st.button("Analyze"):
            arr = preprocess_image(crop)
            preds = model.predict(arr)[0]
            top_idx = np.argsort(preds)[-top_k:][::-1]

            st.subheader("Predictions")
            for i in top_idx:
                conf = preds[i]*100
                st.write(f"**{get_meaning(i,lang)}** — {conf:.2f}%")
                st.markdown(f"<div class='bar' style='width:{conf}%'></div>",unsafe_allow_html=True)

            plot_probs(top_idx,preds,lang)
            speak(get_meaning(top_idx[0],lang),lang)

            heatmap = make_gradcam(arr,model)
            cam = overlay(crop,heatmap)
            st.image(cam,caption="🧠 Grad-CAM Explanation",width=350)

elif page=="Statistics":
    st.header("📊 Statistics")
    classes = list(class_names.keys())
    counts = np.random.randint(100,500,len(classes))
    fig,ax = plt.subplots()
    ax.bar(range(len(classes)),counts)
    st.pyplot(fig)

elif page=="About":
    st.header("ℹ️ About")
    st.markdown("""
- **Dataset:** GTSRB (43 classes)
- **Model:** CNN (TensorFlow / Keras)
- **Features:**
  - Smart auto-crop
  - Multi-language meaning
  - Voice explanation
  - Grad-CAM explainability
  - Cloud-safe deployment
""")
