import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from gtts import gTTS
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="GTSRB AI – Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide"
)

# ================== LOAD MODEL ==================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("gtsrb_model.h5", compile=False)

model = load_model()

# ================== CLASS NAMES (MATCHED 43) ==================
classes = {
    0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)',
    3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)',
    6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)',
    9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection',
    12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles',
    16:'Veh > 3.5 tons prohibited', 17:'No entry', 18:'General caution',
    19:'Dangerous curve left', 20:'Dangerous curve right', 21:'Double curve',
    22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right',
    25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing',
    29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing',
    32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead',
    35:'Ahead only', 36:'Go straight or right', 37:'Go straight or left',
    38:'Keep right', 39:'Keep left', 40:'Roundabout mandatory',
    41:'End of no passing', 42:'End no passing veh > 3.5 tons'
}

# ================== MULTI-LANGUAGE MEANINGS ==================
meanings = {}
for k, v in classes.items():
    meanings[k] = {
        "en": v,
        "ta": f"⚠️ {v} – போக்குவரத்து விதி",
        "hi": f"⚠️ {v} – यातायात नियम"
    }

# ================== SMART DIGIT-FOCUSED CROP ==================
def smart_crop(img):
    img_cv = np.array(img)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(blur,50,150)

    cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        if w>30 and h>30:
            return img.crop((x,y,x+w,y+h))
    return img

# ================== PREDICTION ==================
def predict(img):
    img = img.resize((30,30))
    arr = np.array(img)/255.0
    arr = arr.reshape(1,30,30,3)
    preds = model.predict(arr)[0]
    top = np.argsort(preds)[::-1][:3]
    return preds, top

# ================== GRAD-CAM ==================
def make_gradcam(img_array, model):
    last_conv = None
    for layer in model.layers[::-1]:
        if len(layer.output_shape)==4:
            last_conv = layer
            break

    grad_model = tf.keras.models.Model(
        model.inputs,
        [last_conv.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    heatmap = tf.reduce_sum(tf.multiply(pooled, conv_out[0]), axis=-1)

    heatmap = np.maximum(heatmap,0)
    heatmap /= np.max(heatmap)+1e-8
    return heatmap.numpy()

# ================== VOICE ==================
def speak(text, lang):
    tts = gTTS(text=text, lang=lang)
    fp = BytesIO()
    tts.write_to_fp(fp)
    st.audio(fp.getvalue(), format="audio/mp3")

# ================== UI ==================
st.title("🚦 German Traffic Sign Recognition (AI)")
tabs = st.tabs(["🏠 Predict", "📊 Statistics", "ℹ️ About"])

# ================== TAB 1 ==================
with tabs[0]:
    uploaded = st.file_uploader("Upload Traffic Sign Image", type=["jpg","png"])
    lang = st.selectbox("Language", ["English","Tamil","Hindi"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        crop = smart_crop(img)

        st.image([img, crop], caption=["Original","Smart Crop"], width=300)

        preds, top = predict(crop)
        arr = np.array(crop.resize((30,30)))/255.0
        arr = arr.reshape(1,30,30,3)

        st.subheader("🔮 Predictions")
        for i in top:
            st.success(f"{classes[i]} – {preds[i]*100:.2f}%")

        # BAR CHART
        fig, ax = plt.subplots()
        ax.bar([classes[i] for i in top], [preds[i] for i in top])
        ax.set_ylabel("Probability")
        st.pyplot(fig)

        # MEANING + VOICE
        idx = top[0]
        key = "en" if lang=="English" else "ta" if lang=="Tamil" else "hi"
        meaning = meanings[idx][key]

        st.info(meaning)
        speak(meaning, "en" if key=="en" else "ta" if key=="ta" else "hi")

        # GRAD-CAM
        heatmap = make_gradcam(arr, model)
        heatmap = cv2.resize(heatmap, crop.size)
        heatmap = np.uint8(255*heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(np.array(crop),0.6,heatmap,0.4,0)
        st.image(overlay, caption="🧠 Grad-CAM")

# ================== TAB 2 ==================
with tabs[1]:
    st.subheader("📊 Model Statistics")
    st.write("Total Classes:", len(classes))
    st.write("Model Input Size: 30×30 RGB")
    st.write("Dataset: GTSRB (German Traffic Sign Benchmark)")

# ================== TAB 3 ==================
with tabs[2]:
    st.subheader("ℹ️ About")
    st.write("""
    This AI app uses a CNN trained on **GTSRB dataset**.
    Features:
    - Smart crop
    - Multi-language meanings
    - Voice explanation
    - Grad-CAM visualization
    - Streamlit-safe deployment
    """)
