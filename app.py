import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import pyttsx3

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="German Traffic Sign AI",
    page_icon="🚦",
    layout="wide"
)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.h5", compile=False)

model = load_model()

# ================= CLASSES =================
classes = {
    0: "Speed limit (20km/h)", 1: "Speed limit (30km/h)", 2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)", 4: "Speed limit (70km/h)", 5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)", 7: "Speed limit (100km/h)", 8: "Speed limit (120km/h)",
    9: "No passing", 10: "No passing for vehicles >3.5t", 11: "Right-of-way at intersection",
    12: "Priority road", 13: "Yield", 14: "Stop", 15: "No vehicles",
    16: "Vehicles >3.5t prohibited", 17: "No entry", 18: "General caution",
    19: "Dangerous curve left", 20: "Dangerous curve right", 21: "Double curve",
    22: "Bumpy road", 23: "Slippery road", 24: "Road narrows on right",
    25: "Road work", 26: "Traffic signals", 27: "Pedestrians",
    28: "Children crossing", 29: "Bicycles crossing", 30: "Beware of ice/snow",
    31: "Wild animals crossing", 32: "End of all restrictions",
    33: "Turn right ahead", 34: "Turn left ahead", 35: "Go straight",
    36: "Go straight or right", 37: "Go straight or left",
    38: "Keep right", 39: "Keep left", 40: "Roundabout mandatory",
    41: "End of no passing", 42: "End of no passing (>3.5t)"
}

# ================= MULTI-LANGUAGE MEANINGS =================
# ENGLISH | TAMIL | HINDI (ALL 43)
sign_meanings = {
    i: {
        "en": f"You must obey: {classes[i]}",
        "ta": f"இந்த சின்னத்தின் பொருள்: {classes[i]}",
        "hi": f"इस संकेत का अर्थ है: {classes[i]}"
    } for i in classes
}

# ================= PREPROCESS =================
def preprocess(img):
    img = img.resize((30, 30))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ================= VOICE =================
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# ================= SIDEBAR =================
st.sidebar.title("🚦 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📤 Upload Image", "📷 Webcam", "ℹ️ About"])
language = st.sidebar.selectbox("🌐 Language", ["English", "Tamil", "Hindi"])

lang_code = {"English": "en", "Tamil": "ta", "Hindi": "hi"}[language]

# ================= HOME =================
if page == "🏠 Home":
    st.title("🚦 German Traffic Sign Recognition AI")
    st.markdown("""
    ✅ Upload Image  
    ✅ Webcam Detection  
    ✅ Multi-Language Meaning  
    ✅ Voice Explanation  
    """)

# ================= UPLOAD MODE =================
elif page == "📤 Upload Image":
    st.header("Upload Traffic Sign Image")

    file = st.file_uploader("Upload Image", ["jpg", "png", "jpeg"])
    if file:
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", width=300)

        if st.button("🔍 Predict"):
            img = preprocess(image)
            preds = model.predict(img)
            class_id = int(np.argmax(preds))
            conf = float(np.max(preds))

            st.success(f"Prediction: {classes[class_id]}")
            st.write(f"Confidence: {conf:.2%}")

            meaning = sign_meanings[class_id][lang_code]
            st.info(meaning)

            if st.button("🔊 Voice Explanation"):
                speak(meaning)

# ================= WEBCAM MODE =================
elif page == "📷 Webcam":
    st.header("Live Webcam Detection")

    cam = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    stop = st.button("❌ Stop Webcam")

    while cam.isOpened() and not stop:
        ret, frame = cam.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        processed = preprocess(pil_img)
        preds = model.predict(processed)
        class_id = int(np.argmax(preds))
        conf = float(np.max(preds))

        label = f"{classes[class_id]} ({conf:.0%})"
        cv2.putText(frame, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frame_placeholder.image(frame, channels="BGR")

    cam.release()

# ================= ABOUT =================
elif page == "ℹ️ About":
    st.header("About")
    st.markdown("""
    **German Traffic Sign AI**
    
    • 43 Traffic Signs  
    • CNN Model (GTSRB Dataset)  
    • Streamlit Web App  
    • Multi-Language + Voice  
    """)

# ================= FOOTER =================
st.markdown("---")
st.caption("🚀 Built for AI / ML Projects & Interviews")
