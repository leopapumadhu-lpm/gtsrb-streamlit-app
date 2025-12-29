import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import pyttsx3
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="German Traffic Sign AI", page_icon="🚦", layout="wide")

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# -------------------- CSS --------------------
st.markdown("""
<style>
.main-header { font-size:3rem; color:#1E88E5; text-align:center; font-weight:bold;}
.sub-header { text-align:center; color:#666; margin-bottom:2rem;}
.prediction-card {background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); padding:1.5rem; border-radius:15px; color:white; margin:1rem 0;}
.feature-card {background:white; padding:1.5rem; border-radius:10px; box-shadow:0 4px 6px rgba(0,0,0,0.1); margin:1rem 0; border-left:5px solid #1E88E5;}
.stProgress > div > div > div > div {background-color: #1E88E5;}
</style>
""", unsafe_allow_html=True)

# -------------------- Class Info --------------------
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

# -------------------- Model --------------------
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("best_model.h5")
    except Exception as e:
        st.error(f"Model load error: {str(e)}")
        return None

model = load_model()

# -------------------- Image Preprocess --------------------
def preprocess_image(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((30,30))
    arr = np.array(img)/255.0
    return np.expand_dims(arr, axis=0)

# -------------------- Auto-crop --------------------
def auto_crop(img):
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5),0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=5,maxRadius=100)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x,y,r = circles[0][0]
        x1,y1 = max(0,x-r), max(0,y-r)
        x2,y2 = min(img_cv.shape[1],x+r), min(img_cv.shape[0],y+r)
        cropped = img_cv[y1:y2, x1:x2]
        return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    return img

# -------------------- Multi-language + TTS --------------------
def get_meaning(class_id, lang='en'):
    if class_id in classes:
        return classes[class_id].get(lang, classes[class_id]['en'])
    return "Unknown"

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# -------------------- Download Link --------------------
def get_download_link(img, name):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<a href="data:file/png;base64,{img_str}" download="{name}" style="background-color:#4CAF50;color:white;padding:10px 20px;text-decoration:none;border-radius:5px;">📥 Download Result</a>'

# -------------------- Sidebar --------------------
with st.sidebar:
    st.title("🚦 Navigation")
    tab = st.radio("Go to", ["Home","Upload & Predict","Statistics","About"])
    st.markdown("---")
    top_k = st.slider("Top Predictions",3,10,3)
    lang = st.selectbox("Language", ["en","ta","hi"])
    st.markdown("---")
    st.caption("Built with Streamlit & TensorFlow")

# -------------------- Tabs --------------------
if tab=="Home":
    st.markdown('<h1 class="main-header">🚦 German Traffic Sign AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image to get instant predictions!</p>', unsafe_allow_html=True)
elif tab=="Upload & Predict":
    uploaded_file = st.file_uploader("Upload Image", type=['png','jpg','jpeg'])
    if uploaded_file and model:
        img = Image.open(uploaded_file)
        st.image(img, caption="Original Image", use_column_width=True)
        img_crop = auto_crop(img)
        st.image(img_crop, caption="Cropped Image", use_column_width=True)
        if st.button("Analyze"):
            pred = model.predict(preprocess_image(img_crop))[0]
            top_indices = np.argsort(pred)[-top_k:][::-1]
            st.markdown("### 🔹 Predictions")
            for i, idx in enumerate(top_indices):
                meaning = get_meaning(idx, lang)
                conf = pred[idx]
                st.markdown(f"**#{i+1}** {meaning} — {conf:.2%}")
            # Speak top prediction
            speak_text(get_meaning(top_indices[0], lang))
elif tab=="Statistics":
    st.markdown('<h1 class="main-header">📊 Statistics</h1>', unsafe_allow_html=True)
    st.markdown("Sample Class Distribution")
    sample = list(classes.keys())[:10]
    counts = np.random.randint(100,1000,len(sample))
    fig,ax = plt.subplots()
    ax.bar([classes[i]['en'] for i in sample], counts)
    plt.xticks(rotation=45)
    st.pyplot(fig)
elif tab=="About":
    st.markdown('<h1 class="main-header">ℹ️ About</h1>', unsafe_allow_html=True)
    st.markdown("This app recognizes 43 German traffic signs using CNN with multi-language meanings and voice feedback.")
