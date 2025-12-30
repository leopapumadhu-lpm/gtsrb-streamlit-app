import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import io, base64
from gtts import gTTS
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(
    page_title="German Traffic Sign AI",
    page_icon="🚦",
    layout="wide"
)

# ===========================
# PREMIUM CSS
# ===========================
st.markdown("""
<style>
.main-title {font-size:40px;font-weight:800;color:#1976D2;text-align:center;margin-bottom:30px;}
.card {background:#ffffff;padding:20px;border-radius:16px;margin:10px 0;
box-shadow:0 6px 18px rgba(0,0,0,0.15);border-left:6px solid #1976D2;}
.bar {height:18px;border-radius:8px;margin:8px 0;}
.fade {animation: fadeIn 0.8s;}
@keyframes fadeIn {from {opacity:0;} to {opacity:1;}}
.category-badge {padding:5px 12px;border-radius:20px;font-size:12px;font-weight:600;}
.speed-badge {background:#FFEBEE;color:#D32F2F;}
.prohibitory-badge {background:#F3E5F5;color:#7B1FA2;}
.warning-badge {background:#FFF3E0;color:#F57C00;}
.mandatory-badge {background:#E8F5E8;color:#388E3C;}
.priority-badge {background:#E3F2FD;color:#1976D2;}
</style>
""", unsafe_allow_html=True)

# ===========================
# EXACT 43 CLASS DICTIONARY
# ===========================
CLASSES = {
    0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)',
    3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)',
    6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)',
    9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection',
    12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles',
    16:'Veh > 3.5 tons prohibited', 17:'No entry', 18:'General caution',
    19:'Dangerous curve left', 20:'Dangerous curve right', 21:'Double curve',
    22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right',
    25:'Road work', 26:'Traffic signals', 27:'Pedestrians',
    28:'Children crossing', 29:'Bicycles crossing', 30:'Beware of ice/snow',
    31:'Wild animals crossing', 32:'End speed + passing limits',
    33:'Turn right ahead', 34:'Turn left ahead', 35:'Ahead only',
    36:'Go straight or right', 37:'Go straight or left',
    38:'Keep right', 39:'Keep left', 40:'Roundabout mandatory',
    41:'End of no passing', 42:'End no passing veh > 3.5 tons'
}

# ===========================
# MULTI-LANGUAGE MEANINGS (COMPLETE 43 CLASSES)
# ===========================
MEANINGS = {
    0: {"en": "Speed limit (20km/h)", "hi": "गति सीमा (20 किमी/घंटा)", "ta": "வேக வரம்பு (20 கிமீ/மணி)"},
    1: {"en": "Speed limit (30km/h)", "hi": "गति सीमा (30 किमी/घंटा)", "ta": "வேக வரம்பு (30 கிமீ/மணி)"},
    2: {"en": "Speed limit (50km/h)", "hi": "गति सीमा (50 किमी/घंटा)", "ta": "வேக வரம்பு (50 கிமீ/மணி)"},
    3: {"en": "Speed limit (60km/h)", "hi": "गति सीमा (60 किमी/घंटा)", "ta": "வேக வரம்பு (60 கிமீ/மணி)"},
    4: {"en": "Speed limit (70km/h)", "hi": "गति सीमा (70 किमी/घंटा)", "ta": "வேக வரம்பு (70 கிமீ/மணி)"},
    5: {"en": "Speed limit (80km/h)", "hi": "गति सीमा (80 किमी/घंटा)", "ta": "வேக வரம்பு (80 கிமீ/மணி)"},
    6: {"en": "End of speed limit (80km/h)", "hi": "गति सीमा समाप्त (80 किमी/घंटा)", "ta": "வேக வரம்பு முடிவு (80 கிமீ/மணி)"},
    7: {"en": "Speed limit (100km/h)", "hi": "गति सीमा (100 किमी/घंटा)", "ta": "வேக வரம்பு (100 கிமீ/மணி)"},
    8: {"en": "Speed limit (120km/h)", "hi": "गति सीमा (120 किमी/घंटा)", "ta": "வேக வரம்பு (120 கிமீ/மணி)"},
    9: {"en": "No passing", "hi": "ओवरटेकिंग निषेध", "ta": "முந்துதல் தடை"},
    10: {"en": "No passing for vehicles over 3.5 tons", "hi": "3.5 टन से अधिक वाहनों के लिए ओवरटेकिंग निषेध", "ta": "3.5 டனுக்கு மேற்பட்ட வாகனங்களுக்கு முந்துதல் தடை"},
    11: {"en": "Right-of-way at intersection", "hi": "चौराहे पर प्राथमिकता", "ta": "சந்திப்பில் முன்னுரிமை"},
    12: {"en": "Priority road", "hi": "प्राथमिकता मार्ग", "ta": "முன்னுரிமை சாலை"},
    13: {"en": "Yield", "hi": "रास्ता दें", "ta": "வழி விடுங்கள்"},
    14: {"en": "Stop", "hi": "रुकें", "ta": "நிறுத்தவும்"},
    15: {"en": "No vehicles", "hi": "कोई वाहन नहीं", "ta": "வாகனங்கள் தடை"},
    16: {"en": "Vehicles over 3.5 tons prohibited", "hi": "3.5 टन से अधिक वाहन प्रतिबंधित", "ta": "3.5 டனுக்கு மேற்பட்ட வாகனங்கள் தடை"},
    17: {"en": "No entry", "hi": "प्रवेश निषेध", "ta": "நுழைவு தடை"},
    18: {"en": "General caution", "hi": "सामान्य सावधानी", "ta": "பொது எச்சரிக்கை"},
    19: {"en": "Dangerous curve left", "hi": "खतरनाक वक्र बाएं", "ta": "அபாயகரமான வளைவு இடது"},
    20: {"en": "Dangerous curve right", "hi": "खतरनाक वक्र दाएं", "ta": "அபாயகரமான வளைவு வலது"},
    21: {"en": "Double curve", "hi": "दोहरा वक्र", "ta": "இரட்டை வளைவு"},
    22: {"en": "Bumpy road", "hi": "ऊबड़-खाबड़ सड़क", "ta": "கரடுமுரடான சாலை"},
    23: {"en": "Slippery road", "hi": "फिसलन भरी सड़क", "ta": "வழுக்கும் சாலை"},
    24: {"en": "Road narrows on the right", "hi": "सड़क दाएं संकरी होती है", "ta": "வலதுபுறம் சாலை குறுகலாகிறது"},
    25: {"en": "Road work", "hi": "सड़क कार्य", "ta": "சாலை பணிகள்"},
    26: {"en": "Traffic signals", "hi": "ट्रैफिक सिग्नल", "ta": "போக்குவரத்து சைகைகள்"},
    27: {"en": "Pedestrians", "hi": "पैदल यात्री", "ta": "பாதசாரிகள்"},
    28: {"en": "Children crossing", "hi": "बच्चे पार कर रहे हैं", "ta": "குழந்தைகள் கடக்கிறார்கள்"},
    29: {"en": "Bicycles crossing", "hi": "साइकिल पार कर रही है", "ta": "சைக்கிள்கள் கடக்கின்றன"},
    30: {"en": "Beware of ice/snow", "hi": "बर्फ/हिमपात सावधान", "ta": "பனி/மழை எச்சரிக்கை"},
    31: {"en": "Wild animals crossing", "hi": "जंगली जानवर पार कर रहे हैं", "ta": "காட்டு விலங்குகள் கடக்கின்றன"},
    32: {"en": "End of all speed and passing limits", "hi": "सभी गति और ओवरटेकिंग सीमाएं समाप्त", "ta": "அனைத்து வேக மற்றும் முந்துதல் வரம்புகள் முடிவு"},
    33: {"en": "Turn right ahead", "hi": "आगे दाएं मुड़ें", "ta": "முன்னால் வலதுபுறம் திரும்பு"},
    34: {"en": "Turn left ahead", "hi": "आगे बाएं मुड़ें", "ta": "முன்னால் இடதுபுறம் திரும்பு"},
    35: {"en": "Ahead only", "hi": "केवल आगे", "ta": "நேரே மட்டும்"},
    36: {"en": "Go straight or right", "hi": "सीधे या दाएं जाएं", "ta": "நேரே அல்லது வலப்புறம் செல்லுங்கள்"},
    37: {"en": "Go straight or left", "hi": "सीधे या बाएं जाएं", "ta": "நேரே அல்லது இடப்புறம் செல்லுங்கள்"},
    38: {"en": "Keep right", "hi": "दाएं रहें", "ta": "வலதுபுறம் செல்லுங்கள்"},
    39: {"en": "Keep left", "hi": "बाएं रहें", "ta": "இடதுபுறம் செல்லுங்கள்"},
    40: {"en": "Roundabout mandatory", "hi": "राउंडअबाउट अनिवार्य", "ta": "சுற்றுச்சந்தி கட்டாயம்"},
    41: {"en": "End of no passing", "hi": "ओवरटेकिंग निषेध समाप्त", "ta": "முந்துதல் தடை முடிவு"},
    42: {"en": "End of no passing for vehicles over 3.5 tons", "hi": "3.5 टन से अधिक वाहनों के लिए ओवरटेकिंग निषेध समाप्त", "ta": "3.5 டனுக்கு மேற்பட்ட வாகனங்களுக்கு முந்துதல் தடை முடிவு"}
}

# ===========================
# CATEGORY INFORMATION
# ===========================
def get_category(class_id):
    """Categorize the traffic sign"""
    if class_id <= 8:
        return "Speed Limits"
    elif class_id in [9, 10, 15, 16, 17, 41, 42]:
        return "Prohibitory"
    elif class_id in [11, 12, 13, 14]:
        return "Priority"
    elif class_id in [33, 34, 35, 36, 37, 38, 39, 40]:
        return "Mandatory"
    else:
        return "Warning"

def get_category_color(category):
    colors = {
        "Speed Limits": "#D32F2F",
        "Prohibitory": "#7B1FA2",
        "Warning": "#F57C00",
        "Mandatory": "#388E3C",
        "Priority": "#1976D2"
    }
    return colors.get(category, "#757575")

# ===========================
# LOAD MODEL
# ===========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5", compile=False)

model = load_model()

# ===========================
# SMART DIGIT-FOCUSED CROP
# ===========================
def smart_crop(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    _,th = cv2.threshold(gray,120,255,cv2.THRESH_BINARY_INV)
    cnts,_ = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x,y,w,h = cv2.boundingRect(max(cnts,key=cv2.contourArea))
        img = img.crop((x,y,x+w,y+h))
    return img

# ===========================
# PREPROCESS
# ===========================
def preprocess(img):
    img = img.resize((30,30))
    arr = np.array(img)/255.0
    return np.expand_dims(arr,0)

# ===========================
# GRAD-CAM
# ===========================
def make_gradcam(img_array, model, last_conv="conv2d"):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, tf.argmax(preds[0])]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    heatmap = tf.reduce_sum(tf.multiply(pooled, conv_out[0]), axis=-1)
    heatmap = np.maximum(heatmap,0) / np.max(heatmap)
    return heatmap

# ===========================
# VOICE
# ===========================
def speak(text, lang):
    tts = gTTS(text=text, lang=lang)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    audio = base64.b64encode(fp.read()).decode()
    st.markdown(f"""
    <audio autoplay controls>
    <source src="data:audio/mp3;base64,{audio}">
    </audio>
    """, unsafe_allow_html=True)

# ===========================
# UI
# ===========================
st.markdown('<div class="main-title fade">🚦 German Traffic Sign Recognition AI</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🌐 Settings")
    lang = st.radio("Language", ["en", "ta", "hi"], horizontal=True)
    st.divider()
    st.header("📊 Display Options")
    show_heatmap = st.checkbox("Show Heatmap", True)
    show_top_k = st.slider("Top K Predictions", 3, 10, 5)
    auto_voice = st.checkbox("Auto Voice Explanation", True)

# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    file = st.file_uploader("📤 Upload Traffic Sign Image", ["jpg","png","jpeg"], help="Upload German traffic sign image")
    
    if file:
        img = Image.open(file)
        img_cropped = smart_crop(img)
        
        tab1, tab2 = st.tabs(["📷 Original", "✂️ Processed"])
        with tab1:
            st.image(img, caption="Original Image", use_column_width=True)
        with tab2:
            st.image(img_cropped, caption="Cropped & Processed", use_column_width=True)
        
        # Process and Predict
        arr = preprocess(img_cropped)
        preds = model.predict(arr, verbose=0)[0]
        top_class = np.argmax(preds)
        top_confidence = preds[top_class] * 100
        
        # Get category
        category = get_category(top_class)
        category_color = get_category_color(category)
        
        # Display Results
        st.markdown(f"""
        <div class="card fade">
            <h2>✅ {CLASSES[top_class]}</h2>
            <span class="category-badge" style="background:{category_color}20;color:{category_color}">
                {category}
            </span>
            <h4>{MEANINGS[top_class][lang]}</h4>
            <h3 style="color:{'#4CAF50' if top_confidence > 90 else '#FF9800' if top_confidence > 70 else '#F44336'}">
                {top_confidence:.2f}% confidence
            </h3>
            <p><strong>Class ID:</strong> {top_class}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Voice Explanation
        if auto_voice:
            with st.spinner("🔊 Generating voice explanation..."):
                speak(f"This is a {CLASSES[top_class]} sign with {top_confidence:.1f} percent confidence", 
                     "en" if lang == "en" else lang)
        
        # Top-K Predictions
        st.subheader("📊 Top Predictions")
        top_indices = np.argsort(preds)[-show_top_k:][::-1]
        
        # Create dataframe for visualization
        df = pd.DataFrame({
            'Sign': [CLASSES[i] for i in top_indices],
            'Confidence': [preds[i]*100 for i in top_indices],
            'Class ID': top_indices
        })
        
        # Bar chart
        fig = px.bar(df, x='Confidence', y='Sign', 
                     orientation='h', color='Confidence',
                     color_continuous_scale=['#FF5252', '#FF9800', '#4CAF50'],
                     title=f'Top-{show_top_k} Predictions')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        with st.expander("🔍 View Detailed Predictions"):
            st.dataframe(df.style.format({'Confidence': '{:.2f}%'})
                        .background_gradient(subset=['Confidence'], cmap='YlOrRd'),
                        use_container_width=True)
        
        # Grad-CAM Heatmap
        if show_heatmap:
            st.subheader("🧠 AI Attention Heatmap (Grad-CAM)")
            with st.spinner("Generating heatmap..."):
                heat = make_gradcam(arr, model)
                heat = cv2.resize(heat, (300, 300))
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                
                # Original
                ax1.imshow(img_cropped)
                ax1.set_title('Processed Image')
                ax1.axis('off')
                
                # Heatmap
                im = ax2.imshow(img_cropped)
                ax2.imshow(heat, cmap="jet", alpha=0.5)
                ax2.set_title('AI Focus Areas')
                ax2.axis('off')
                
                plt.colorbar(plt.cm.ScalarMappable(cmap="jet"), ax=ax2, fraction=0.046, pad=0.04)
                st.pyplot(fig)

with col2:
    st.markdown("""
    <div class="card">
        <h3>ℹ️ About This System</h3>
        <p><strong>Model:</strong> CNN trained on GTSRB</p>
        <p><strong>Classes:</strong> 43 German traffic signs</p>
        <p><strong>Accuracy:</strong> ~99%</p>
        <hr>
        <h4>🎯 Common Signs</h4>
        <p>• <strong>Class 4:</strong> Speed limit (70km/h)</p>
        <p>• <strong>Class 13:</strong> Yield</p>
        <p>• <strong>Class 17:</strong> No entry</p>
        <p>• <strong>Class 22:</strong> Bumpy road</p>
        <p>• <strong>Class 38:</strong> Keep right</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Stats
    st.markdown("""
    <div class="card">
        <h4>📈 Performance</h4>
        <p>✅ 43-class recognition</p>
        <p>✅ Multi-language support</p>
        <p>✅ Visual explanations</p>
        <p>✅ Voice output</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Category Legend
    st.markdown("""
    <div class="card">
        <h4>🏷️ Sign Categories</h4>
        <p><span style="color:#D32F2F">●</span> Speed Limits</p>
        <p
