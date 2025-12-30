import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os
import json
from datetime import datetime
import tensorflow as tf
from gtts import gTTS
import io
import base64

# Set page config
st.set_page_config(
    page_title="Traffic Sign Recognizer",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================================
# ✅ EXACT 43-CLASS DICTIONARY (MATCHING MODEL)
# ===========================================
CLASSES = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',  # ← THIS IS CLASS 4 for "70" sign
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing veh over 3.5 tons',
    11: 'Right-of-way at intersection',
    12: 'Priority road',
    13: 'Yield',  # ← THIS IS CLASS 13 for "Yield" sign
    14: 'Stop',
    15: 'No vehicles',
    16: 'Veh > 3.5 tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve left',
    20: 'Dangerous curve right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End speed + passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End no passing veh > 3.5 tons'
}

# Multi-language support (English, Hindi, Tamil)
TRANSLATIONS = {
    0: {'hi': '20 किमी/घंटा गति सीमा', 'ta': '20 கிமீ/மணி வேக வரம்பு'},
    1: {'hi': '30 किमी/घंटा गति सीमा', 'ta': '30 கிமீ/மணி வேக வரம்பு'},
    2: {'hi': '50 किमी/घंटा गति सीमा', 'ta': '50 கிமீ/மணி வேக வரம்பு'},
    3: {'hi': '60 किमी/घंटा गति सीमा', 'ta': '60 கிமீ/மணி வேக வரம்பு'},
    4: {'hi': '70 किमी/घंटा गति सीमा', 'ta': '70 கிமீ/மணி வேக வரம்பு'},  # 70 km/h in Hindi/Tamil
    5: {'hi': '80 किमी/घंटा गति सीमा', 'ta': '80 கிமீ/மணி வேக வரம்பு'},
    13: {'hi': 'रास्ता दें', 'ta': 'வழி விடுங்கள்'},  # Yield in Hindi/Tamil
    14: {'hi': 'रुकें', 'ta': 'நிறுத்து'},  # Stop
    17: {'hi': 'प्रवेश निषेध', 'ta': 'நுழைவு தடை'},  # No entry
    18: {'hi': 'सामान्य सावधानी', 'ta': 'பொது எச்சரிக்கை'},  # General caution
    22: {'hi': 'ऊबड़-खाबड़ सड़क', 'ta': 'கரடுமுரடான சாலை'},  # Bumpy road
    23: {'hi': 'फिसलन भरी सड़क', 'ta': 'வழுக்கும் சாலை'},  # Slippery road
    38: {'hi': 'दाएं रहें', 'ta': 'வலதுபுறம் செல்லுங்கள்'},  # Keep right
}

# Category grouping
CATEGORIES = {
    'Speed Limits': list(range(0, 9)),
    'Prohibitory': [9, 10, 15, 16, 17, 41, 42],
    'Mandatory': [11, 12, 13, 14, 33, 34, 35, 36, 37, 38, 39, 40],
    'Warning': list(range(18, 32)),
    'End of Restrictions': [6, 32, 41, 42]
}

# ===========================================
# LOAD MODEL
# ===========================================
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        # Update this path to your actual model file
        model = tf.keras.models.load_model('best_model.h5')
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# ===========================================
# PREPROCESSING FUNCTIONS
# ===========================================
def preprocess_image(image, target_size=(32, 32)):
    """Preprocess image exactly like training"""
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Convert RGBA to RGB if needed
    if img_array.shape[-1] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Convert RGB to BGR for OpenCV
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Resize to 32x32
    img_resized = cv2.resize(img_array, target_size)
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img_normalized = img_rgb.astype('float32') / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch, img_resized

# ===========================================
# PREDICTION FUNCTIONS
# ===========================================
def predict_image(image, top_k=5):
    """Get predictions with top K results"""
    if model is None:
        st.error("Model not loaded!")
        return None, None, None
    
    # Preprocess image
    processed_img, original_resized = preprocess_image(image)
    
    # Get predictions
    predictions = model.predict(processed_img, verbose=0)[0]
    
    # Get top K predictions
    top_k_indices = np.argsort(predictions)[-top_k:][::-1]
    top_k_probs = predictions[top_k_indices]
    
    # Create results dictionary
    results = []
    for idx, prob in zip(top_k_indices, top_k_probs):
        results.append({
            'class_id': int(idx),
            'class_name': CLASSES[idx],
            'probability': float(prob),
            'percentage': float(prob * 100),
            'hindi': TRANSLATIONS.get(idx, {}).get('hi', CLASSES[idx]),
            'tamil': TRANSLATIONS.get(idx, {}).get('ta', CLASSES[idx])
        })
    
    return results, predictions, original_resized

def generate_heatmap(image, predictions):
    """Generate a simple heatmap visualization"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # Original image
    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Uploaded Image')
    ax[0].axis('off')
    
    # Prediction probabilities
    classes_list = list(CLASSES.values())
    sorted_indices = np.argsort(predictions)[::-1][:10]
    sorted_probs = predictions[sorted_indices]
    sorted_names = [classes_list[i][:20] + '...' if len(classes_list[i]) > 20 else classes_list[i] 
                    for i in sorted_indices]
    
    bars = ax[1].barh(range(len(sorted_names)), sorted_probs * 100)
    ax[1].set_yticks(range(len(sorted_names)))
    ax[1].set_yticklabels(sorted_names)
    ax[1].set_xlabel('Probability (%)')
    ax[1].set_title('Top 10 Predictions')
    ax[1].invert_yaxis()
    
    # Color bars by probability
    for i, bar in enumerate(bars):
        if i == 0:
            bar.set_color('green')
        elif sorted_probs[i] > 0.1:
            bar.set_color('orange')
        else:
            bar.set_color('gray')
    
    plt.tight_layout()
    return fig

def text_to_speech(text, lang='en'):
    """Convert text to speech"""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.warning(f"Could not generate speech: {e}")
        return None

# ===========================================
# SESSION STATE
# ===========================================
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

if 'stats' not in st.session_state:
    st.session_state.stats = {
        'total_predictions': 0,
        'correct_predictions': 0,
        'confidence_sum': 0,
        'by_category': {cat: 0 for cat in CATEGORIES.keys()},
        'by_class': {i: 0 for i in range(43)}
    }

# ===========================================
# SIDEBAR NAVIGATION
# ===========================================
st.sidebar.title("🚦 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Predict", "📊 Statistics", "ℹ️ About"])

st.sidebar.markdown("---")
st.sidebar.subheader("Settings")
top_k = st.sidebar.slider("Top K predictions", 1, 10, 5)
show_heatmap = st.sidebar.checkbox("Show heatmap", True)
enable_voice = st.sidebar.checkbox("Enable voice explanation", True)

st.sidebar.markdown("---")
st.sidebar.info(f"**Model:** GTSRB Traffic Sign\n**Classes:** 43\n**Accuracy:** ~99%")

# ===========================================
# PAGE 1: HOME
# ===========================================
if page == "🏠 Home":
    st.title("🚦 Traffic Sign Recognition System")
    st.markdown("""
    ### Welcome to the Intelligent Traffic Sign Recognizer
    
    This system uses a **deep learning model** trained on the **German Traffic Sign Recognition Benchmark (GTSRB)** 
    dataset to accurately identify 43 different types of traffic signs.
    
    ### Features:
    ✅ **43-class accurate recognition** - No more 70 vs Yield confusion!  
    ✅ **Top-K predictions** - See multiple possibilities  
    ✅ **Multi-language support** - English, Hindi, Tamil  
    ✅ **Visual explanations** - Heatmaps and probability charts  
    ✅ **Voice explanations** - Hear the prediction  
    ✅ **Statistics tracking** - Monitor prediction history  
    
    ### How to use:
    1. Go to the **🔍 Predict** page
    2. Upload a traffic sign image
    3. View predictions with confidence scores
    4. Explore visual explanations
    
    ### Supported Sign Categories:
    """)
    
    # Display categories
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Speed Limits**")
        st.markdown("- 20 km/h to 120 km/h")
        st.markdown("- End of speed limits")
    
    with col2:
        st.markdown("**Warning Signs**")
        st.markdown("- Dangerous curves")
        st.markdown("- Bumpy/Slippery roads")
        st.markdown("- Pedestrians/Animals")
    
    with col3:
        st.markdown("**Regulatory Signs**")
        st.markdown("- Stop/Yield/No entry")
        st.markdown("- No passing")
        st.markdown("- Keep right/left")
    
    st.markdown("---")
    
    # Show sample images
    st.subheader("Sample Traffic Signs")
    sample_cols = st.columns(5)
    sample_classes = [4, 13, 17, 22, 38]  # 70, Yield, No entry, Bumpy, Keep right
    
    for idx, col in enumerate(sample_cols):
        with col:
            class_id = sample_classes[idx]
            st.markdown(f"**{CLASSES[class_id]}**")
            st.markdown(f"*Class ID: {class_id}*")
            st.caption(TRANSLATIONS.get(class_id, {}).get('hi', ''))

# ===========================================
# PAGE 2: PREDICT
# ===========================================
elif page == "🔍 Predict":
    st.title("🔍 Traffic Sign Prediction")
    
    # Upload section
    uploaded_file = st.file_uploader("Choose a traffic sign image...", 
                                     type=['jpg', 'jpeg', 'png', 'bmp'])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Make prediction
            if st.button("🚀 Analyze Sign", type="primary", use_container_width=True):
                with st.spinner("Analyzing traffic sign..."):
                    # Get predictions
                    results, all_probs, processed_img = predict_image(image, top_k=top_k)
                    
                    if results:
                        # Update statistics
                        st.session_state.stats['total_predictions'] += 1
                        top_prediction = results[0]
                        st.session_state.stats['confidence_sum'] += top_prediction['percentage']
                        
                        # Track by class
                        st.session_state.stats['by_class'][top_prediction['class_id']] += 1
                        
                        # Track by category
                        for category, class_ids in CATEGORIES.items():
                            if top_prediction['class_id'] in class_ids:
                                st.session_state.stats['by_category'][category] += 1
                                break
                        
                        # Store in history
                        history_entry = {
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'class_id': top_prediction['class_id'],
                            'class_name': top_prediction['class_name'],
                            'confidence': top_prediction['percentage'],
                            'filename': uploaded_file.name
                        }
                        st.session_state.predictions_history.append(history_entry)
                        
                        # Display results
                        st.success(f"✅ Analysis complete!")
                        
                        # Top prediction in a nice box
                        with st.container():
                            st.markdown("---")
                            st.subheader("🎯 Top Prediction")
                            
                            pred_col1, pred_col2, pred_col3 = st.columns([1, 2, 1])
                            
                            with pred_col1:
                                st.metric("Class ID", top_prediction['class_id'])
                            
                            with pred_col2:
                                st.metric("Sign", top_prediction['class_name'])
                                st.caption(f"Hindi: {top_prediction['hindi']}")
                                st.caption(f"Tamil: {top_prediction['tamil']}")
                            
                            with pred_col3:
                                st.metric("Confidence", f"{top_prediction['percentage']:.2f}%")
                                color = "green" if top_prediction['percentage'] > 90 else "orange" if top_prediction['percentage'] > 70 else "red"
                                st.markdown(f"<p style='color:{color}; font-size:12px;'>" + 
                                           ("High confidence" if top_prediction['percentage'] > 90 else 
                                            "Medium confidence" if top_prediction['percentage'] > 70 else "Low confidence") + 
                                           "</p>", unsafe_allow_html=True)
                        
                        # Top-K predictions
                        st.markdown("---")
                        st.subheader(f"📊 Top-{top_k} Predictions")
                        
                        # Create dataframe for visualization
                        df_topk = pd.DataFrame(results)
                        df_topk['Probability (%)'] = df_topk['percentage'].round(2)
                        
                        # Bar chart
                        fig = px.bar(df_topk, 
                                    x='Probability (%)', 
                                    y='class_name',
                                    orientation='h',
                                    color='Probability (%)',
                                    color_continuous_scale=['red', 'orange', 'green'],
                                    title=f'Top-{top_k} Predictions (Confidence Scores)')
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed table
                        st.dataframe(df_topk[['class_id', 'class_name', 'Probability (%)']].style.format({
                            'Probability (%)': '{:.2f}%'
                        }).background_gradient(subset=['Probability (%)'], cmap='YlOrRd'), 
                        use_container_width=True)
                        
                        # Heatmap visualization
                        if show_heatmap and processed_img is not None:
                            st.markdown("---")
                            st.subheader("🔥 Prediction Heatmap")
                            heatmap_fig = generate_heatmap(processed_img, all_probs)
                            st.pyplot(heatmap_fig)
                        
                        # Voice explanation
                        if enable_voice:
                            st.markdown("---")
                            st.subheader("🔊 Voice Explanation")
                            
                            explanation_text = f"This is a {top_prediction['class_name']} sign with {top_prediction['percentage']:.1f} percent confidence."
                            
                            audio_bytes = text_to_speech(explanation_text, 'en')
                            if audio_bytes:
                                st.audio(audio_bytes, format='audio/mp3')
                                st.caption("Click play to hear the prediction")
                            
                            # Multi-language voice options
                            lang_col1, lang_col2, lang_col3 = st.columns(3)
                            with lang_col1:
                                if st.button("English", use_container_width=True):
                                    audio_en = text_to_speech(explanation_text, 'en')
                                    if audio_en:
                                        st.audio(audio_en, format='audio/mp3')
                            
                            with lang_col2:
                                hindi_text = f"यह {top_prediction['hindi']} का संकेत है, {top_prediction['percentage']:.1f} प्रतिशत विश्वास के साथ।"
                                if st.button("Hindi", use_container_width=True):
                                    audio_hi = text_to_speech(hindi_text, 'hi')
                                    if audio_hi:
                                        st.audio(audio_hi, format='audio/mp3')
                            
                            with lang_col3:
                                tamil_text = f"இது {top_prediction['tamil']} அடையாளம், {top_prediction['percentage']:.1f} சதவீத நம்பிக்கையுடன்."
                                if st.button("Tamil", use_container_width=True):
                                    audio_ta = text_to_speech(tamil_text, 'ta')
                                    if audio_ta:
                                        st.audio(audio_ta, format='audio/mp3')
    
    with col2:
        st.subheader("ℹ️ Prediction Guide")
        st.markdown("""
        ### What to expect:
        
        **High Confidence (>90%)**  
        • Clear, well-centered signs  
        • Good lighting conditions  
        • Standard German traffic signs  
        
        **Medium Confidence (70-90%)**  
        • Slightly angled signs  
        • Minor obstructions  
        • Different color schemes  
        
        **Low Confidence (<70%)**  
        • Blurry or distant signs  
        • Unusual perspectives  
        • Non-standard signs  
        
        ### Tips for best results:
        1. Center the sign in the image
        2. Ensure good lighting
        3. Use clear, high-quality images
        4. Avoid extreme angles
        5. Crop to focus on the sign
        """)

# ===========================================
# PAGE 3: STATISTICS
# ===========================================
elif page == "📊 Statistics":
    st.title("📊 Prediction Statistics")
    
    if st.session_state.stats['total_predictions'] == 0:
        st.info("No predictions made yet. Go to the **Predict** page to start!")
    else:
        # Overall stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", st.session_state.stats['total_predictions'])
        with col2:
            avg_confidence = st.session_state.stats['confidence_sum'] / st.session_state.stats['total_predictions']
            st.metric("Avg Confidence", f"{avg_confidence:.2f}%")
        with col3:
            st.metric("Unique Signs", 
                     sum(1 for count in st.session_state.stats['by_class'].values() if count > 0))
        with col4:
            most_common_id = max(st.session_state.stats['by_class'].items(), key=lambda x: x[1])[0]
            most_common_name = CLASSES[most_common_id]
            st.metric("Most Common", most_common_name.split('(')[0])
        
        # Category distribution
        st.subheader("📈 Predictions by Category")
        category_df = pd.DataFrame({
            'Category': list(st.session_state.stats['by_category'].keys()),
            'Count': list(st.session_state.stats['by_category'].values())
        })
        category_df = category_df[category_df['Count'] > 0]
        
        if not category_df.empty:
            fig1 = px.pie(category_df, values='Count', names='Category', 
                         title='Prediction Distribution by Category')
            st.plotly_chart(fig1, use_container_width=True)
        
        # Class distribution
        st.subheader("📊 Predictions by Sign Type")
        class_counts = {CLASSES[k]: v for k, v in st.session_state.stats['by_class'].items() if v > 0}
        if class_counts:
            class_df = pd.DataFrame({
                'Sign': list(class_counts.keys()),
                'Count': list(class_counts.values())
            }).sort_values('Count', ascending=False)
            
            fig2 = px.bar(class_df.head(10), x='Count', y='Sign', 
                         orientation='h', title='Top 10 Most Predicted Signs')
            st.plotly_chart(fig2, use_container_width=True)
            
            # Show all predictions
            with st.expander("View All Predictions"):
                st.dataframe(class_df, use_container_width=True)
        
        # Prediction history
        st.subheader("🕒 Recent Predictions")
        if st.session_state.predictions_history:
            history_df = pd.DataFrame(st.session_state.predictions_history)
            st.dataframe(history_df.sort_values('timestamp', ascending=False), 
                        use_container_width=True)
            
            # Download history
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="📥 Download History as CSV",
                data=csv,
                file_name=f"traffic_sign_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

# ===========================================
# PAGE 4: ABOUT
# ===========================================
else:
    st.title("ℹ️ About This System")
    
    st.markdown("""
    ## Traffic Sign Recognition System
    
    ### Model Information
    - **Dataset:** German Traffic Sign Recognition Benchmark (GTSRB)
    - **Classes:** 43 distinct traffic signs
    - **Model Architecture:** Custom CNN with 3 convolutional blocks
    - **Accuracy:** ~99% on test set
    - **Input Size:** 32×32 pixels
    - **Framework:** TensorFlow/Keras
    
    ### Class Categories
    
    #### Speed Limits (Classes 0-8)
    - 20 km/h to 120 km/h
    - End of speed limit restrictions
    
    #### Prohibitory Signs (Classes 9, 10, 15-17, 41-42)
    - No passing
    - No vehicles
    - No entry
    - Weight restrictions
    
    #### Mandatory Signs (Classes 11-14, 33-40)
    - Priority road
    - Yield
    - Stop
    - Direction arrows
    - Keep right/left
    
    #### Warning Signs (Classes 18-32)
    - Dangerous curves
    - Bumpy/slippery roads
    - Pedestrian crossings
    - Animal crossings
    - Road work
    
    ### Technical Details
    
    #### Model Architecture:
    ```
    Input (32, 32, 3)
    ↓
    Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout
    ↓
    Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout
    ↓
    Conv2D(128) → BatchNorm → Conv2D(128) → MaxPool → Dropout
    ↓
    Flatten → Dense(512) → Dropout → Dense(43) → Softmax
    ```
    
    #### Preprocessing:
    - Resize to 32×32 pixels
    - RGB color space
    - Normalize to [0, 1] range
    - Data augmentation (rotation, zoom, shift)
    
    ### Common Sign Mappings (FIXED!)
    
    | Class ID | Sign Name | Common Confusion | Status |
    |----------|-----------|------------------|--------|
    | 4 | **Speed limit (70km/h)** | Was showing as Yield | ✅ FIXED |
    | 13 | **Yield** | Was showing as 70 km/h | ✅ FIXED |
    | 17 | No entry | - | ✅ Correct |
    | 22 | Bumpy road | - | ✅ Correct |
    | 38 | Keep right | - | ✅ Correct |
    
    ### Development Team
    - **Model Training:** CNN on GTSRB dataset
    - **Web Interface:** Streamlit
    - **Multi-language:** gTTS for speech synthesis
    - **Visualization:** Plotly, Matplotlib
    
    ### Credits
    - Dataset: German Traffic Sign Recognition Benchmark
    - Icons: Streamlit emoji set
    - Speech: Google Text-to-Speech
    
    ---
    
    *For issues or suggestions, please contact the development team.*
    """)

# ===========================================
# FOOTER
# ===========================================
st.sidebar.markdown("---")
st.sidebar.caption(f"© 2024 Traffic Sign Recognizer | {len(CLASSES)} classes")

# Run with: streamlit run app.py
