import streamlit as st
import cv2
import pandas as pd
from datetime import datetime
import os

from src.predict import is_model_available, predict_image
from src.utils import load_image
from src.disease_info import DISEASE_INFO
from src.history import save_prediction

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Derma Scan AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# CUSTOM STYLING (Aesthetic enhancements)
# ==========================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main banner gradient */
    .banner {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 30px;
        border-radius: 18px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px rgba(56, 239, 125, 0.2);
    }
    
    .banner h1 {
        margin: 0;
        font-weight: 700;
        font-size: 2.8rem;
        color: white !important;
    }
    
    .banner p {
        margin: 10px 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }

    /* Card styling */
    .card {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #eef2f5;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.08);
    }

    /* Dark mode override for card styling */
    @media (prefers-color-scheme: dark) {
        .card {
            background: #1e293b;
            border: 1px solid #334155;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        }
    }

    /* Section titles */
    .section-title {
        font-weight: 600;
        font-size: 1.5rem;
        margin-top: 0;
        margin-bottom: 15px;
        border-left: 5px solid #11998e;
        padding-left: 10px;
    }
    
    /* Highlight boxes */
    .highlight-box {
        background-color: #f0fdf4;
        border-left: 4px solid #16a34a;
        padding: 15px;
        border-radius: 0 12px 12px 0;
        margin-bottom: 15px;
        color: #166534;
        font-weight: 500;
    }
    
    @media (prefers-color-scheme: dark) {
        .highlight-box {
            background-color: #064e3b;
            border-left: 4px solid #34d399;
            color: #ecfdf5;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================
# SIDEBAR
# ==========================
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <span style="font-size: 4rem;">🩺</span>
            <h2 style="margin: 10px 0 0 0; font-weight: 700;">Derma Scan AI</h2>
            <p style="color: #64748b; font-size: 0.95rem;">Skin Lesion Analyzer</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Model Status
    st.markdown("### 🤖 System Diagnostics")
    model_loaded = is_model_available()
    if model_loaded:
        st.success("ML Engine: Active (Random Forest)")
    else:
        st.warning("ML Engine: Heuristics Fallback")
        
    st.markdown("---")
    
    st.markdown(
        """
        ### 🧪 Classifier Capabilities
        The AI model classifies images into seven types of lesions:
        - **Melanoma** (Malignant)
        - **Melanocytic Nevi** (Common Mole)
        - **Benign Keratosis** (Solar Lentigo/Seborrheic)
        - **Basal Cell Carcinoma** (Common Cancer)
        - **Actinic Keratosis** (Pre-cancerous)
        - **Vascular Lesion** (Angiomas etc.)
        - **Dermatofibroma** (Fibrous Nodule)
        
        ---
        ### 🛠 Technologies
        - Python 3.x
        - Scikit-learn
        - OpenCV
        - Streamlit
        - Pandas & NumPy
        - Scikit-image (HOG)
        """
    )

# ==========================
# HEADER BANNER
# ==========================
st.markdown(
    """
    <div class="banner">
        <h1>🩺 Derma Scan AI</h1>
        <p>Advanced Non-Invasive Skin Lesion Classification System</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================
# IMAGE UPLOAD SECTION
# ==========================
st.markdown("<div class='card'><h3 class='section-title'>📤 Image Upload</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload a high-resolution skin lesion image (JPEG, JPG, PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)
st.markdown("</div>", unsafe_allow_html=True)

# ==========================
# MAIN PREDICTION PIPELINE
# ==========================
if uploaded_file is not None:
    # Load and process image
    image = load_image(uploaded_file)

    if image is None:
        st.error("The uploaded file could not be read as an image. Please verify it is a valid, uncorrupted image file.")
    else:
        # Convert BGR to RGB for rendering
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown("<div class='card'><h3 class='section-title'>🖼 Uploaded Image</h3>", unsafe_allow_html=True)
            st.image(
                rgb,
                use_container_width=True,
                caption=f"Source: {uploaded_file.name}"
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='card'><h3 class='section-title'>🔬 Prediction Engine</h3>", unsafe_allow_html=True)
            
            # Display engine details
            if not model_loaded:
                st.info("💡 Note: The trained model assets were not found in the `models/` folder. The app is utilizing a local rule-based heuristic fallback.")
            else:
                st.success("⚡ Ready: Machine learning model successfully loaded and primed.")

            st.write("")
            
            # Action Button
            if st.button("🔍 Analyze Lesion", use_container_width=True):
                with st.spinner("Analyzing image features and running prediction..."):
                    # Execute prediction
                    disease, confidence = predict_image(image)
                
                # Success Prediction Display
                st.markdown(
                    f"""
                    <div class='highlight-box'>
                        🎉 Analysis Result: {disease}<br/>
                        📊 Confidence Score: {confidence:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Confidence Progress
                st.progress(int(confidence))

                # Save Prediction to History
                save_prediction(
                    uploaded_file.name,
                    disease,
                    confidence
                )

                # Fetch Disease Info Details
                if disease in DISEASE_INFO:
                    info = DISEASE_INFO[disease]
                    
                    st.markdown("<h4 style='margin-top: 20px;'>📖 Pathology Description</h4>", unsafe_allow_html=True)
                    st.write(info["description"])

                    st.markdown("<h4 style='margin-top: 15px;'>🛡 Recommended Precautions</h4>", unsafe_allow_html=True)
                    for item in info["precautions"]:
                        st.markdown(f"✅ {item}")
                else:
                    st.warning("⚠️ No detailed information is available for this class.")

                # Generate Report
                report_content = f"""DERMA SCAN AI REPORT
====================================
Date      : {datetime.now().strftime('%Y-%m-%d')}
Time      : {datetime.now().strftime('%H:%M:%S')}
Image     : {uploaded_file.name}
------------------------------------
Prediction: {disease}
Confidence: {confidence:.2f}%
------------------------------------
DISCLAIMER: This analysis report is generated by a Machine Learning model. It is designed for educational/research support and does not constitute formal medical advice. Please consult a professional dermatologist for diagnosis.
"""
                st.write("")
                st.download_button(
                    label="📥 Download Clinical Report",
                    data=report_content,
                    file_name=f"DermaScan_Report_{uploaded_file.name.split('.')[0]}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            st.markdown("</div>", unsafe_allow_html=True)

# ==========================
# HISTORY TRACKING SECTION
# ==========================
st.markdown("<div class='card'><h3 class='section-title'>📜 Prediction History</h3>", unsafe_allow_html=True)

history_file = "reports/prediction_history.csv"
if os.path.exists(history_file):
    try:
        # Load and reverse so latest is on top
        history_df = pd.read_csv(history_file).iloc[::-1]
        
        # Display as styled dataframe
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )
    except Exception as e:
        st.error(f"Error reading prediction history: {e}")
else:
    st.info("No prediction history recorded yet. Run a prediction to start tracking history.")

st.markdown("</div>", unsafe_allow_html=True)

# ==========================
# FOOTER
# ==========================
st.markdown(
    """
    <div style="text-align: center; margin-top: 40px; padding: 20px; border-top: 1px solid #e2e8f0; color: #94a3b8; font-size: 0.9rem;">
        <p>Developed for Final Year Project &copy; 2026 | <b>Derma Scan AI</b></p>
        <p style="font-size: 0.8rem; margin-top: 5px;">Disclaimer: This application uses AI for skin lesion evaluation. It is not a replacement for professional clinical advice.</p>
    </div>
    """,
    unsafe_allow_html=True
)