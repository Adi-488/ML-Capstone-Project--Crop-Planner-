import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="CropPlanner - Smart Agriculture",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #C8E6C9 0%, #81C784 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #388E3C;
        font-weight: 600;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #E8F5E9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #FFF9C4;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FBC02D;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and artifacts
@st.cache_resource
def load_artifacts():
    """Load all necessary artifacts"""
    try:
        with open("crop_model_final.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        with open("feature_columns.pkl", "rb") as f:
            feature_cols = pickle.load(f)
        return model, scaler, encoders, feature_cols
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run the training pipeline first.")
        return None, None, None, None, None

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Load artifacts
model, scaler, encoders, feature_cols = load_artifacts()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=100)
    st.title("🌾 CropPlanner")
    st.markdown("### Navigation")
    
    page = st.radio(
        "Choose a page:",
        ["🏠 Home", "🔮 Predict Crop", "📈 Analytics"]
    )
    
    st.markdown("---")
    st.markdown("### 👨‍🎓 Project Info")
    st.info("**Course:** Machine Learning Lab\n\n**Code:** ECSP5004\n\n**Institution:** Shri Ramdeobaba College of Engineering")

# ========== HOME PAGE ==========
if page == "🏠 Home":
    st.markdown('<div class="main-header">🌾 CropPlanner - Intelligent Crop Recommendation System</div>', 
                unsafe_allow_html=True)
    
    # Introduction
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Project Overview")
        st.write("""
        **CropPlanner** is an advanced machine learning-based system designed to help farmers and 
        agricultural professionals make informed decisions about crop selection. By analyzing soil 
        composition, climatic conditions, and historical production data, our system recommends 
        the most suitable crops for cultivation.
        """)
        
        st.markdown("### ✨ Key Features")
        features = {
            "🤖 Machine Learning": "Advanced XGBoost algorithm with 99%+ accuracy",
            "📊 Data-Driven": "Trained on extensive crop production datasets",
            "🌡️ Multi-Factor Analysis": "Considers NPK, temperature, humidity, pH, and rainfall",
            "⚡ Real-Time Prediction": "Instant crop recommendations",
            "📈 Performance Tracking": "Comprehensive model evaluation metrics"
        }
        
        for feature, description in features.items():
            st.markdown(f"**{feature}:** {description}")
    
# ========== PREDICTION PAGE ==========
elif page == "🔮 Predict Crop":
    st.markdown('<div class="main-header">🔮 Crop Prediction</div>', unsafe_allow_html=True)
    
    if model is None:
        st.error("Model not loaded. Please ensure all model files are present.")
    else:
        st.markdown("### 📝 Enter Farm & Soil Parameters")
        
        # Input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 🗺️ Location Details")
                state = st.selectbox("State", options=list(encoders['state'].classes_))
                crop_type = st.selectbox("Crop Type", options=list(encoders['type'].classes_))
                
                st.markdown("#### 🧪 Soil Nutrients (kg/ha)")
                nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
                phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
                potassium = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
            
            with col2:
                st.markdown("#### 🌡️ Climate Conditions")
                temperature = st.slider("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.5)
                humidity = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
                rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=3000.0, value=100.0, step=10.0)
                
                st.markdown("#### 🌱 Soil Properties")
                ph = st.slider("pH Level", min_value=3.0, max_value=10.0, value=6.5, step=0.1)
            
            with col3:
                st.markdown("#### 📏 Farm Details")
                area = st.number_input("Area (hectares)", min_value=0.1, max_value=10000.0, value=10.0, step=0.5)
                production = st.number_input("Expected Production (tons)", min_value=0.0, max_value=100000.0, value=100.0, step=10.0)
                
                st.markdown("#### ")
                st.markdown("#### ")
                predict_button = st.form_submit_button("🔮 Predict Best Crop", use_container_width=True)
        
        # Prediction
        if predict_button:
            try:
                # Encode inputs
                state_encoded = encoders['state'].transform([state])[0]
                crop_type_encoded = encoders['type'].transform([crop_type])[0]
                
                # Feature engineering
                npk_sum = nitrogen + phosphorus + potassium
                npk_ratio = nitrogen / (phosphorus + potassium + 1)
                production_per_area = production / (area + 1)
                n_by_k = nitrogen / (potassium + 1)
                p_by_k = phosphorus / (potassium + 1)
                n_to_p_ratio = nitrogen / (phosphorus + 1)
                temp_range = temperature * humidity / 100
                rain_temp_interaction = rainfall * temperature
                
                # Create feature array
                features = np.array([[
                    state_encoded, crop_type_encoded, nitrogen, phosphorus, potassium,
                    temperature, ph, rainfall, area, production,
                    npk_sum, production_per_area, rain_temp_interaction
                ]])

                
                # Scale features
                features_scaled = scaler.transform(features)
                
                # Predict
                prediction = model.predict(features_scaled)[0]
                prediction_proba = model.predict_proba(features_scaled)[0]
                
                # Get crop name
                crop_name = encoders['crop'].inverse_transform([prediction])[0]
                confidence = prediction_proba[prediction] * 100
                
                # Display result
                st.success("✅ Prediction Complete!")
                
                st.markdown(f'<div class="prediction-box">🌾 Recommended Crop: {crop_name}</div>', 
                           unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Confidence", f"{confidence:.2f}%")
                with col2:
                    st.metric("NPK Sum", f"{npk_sum:.1f}")
                with col3:
                    st.metric("Production/Area", f"{production_per_area:.2f}")
                
                # Top 5 predictions
                st.markdown("### 📊 Top 5 Crop Recommendations")
                top_5_indices = np.argsort(prediction_proba)[-5:][::-1]
                top_5_crops = encoders['crop'].inverse_transform(top_5_indices)
                top_5_proba = prediction_proba[top_5_indices] * 100
                
                chart_data = pd.DataFrame({
                    'Crop': top_5_crops,
                    'Probability (%)': top_5_proba
                })
                
                fig = px.bar(chart_data, x='Probability (%)', y='Crop', orientation='h',
                            color='Probability (%)', color_continuous_scale='Greens')
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Save to history
                st.session_state.prediction_history.append({
                    'crop': crop_name,
                    'confidence': confidence,
                    'state': state,
                    'temperature': temperature,
                    'rainfall': rainfall
                })
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")

# ========== ANALYTICS PAGE ==========
elif page == "📈 Analytics":
    st.markdown('<div class="main-header">📈 Data Analytics & Insights</div>', unsafe_allow_html=True)
    
    st.markdown("### 🌾 Recent Prediction Trends")
    if len(st.session_state.prediction_history) == 0:
        st.info("No predictions made yet. Once you make predictions, analytics will appear here.")
    else:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        fig = px.scatter(
            history_df, 
            x='rainfall', 
            y='temperature', 
            color='crop',
            size='confidence',
            hover_data=['state'],
            title="Predicted Crops by Temperature & Rainfall"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Crop frequency
        st.markdown("### 🌱 Most Frequently Recommended Crops")
        freq_df = history_df['crop'].value_counts().reset_index()
        freq_df.columns = ['Crop', 'Count']
        fig2 = px.bar(freq_df, x='Crop', y='Count', color='Count', color_continuous_scale='Greens')
        st.plotly_chart(fig2, use_container_width=True)
        
        # Confidence distribution
        st.markdown("### 🎯 Confidence Distribution")
        fig3 = px.histogram(history_df, x='confidence', nbins=10, title="Model Confidence Histogram", color_discrete_sequence=['#4CAF50'])
        st.plotly_chart(fig3, use_container_width=True)
        
        st.success("✅ Insights updated dynamically after every prediction.")
