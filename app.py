import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import json
import plotly.express as px
import re
import warnings

warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Drug Interaction Predictor",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS Styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E86AB;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.3rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .risk-critical {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(255,65,108,0.3);
        animation: pulse 2s infinite;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff9a56 0%, #ff6a00 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(255,154,86,0.3);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
        padding: 2rem;
        border-radius: 15px;
        color: #333;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(253,203,110,0.3);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0,184,148,0.3);
    }
    
    .input-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        border: 1px solid #dee2e6;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2E86AB;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .info-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .disclaimer {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 2rem 0;
        color: #856404;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 8px;
    }
    
    .stNumberInput > div > div {
        background-color: white;
        border-radius: 8px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.4);
    }
</style>
""", unsafe_allow_html=True)

def extract_dosage(composition):
    """Extract dosage information from drug composition"""
    if pd.isna(composition):
        return 0.0
    dosage_match = re.search(r'(\d+(?:\.\d+)?)\s*mg', str(composition).lower())
    return float(dosage_match.group(1)) if dosage_match else 0.0

@st.cache_data
def load_and_prepare_data():
    """Load the medicine dataset"""
    try:
        df = pd.read_csv('medicine_data_OG.csv')
        print(f"Loaded {len(df)} records")
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def prepare_dataset(df):
    """Prepare and expand the dataset for training"""
    expanded_rows = []
    
    for idx, row in df.iterrows():
        try:
            if pd.isna(row['drug_interactions']):
                continue
            
            interactions = json.loads(row['drug_interactions'])
            drugs = interactions.get('drug', [])
            effects = interactions.get('effect', [])
            
            if len(drugs) == len(effects) and len(drugs) > 0:
                for drug, effect in zip(drugs, effects):
                    expanded_rows.append({
                        'primary_drug': str(row['product_name']),
                        'interacting_drug': str(drug).strip(),
                        'risk_level': str(effect).strip(),
                        'category': str(row['sub_category']),
                        'dosage': extract_dosage(row['salt_composition'])
                    })
        except:
            continue
    
    result_df = pd.DataFrame(expanded_rows)
    print(f"Prepared {len(result_df)} interaction records")
    return result_df

def create_features(df):
    """Create encoded features for machine learning"""
    le_dict = {}
    
    for col in ['primary_drug', 'interacting_drug', 'category']:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    
    le_target = LabelEncoder()
    df['risk_encoded'] = le_target.fit_transform(df['risk_level'])
    le_dict['risk_level'] = le_target
    
    feature_columns = ['primary_drug_encoded', 'interacting_drug_encoded', 'category_encoded', 'dosage']
    X = df[feature_columns].values
    y = df['risk_encoded'].values
    
    return X, y, le_dict

def train_model(X, y):
    """Train the Random Forest model"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    rf = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.3f}")
    
    return rf, accuracy

@st.cache_resource
def load_trained_model():
    """Load and train the complete model pipeline"""
    df = load_and_prepare_data()
    if df is None:
        return None, None, None, None
    
    processed_df = prepare_dataset(df)
    if len(processed_df) == 0:
        return None, None, None, None
    
    X, y, le_dict = create_features(processed_df)
    model, accuracy = train_model(X, y)
    
    return model, le_dict, processed_df, accuracy

def predict_risk(model, primary_drug, interacting_drug, category, dosage, le_dict):
    """Predict drug interaction risk"""
    try:
        primary_encoded = le_dict['primary_drug'].transform([primary_drug])[0] if primary_drug in le_dict['primary_drug'].classes_ else 0
        interacting_encoded = le_dict['interacting_drug'].transform([interacting_drug])[0] if interacting_drug in le_dict['interacting_drug'].classes_ else 0
        category_encoded = le_dict['category'].transform([category])[0] if category in le_dict['category'].classes_ else 0
        
        features = np.array([[primary_encoded, interacting_encoded, category_encoded, dosage]])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        risk_level = le_dict['risk_level'].inverse_transform([prediction])[0]
        
        return risk_level, probabilities
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None, None

def display_risk_result(risk, probs, le_dict):
    """Display the risk prediction results with appropriate styling"""
    if risk == 'LIFE-THREATENING':
        st.markdown(f"""
        <div class="risk-critical">
            <h1>🚨 CRITICAL ALERT</h1>
            <h2>LIFE-THREATENING INTERACTION</h2>
            <p><strong>Risk Level: {risk}</strong></p>
            <p>⚠️ Extremely dangerous combination - Avoid immediately!</p>
            <p>🏥 Seek immediate medical attention if taken together</p>
        </div>
        """, unsafe_allow_html=True)
    
    elif risk == 'SERIOUS':
        st.markdown(f"""
        <div class="risk-high">
            <h1>⚠️ HIGH RISK WARNING</h1>
            <h2>SERIOUS INTERACTION DETECTED</h2>
            <p><strong>Risk Level: {risk}</strong></p>
            <p>🩺 Medical supervision required immediately</p>
            <p>📋 Consider alternative medications</p>
        </div>
        """, unsafe_allow_html=True)
    
    elif risk == 'MODERATE':
        st.markdown(f"""
        <div class="risk-medium">
            <h1>⚡ MODERATE RISK</h1>
            <h2>CAUTION ADVISED</h2>
            <p><strong>Risk Level: {risk}</strong></p>
            <p>👨‍⚕️ Monitor closely for side effects</p>
            <p>📞 Inform your healthcare provider</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.markdown(f"""
        <div class="risk-low">
            <h1>✅ LOW RISK</h1>
            <h2>MINOR INTERACTION</h2>
            <p><strong>Risk Level: {risk}</strong></p>
            <p>😊 Generally safe combination</p>
            <p>📝 Continue as prescribed, stay aware</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Probability visualization
    risk_levels = le_dict['risk_level'].classes_
    prob_df = pd.DataFrame({
        'Risk Level': risk_levels,
        'Probability': probs * 100
    }).sort_values('Probability', ascending=True)
    
    fig = px.bar(
        prob_df, 
        x='Probability', 
        y='Risk Level', 
        orientation='h',
        title="🎯 Risk Probability Analysis",
        labels={'Probability': 'Probability (%)', 'Risk Level': 'Risk Level'},
        color='Probability',
        color_continuous_scale=['#00b894', '#fdcb6e', '#ff6a00', '#ff416c']
    )
    fig.update_layout(
        height=400, 
        showlegend=False,
        title_font_size=18,
        title_x=0.5
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Confidence display
    confidence = np.max(probs) * 100
    st.markdown(f"""
    <div class="prediction-card">
        <h3>🎯 Prediction Confidence</h3>
        <h2>{confidence:.1f}%</h2>
        <p>Model confidence in the prediction</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    # Header Section
    st.markdown('<div class="main-header">💊 Drug Interaction Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Medical Risk Assessment System</div>', unsafe_allow_html=True)
    
    # Load model
    model, le_dict, df, accuracy = load_trained_model()
    
    if model is None:
        st.error("❌ Failed to load model. Please check medicine_data_OG.csv file.")
        return
    
    # Model Status Display
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div class="prediction-card">
            <h3>🎯 Model Status: Ready</h3>
            <h2>Accuracy: {accuracy:.1%}</h2>
            <p>Trained on {len(df)} drug interactions</p>
            <p>🤖 Random Forest Classifier</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Input Section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔍 Enter Drug Information for Risk Assessment</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏥 Primary Drug Information")
        primary_drug = st.selectbox(
            "Primary Drug", 
            options=sorted(df['primary_drug'].unique()),# type: ignore
            help="Select the main medication being taken"
        )
        
        interacting_drug = st.selectbox(
            "Interacting Drug", 
            options=sorted(df['interacting_drug'].unique()),# type: ignore
            help="Select the drug that may interact with the primary drug"
        )
    
    with col2:
        st.markdown("#### 📋 Additional Information")
        category = st.selectbox(
            "Drug Category", 
            options=sorted(df['category'].unique()), # type: ignore
            help="Medical category of the primary drug"
        )
        
        dosage = st.number_input(
            "Dosage (mg)", 
            min_value=0.0, 
            value=10.0, 
            step=0.1,
            help="Dosage amount in milligrams"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction Button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("🎯 Predict Risk Level", type="primary", use_container_width=True)
    
    # Prediction Results
    if predict_button:
        with st.spinner("🔄 Analyzing drug interaction..."):
            risk, probs = predict_risk(model, primary_drug, interacting_drug, category, dosage, le_dict)
        
        if risk:
            st.markdown("---")
            st.markdown('<div class="section-header">📊 Risk Assessment Results</div>', unsafe_allow_html=True)
            display_risk_result(risk, probs, le_dict)
        else:
            st.error("❌ Prediction failed. Please check your inputs and try again.")
    
    # Disclaimer Section
    st.markdown('<div class="disclaimer"><h4>⚠️ Important Medical Disclaimer</h4><p><strong>Educational Purpose Only:</strong> This tool is designed for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    