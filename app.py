"""
Hybrid Quantum-Classical Irrigation Prediction Web App
=======================================================
A Streamlit application for predicting irrigation requirements using
a hybrid VQC (Variational Quantum Classifier) + Random Forest model.

Usage:
    streamlit run app.py

Author: Smart Irrigation System
"""

import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import required classes for model unpickling
try:
    import pennylane as qml
    from pennylane import numpy as pnp
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    st.warning("⚠️ PennyLane not installed. Some features may be limited.")

# ==================== MODEL CLASSES (Required for unpickling) ====================
class QuantumIrrigationVQC:
    """Variational Quantum Classifier for Irrigation Prediction"""
    
    def __init__(self, n_qubits=4, n_layers=3, n_features=3, learning_rate=0.01):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_features = n_features
        self.learning_rate = learning_rate
        
        if PENNYLANE_AVAILABLE:
            # Initialize quantum device
            self.dev = qml.device("default.qubit", wires=n_qubits)
            
            # Initialize components
            self.weights = None
            self.scaler = MinMaxScaler(feature_range=(0, np.pi))
            self.training_costs = []
            self.is_trained = False
            self.training_time = 0
            
            # Create quantum circuit
            self.quantum_circuit = qml.QNode(
                self._circuit, 
                self.dev, 
                diff_method="parameter-shift"
            )
    
    def _feature_encoding(self, x):
        """Angle encoding for features"""
        for i in range(min(self.n_features, self.n_qubits)):
            qml.RY(x[i], wires=i)
    
    def _variational_ansatz(self, weights):
        """Variational ansatz with RY rotations and CNOT entanglement"""
        for layer in range(self.n_layers):
            # Rotation layer
            for qubit in range(self.n_qubits):
                qml.RY(weights[layer, qubit], wires=qubit)
            
            # Entanglement layer
            for qubit in range(self.n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])
            
            # Ring entanglement
            if self.n_qubits > 1:
                qml.CNOT(wires=[self.n_qubits - 1, 0])
    
    def _circuit(self, x, weights):
        """Complete quantum circuit"""
        self._feature_encoding(x)
        self._variational_ansatz(weights)
        return qml.expval(qml.PauliZ(0))
    
    def _circuit_all_qubits(self, x, weights):
        """Circuit that returns expectation values from all qubits"""
        self._feature_encoding(x)
        self._variational_ansatz(weights)
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def extract_quantum_features(self, X):
        """Extract quantum features (expectation values from all qubits)"""
        if not self.is_trained:
            raise ValueError("VQC must be trained before extracting features")
        
        X_scaled = self.scaler.transform(X)
        quantum_features = []
        
        # Create circuit that measures all qubits
        circuit_all = qml.QNode(self._circuit_all_qubits, self.dev)
        
        for x_sample in X_scaled:
            try:
                expectations = circuit_all(x_sample, self.weights)
                quantum_features.append(expectations)
            except Exception as e:
                print(f"Feature extraction error: {e}")
                quantum_features.append([0.0] * self.n_qubits)
        
        return np.array(quantum_features)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        probabilities = []
        
        for x_sample in X_scaled:
            try:
                expectation = self.quantum_circuit(x_sample, self.weights)
                prob = (expectation + 1) / 2  # Convert to probability
                prob = np.clip(prob, 0.001, 0.999)
                probabilities.append(float(prob))
            except Exception as e:
                print(f"Prediction error: {e}")
                probabilities.append(0.5)
        
        return np.array(probabilities)
    
    def predict(self, X):
        """Predict binary classes"""
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)


class HybridQuantumClassicalModel:
    """Hybrid model combining VQC quantum features with Random Forest"""
    
    def __init__(self, vqc_model, rf_model, feature_scaler=None):
        self.vqc_model = vqc_model
        self.rf_model = rf_model
        self.feature_scaler = feature_scaler
        
    def predict(self, X):
        """Make predictions using hybrid features"""
        # Extract quantum features
        quantum_features = self.vqc_model.extract_quantum_features(X)
        
        # Combine with classical features
        hybrid_features = np.concatenate([X, quantum_features], axis=1)
        
        # Scale if scaler exists
        if self.feature_scaler is not None:
            hybrid_features = self.feature_scaler.transform(hybrid_features)
        
        return self.rf_model.predict(hybrid_features)
    
    def predict_proba(self, X):
        """Predict probabilities using hybrid features"""
        quantum_features = self.vqc_model.extract_quantum_features(X)
        hybrid_features = np.concatenate([X, quantum_features], axis=1)
        
        if self.feature_scaler is not None:
            hybrid_features = self.feature_scaler.transform(hybrid_features)
        
        return self.rf_model.predict_proba(hybrid_features)


# ==================== CONFIGURATION ====================
CONFIG = {
    'model_path': 'models/hybrid_irrigation_model.pkl',
    'feature_names': ['Soil Moisture', 'Temperature', 'Air Humidity'],
    'feature_ranges': {
        'Soil Moisture': (0, 1000),
        'Temperature': (-10, 50),
        'Air Humidity': (0, 100)
    },
    'feature_defaults': {
        'Soil Moisture': 600.0,
        'Temperature': 25.0,
        'Air Humidity': 60.0
    },
    'feature_units': {
        'Soil Moisture': 'units',
        'Temperature': '°C',
        'Air Humidity': '%'
    },
    'app_title': '🌱 Smart Irrigation Predictor',
    'app_description': 'Hybrid Quantum-Classical AI Model for Irrigation Decision Making'
}


# ==================== UTILITY FUNCTIONS ====================
@st.cache_resource
def load_model():
    """Load the trained hybrid model with caching"""
    model_path = Path(CONFIG['model_path'])
    
    if not model_path.exists():
        return None, f"Model file '{CONFIG['model_path']}' not found"
    
    if not PENNYLANE_AVAILABLE:
        return None, "PennyLane is required but not installed. Run: pip install pennylane"
    
    try:
        model = joblib.load(model_path)
        return model, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


def create_confidence_gauge(confidence, prediction):
    """Create a Plotly gauge chart for confidence visualization"""
    
    # Determine color based on confidence level
    if confidence >= 80:
        color = "#10b981"  # Green
    elif confidence >= 60:
        color = "#f59e0b"  # Orange
    else:
        color = "#ef4444"  # Red
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Prediction Confidence", 'font': {'size': 20, 'color': '#1f2937'}},
        number={'suffix': "%", 'font': {'size': 40, 'color': '#1f2937'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#6b7280"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, 50], 'color': '#fee2e2'},
                {'range': [50, 70], 'color': '#fef3c7'},
                {'range': [70, 100], 'color': '#d1fae5'}
            ],
            'threshold': {
                'line': {'color': "#1f2937", 'width': 4},
                'thickness': 0.75,
                'value': 75
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'family': "Arial, sans-serif"}
    )
    
    return fig


def create_feature_importance_chart(features, values):
    """Create a bar chart showing input feature values"""
    
    # Normalize values to 0-100 scale for visualization
    normalized_values = []
    for feature, value in zip(features, values):
        min_val, max_val = CONFIG['feature_ranges'][feature]
        normalized = ((value - min_val) / (max_val - min_val)) * 100
        normalized_values.append(normalized)
    
    colors = ['#3b82f6', '#10b981', '#f59e0b']
    
    fig = go.Figure(data=[
        go.Bar(
            x=features,
            y=normalized_values,
            text=[f"{v:.1f}" for v in values],
            textposition='auto',
            marker_color=colors,
            hovertemplate='<b>%{x}</b><br>Value: %{text}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Input Feature Values (Normalized Scale)",
        xaxis_title="Features",
        yaxis_title="Normalized Value (0-100)",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'family': "Arial, sans-serif", 'color': '#1f2937'},
        showlegend=False
    )
    
    fig.update_yaxes(range=[0, 100], gridcolor='#e5e7eb')
    fig.update_xaxes(gridcolor='#e5e7eb')

    
    return fig


def get_recommendation(prediction, confidence, soil_moisture, temperature, humidity):
    """Generate irrigation recommendation based on prediction and conditions"""
    
    if prediction == 1:  # Pump ON
        base_msg = "💧 **Irrigation Recommended**"
        
        reasons = []
        if soil_moisture < 500:
            reasons.append("Low soil moisture detected")
        if temperature > 30:
            reasons.append("High temperature increases evaporation")
        if humidity < 50:
            reasons.append("Low humidity increases water demand")
        
        if reasons:
            reason_text = "\n- " + "\n- ".join(reasons)
        else:
            reason_text = "\n- Optimal irrigation conditions detected"
        
        if confidence >= 80:
            confidence_msg = "✅ High confidence prediction"
        elif confidence >= 60:
            confidence_msg = "⚠️ Moderate confidence - monitor conditions"
        else:
            confidence_msg = "⚠️ Low confidence - verify with manual inspection"
        
        return f"{base_msg}\n\n**Reasons:**{reason_text}\n\n{confidence_msg}"
    
    else:  # Pump OFF
        base_msg = "🌤️ **No Irrigation Required**"
        
        reasons = []
        if soil_moisture > 700:
            reasons.append("Adequate soil moisture levels")
        if temperature < 25:
            reasons.append("Moderate temperature conditions")
        if humidity > 60:
            reasons.append("High humidity reduces water loss")
        
        if reasons:
            reason_text = "\n- " + "\n- ".join(reasons)
        else:
            reason_text = "\n- Current conditions do not require irrigation"
        
        if confidence >= 80:
            confidence_msg = "✅ High confidence prediction"
        elif confidence >= 60:
            confidence_msg = "⚠️ Moderate confidence - continue monitoring"
        else:
            confidence_msg = "⚠️ Low confidence - verify with manual inspection"
        
        return f"{base_msg}\n\n**Reasons:**{reason_text}\n\n{confidence_msg}"


# ==================== MAIN APP ====================
def main():
    """Main application function"""
    
    # Page configuration
    st.set_page_config(
        page_title=CONFIG['app_title'],
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            padding: 0rem 1rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #10b981;
            color: white;
            font-size: 18px;
            font-weight: bold;
            padding: 0.75rem 2rem;
            border-radius: 10px;
            border: none;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #059669;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .prediction-box {
            padding: 2rem;
            border-radius: 10px;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .pump-on {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .pump-off {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }
        h1 {
            color: #1f2937;
            font-weight: 800;
        }
        h2, h3 {
            color: #374151;
        }
        .stAlert {
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title(CONFIG['app_title'])
    st.markdown(f"**{CONFIG['app_description']}**")
    st.markdown("---")
    
    # Load model
    model, error = load_model()
    
    if error:
        st.error(f"❌ **{error}**")
        st.warning("""
        **Model not found!** Please ensure you have:
        1. Trained the hybrid model by running `training.py`
        2. The file `hybrid_irrigation_model.pkl` exists in the same directory as this app
        3. All required dependencies are installed
        """)
        st.info("""
        **Install dependencies:**
        ```bash
        pip install streamlit numpy pandas scikit-learn pennylane joblib plotly
        ```
        
        **Training the model:**
        ```bash
        python training.py
        ```
        
        **Running this app:**
        ```bash
        streamlit run app.py
        ```
        """)
        return
    
    st.success("✅ Hybrid Quantum-Classical Model Loaded Successfully!")
    
    # Sidebar - Model Information
    with st.sidebar:
        st.header("📊 Model Information")
        st.markdown("""
        **Architecture:**
        - 🔬 VQC Quantum Feature Extraction
        - 🌲 Random Forest Classifier
        - 🤝 Hybrid Quantum-Classical Approach
        
        **Features:**
        - Soil Moisture (0-1000 units)
        - Temperature (-10 to 50°C)
        - Air Humidity (0-100%)
        """)
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.markdown("""
        This application uses a hybrid quantum-classical machine learning model 
        to predict irrigation requirements based on environmental conditions.
        
        The model combines:
        - **VQC**: Variational Quantum Classifier for quantum feature extraction
        - **Random Forest**: Classical ML for final classification
        """)
        
        st.markdown("---")
        st.markdown("**Developed by:** Smart Irrigation Team")
        st.markdown("**Model Version:** 1.0")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🎛️ Input Parameters")
        
        # Input fields with descriptions
        st.markdown("### Soil Moisture")
        soil_moisture = st.slider(
            "Soil moisture level (lower = drier soil)",
            min_value=float(CONFIG['feature_ranges']['Soil Moisture'][0]),
            max_value=float(CONFIG['feature_ranges']['Soil Moisture'][1]),
            value=CONFIG['feature_defaults']['Soil Moisture'],
            step=10.0,
            help="Soil moisture sensor reading. Lower values indicate drier soil requiring irrigation."
        )
        
        st.markdown("### Temperature")
        temperature = st.slider(
            "Air temperature in degrees Celsius",
            min_value=float(CONFIG['feature_ranges']['Temperature'][0]),
            max_value=float(CONFIG['feature_ranges']['Temperature'][1]),
            value=CONFIG['feature_defaults']['Temperature'],
            step=0.5,
            help="Current air temperature. Higher temperatures increase water evaporation."
        )
        
        st.markdown("### Air Humidity")
        humidity = st.slider(
            "Relative air humidity percentage",
            min_value=float(CONFIG['feature_ranges']['Air Humidity'][0]),
            max_value=float(CONFIG['feature_ranges']['Air Humidity'][1]),
            value=CONFIG['feature_defaults']['Air Humidity'],
            step=1.0,
            help="Relative humidity in the air. Lower humidity increases plant water demand."
        )
        
        st.markdown("---")
        
        # Predict button
        predict_button = st.button("🔮 Predict Irrigation Requirement", type="primary")
    
    with col2:
        st.header("📈 Prediction Results")
        
        if predict_button:
            # Prepare input data
            input_data = np.array([[soil_moisture, temperature, humidity]])
            
            # Show loading animation
            with st.spinner("🔄 Processing with hybrid quantum-classical model..."):
                try:
                    # Make prediction
                    prediction = model.predict(input_data)[0]
                    probabilities = model.predict_proba(input_data)[0]
                    
                    # Get confidence
                    confidence = probabilities[prediction] * 100
                    
                    # Display prediction
                    st.markdown("### Prediction Result")
                    
                    if prediction == 1:
                        st.markdown(
                            f'<div class="prediction-box pump-on">'
                            f'<h2>💧 PUMP ON</h2>'
                            f'<p style="font-size: 18px;">Irrigation is required based on current conditions.</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="prediction-box pump-off">'
                            f'<h2>🌤️ PUMP OFF</h2>'
                            f'<p style="font-size: 18px;">No irrigation needed at this time.</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Display confidence gauge
                    st.plotly_chart(
                        create_confidence_gauge(confidence, prediction),
                        use_container_width=True
                    )
                    
                    # Display recommendation
                    st.markdown("### 💡 Recommendation")
                    recommendation = get_recommendation(
                        prediction, confidence, soil_moisture, temperature, humidity
                    )
                    st.info(recommendation)
                    
                    # Display probabilities
                    st.markdown("### 📊 Class Probabilities")
                    prob_col1, prob_col2 = st.columns(2)
                    with prob_col1:
                        st.metric(
                            "Pump OFF Probability",
                            f"{probabilities[0]*100:.1f}%"
                        )
                    with prob_col2:
                        st.metric(
                            "Pump ON Probability",
                            f"{probabilities[1]*100:.1f}%"
                        )
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
                    st.exception(e)
        else:
            st.info("👈 Adjust the input parameters and click **Predict** to get irrigation recommendations.")
            
            # Show example prediction
            st.markdown("### 📝 Example Scenarios")
            st.markdown("""
            **Scenario 1: Dry Conditions**
            - Soil Moisture: 300
            - Temperature: 35°C
            - Humidity: 40%
            - Expected: **PUMP ON** 💧
            
            **Scenario 2: Wet Conditions**
            - Soil Moisture: 850
            - Temperature: 20°C
            - Humidity: 80%
            - Expected: **PUMP OFF** 🌤️
            
            **Scenario 3: Moderate Conditions**
            - Soil Moisture: 600
            - Temperature: 25°C
            - Humidity: 60%
            - Prediction varies based on hybrid model
            """)
    
    # Feature visualization
    st.markdown("---")
    st.header("📊 Input Feature Visualization")
    st.plotly_chart(
        create_feature_importance_chart(
            CONFIG['feature_names'],
            [soil_moisture, temperature, humidity]
        ),
        use_container_width=True
    )
    
    # Additional information
    st.markdown("---")
    st.markdown("""
    ### 🔬 How It Works
    
    This application uses a **Hybrid Quantum-Classical Machine Learning Model** that combines:
    
    1. **Variational Quantum Classifier (VQC)**: Extracts quantum features from input data using quantum circuits
    2. **Random Forest Classifier**: Processes both classical and quantum features for final prediction
    
    The hybrid approach leverages quantum computing's pattern recognition capabilities while maintaining 
    the robustness of classical machine learning.
    
    ---
    
    ### 📖 Feature Descriptions
    
    - **Soil Moisture**: Sensor reading indicating soil water content (0 = very dry, 1000 = very wet)
    - **Temperature**: Ambient air temperature affecting evaporation rates
    - **Air Humidity**: Relative humidity affecting plant transpiration and water demand
    
    ---
    
    ### ⚙️ Technical Details
    
    - **Model Type**: Hybrid VQC + Random Forest
    - **Quantum Qubits**: 4
    - **Quantum Layers**: 3
    - **RF Estimators**: 100
    - **Training Accuracy**: Varies based on dataset
    
    ---
    
    <div style="text-align: center; padding: 2rem; background-color: #f3f4f6; border-radius: 10px; margin-top: 2rem;">
        <p style="color: #6b7280; font-size: 14px; margin: 0;">
            Developed by <strong>Smart Irrigation Team</strong> | Powered by Quantum Machine Learning
        </p>
        <p style="color: #9ca3af; font-size: 12px; margin-top: 0.5rem;">
            © 2025 Hybrid Quantum-Classical Irrigation System | Version 1.0
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()