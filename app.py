import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os
import warnings
warnings.filterwarnings('ignore')

# Import PennyLane for VQC model
try:
    import pennylane as qml
    from pennylane import numpy as pnp
    from sklearn.preprocessing import MinMaxScaler
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    st.warning("⚠️ PennyLane not installed. VQC model will not be available.")


# VQC Class Definition (must match the training.py version)
class QuantumIrrigationVQC:
    """Variational Quantum Classifier for Irrigation Prediction"""
    
    def __init__(self, n_qubits=4, n_layers=3, n_features=3, learning_rate=0.01):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_features = n_features
        self.learning_rate = learning_rate
        
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
    
    def _prediction_from_expectation(self, expectation_val):
        """Convert expectation value [-1, 1] to probability [0, 1]"""
        probability = (expectation_val + 1) / 2
        return np.clip(probability, 0.001, 0.999)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        probabilities = []
        
        for x_sample in X_scaled:
            try:
                expectation = self.quantum_circuit(x_sample, self.weights)
                prob = self._prediction_from_expectation(expectation)
                probabilities.append(float(prob))
            except Exception as e:
                probabilities.append(0.5)
        
        return np.array(probabilities)
    
    def predict(self, X):
        """Predict binary classes"""
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)

# Page configuration
st.set_page_config(
    page_title="Smart Irrigation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    .prediction-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .highlight-best {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #2196F3;
        margin: 10px 0;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        padding: 20px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load all trained models and scalers"""
    models = {}
    
    # Get the directory where app.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, 'models')
    
    try:
        # Check if models directory exists
        if not os.path.exists(model_dir):
            st.error(f"Models directory not found: {model_dir}")
            st.info("Please create a 'models' folder in the same directory as app.py")
            return {}
        
        # Load VQC
        vqc_model_path = os.path.join(model_dir, 'vqc_model.pkl')
        vqc_scaler_path = os.path.join(model_dir, 'vqc_scaler.pkl')
        if os.path.exists(vqc_model_path) and PENNYLANE_AVAILABLE:
            models['VQC'] = {
                'model': joblib.load(vqc_model_path),
                'scaler': joblib.load(vqc_scaler_path) if os.path.exists(vqc_scaler_path) else None,
                'type': 'quantum'
            }
        elif os.path.exists(vqc_model_path) and not PENNYLANE_AVAILABLE:
            st.warning("VQC model found but PennyLane is not installed. Skipping VQC.")
        
        # Load Logistic Regression
        lr_model_path = os.path.join(model_dir, 'logistic_regression_model.pkl')
        lr_scaler_path = os.path.join(model_dir, 'logistic_regression_scaler.pkl')
        if os.path.exists(lr_model_path):
            models['Logistic Regression'] = {
                'model': joblib.load(lr_model_path),
                'scaler': joblib.load(lr_scaler_path) if os.path.exists(lr_scaler_path) else None,
                'type': 'classical'
            }
        
        # Load SVM
        svm_model_path = os.path.join(model_dir, 'svm_model.pkl')
        svm_scaler_path = os.path.join(model_dir, 'svm_scaler.pkl')
        if os.path.exists(svm_model_path):
            models['SVM'] = {
                'model': joblib.load(svm_model_path),
                'scaler': joblib.load(svm_scaler_path) if os.path.exists(svm_scaler_path) else None,
                'type': 'classical'
            }
        
        # Load Random Forest
        rf_model_path = os.path.join(model_dir, 'random_forest_model.pkl')
        if os.path.exists(rf_model_path):
            models['Random Forest'] = {
                'model': joblib.load(rf_model_path),
                'scaler': None,
                'type': 'classical'
            }
        
        return models
    
    except Exception as e:
        st.error(f"Error loading models: {e}")
        import traceback
        st.error(traceback.format_exc())
        return {}


def make_predictions(models, soil_moisture, temperature, humidity):
    """Make predictions using all loaded models"""
    X_input = np.array([[soil_moisture, temperature, humidity]])
    predictions = {}
    
    for model_name, model_info in models.items():
        try:
            model = model_info['model']
            scaler = model_info['scaler']
            
            # Scale input if scaler exists
            if scaler is not None:
                X_scaled = scaler.transform(X_input)
            else:
                X_scaled = X_input
            
            # Get prediction and probability
            if model_name == 'VQC':
                # VQC has custom predict_proba method
                prob = model.predict_proba(X_input)[0]
                pred = 1 if prob > 0.5 else 0
            else:
                pred = model.predict(X_scaled)[0]
                if hasattr(model, 'predict_proba'):
                    prob_array = model.predict_proba(X_scaled)[0]
                    prob = prob_array[1]  # Probability of class 1
                else:
                    prob = float(pred)
            
            predictions[model_name] = {
                'prediction': int(pred),
                'confidence': float(prob) if pred == 1 else float(1 - prob),
                'probability': float(prob),
                'type': model_info['type']
            }
        
        except Exception as e:
            st.warning(f"Error with {model_name}: {e}")
            predictions[model_name] = {
                'prediction': 0,
                'confidence': 0.5,
                'probability': 0.5,
                'type': model_info['type'],
                'error': str(e)
            }
    
    return predictions


def plot_confidence_comparison(predictions):
    """Create interactive bar chart comparing model confidences"""
    model_names = list(predictions.keys())
    confidences = [pred['confidence'] * 100 for pred in predictions.values()]
    colors = ['#FF6B6B' if pred['type'] == 'quantum' else '#4ECDC4' 
              for pred in predictions.values()]
    
    fig = go.Figure(data=[
        go.Bar(
            x=model_names,
            y=confidences,
            marker_color=colors,
            text=[f"{conf:.1f}%" for conf in confidences],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': '📊 Model Confidence Comparison',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50', 'family': 'Arial Black'}
        },
        xaxis_title='Model',
        yaxis_title='Confidence (%)',
        yaxis_range=[0, 105],
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400,
        showlegend=False,
        font=dict(size=12),
        hovermode='x'
    )
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=0.5)
    
    return fig


def plot_prediction_gauge(prediction, confidence):
    """Create gauge chart for prediction visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Pump Status" if prediction == 1 else "No Irrigation", 
               'font': {'size': 24, 'color': '#2c3e50'}},
        delta={'reference': 50, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#4CAF50" if prediction == 1 else "#FF5722"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffebee'},
                {'range': [50, 75], 'color': '#fff3e0'},
                {'range': [75, 100], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#2c3e50", 'family': "Arial"}
    )
    
    return fig


def main():
    # Header
    st.markdown("""
        <h1>🌾 Quantum-Enhanced Smart Irrigation System</h1>
        <p style='text-align: center; color: #7f8c8d; font-size: 18px;'>
            AI-Powered Precision Agriculture | Quantum + Classical ML
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load models
    with st.spinner("🔄 Loading AI models..."):
        models = load_models()
    
    if not models:
        st.error("❌ **No models found!** Please ensure model files are in the 'models/' directory.")
        st.info("""
        **Required files:**
        - `vqc_model.pkl` and `vqc_scaler.pkl`
        - `logistic_regression_model.pkl` and `logistic_regression_scaler.pkl`
        - `svm_model.pkl` and `svm_scaler.pkl`
        - `random_forest_model.pkl`
        """)
        return
    
    st.success(f"✅ Successfully loaded **{len(models)}** models")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("🎛️ Input Parameters")
        st.markdown("Adjust the sliders to set environmental conditions:")
        
        # Check if example scenario was clicked
        if 'example' in st.session_state:
            default_soil = st.session_state.example['soil']
            default_temp = st.session_state.example['temp']
            default_hum = st.session_state.example['hum']
            del st.session_state.example
        else:
            default_soil = 600.0
            default_temp = 28.0
            default_hum = 60.0
        
        st.markdown("### 💧 Soil Moisture")
        soil_moisture = st.slider(
            "Soil Moisture",
            min_value=350.0,
            max_value=970.0,
            value=default_soil,
            step=1.0,
            help="Soil moisture reading (350-970 range from dataset)"
        )
        
        st.markdown("### 🌡️ Temperature")
        temperature = st.slider(
            "Temperature (°C)",
            min_value=18.0,
            max_value=39.0,
            value=default_temp,
            step=0.1,
            help="Ambient air temperature in Celsius"
        )
        
        st.markdown("### 💨 Air Humidity")
        humidity = st.slider(
            "Air Humidity (%)",
            min_value=38.0,
            max_value=82.0,
            value=default_hum,
            step=0.5,
            help="Relative humidity percentage in the air"
        )
        
        st.markdown("---")
        predict_button = st.button("🚀 PREDICT PUMP STATUS", use_container_width=True)
        
        st.markdown("---")
        st.markdown("""
        <div class='info-box'>
        <b>ℹ️ About</b><br>
        This system uses quantum and classical machine learning to predict whether the irrigation pump should be ON or OFF based on environmental conditions.
        <br><br>
        <b>Dataset Ranges:</b><br>
        • Soil Moisture: 350-970<br>
        • Temperature: 18-39°C<br>
        • Humidity: 38-82%
        </div>
        """, unsafe_allow_html=True)
        
        # Add example scenarios
        st.markdown("---")
        st.markdown("### 📋 Example Scenarios")
        
        if st.button("🌵 Dry Conditions", use_container_width=True):
            st.session_state.example = {'soil': 400.0, 'temp': 35.0, 'hum': 45.0}
            st.rerun()
        
        if st.button("🌧️ Wet Conditions", use_container_width=True):
            st.session_state.example = {'soil': 850.0, 'temp': 22.0, 'hum': 75.0}
            st.rerun()
        
        if st.button("☀️ Hot & Moderate", use_container_width=True):
            st.session_state.example = {'soil': 600.0, 'temp': 36.0, 'hum': 55.0}
            st.rerun()
    
    # Main content area
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>💧 Soil Moisture</h3>
            <h1>{soil_moisture:.1f}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>🌡️ Temperature</h3>
            <h1>{temperature:.1f}°C</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>💨 Humidity</h3>
            <h1>{humidity:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Show initial message before prediction
    if not predict_button and 'predictions' not in st.session_state:
        st.markdown("""
        <div style='text-align: center; padding: 50px; background-color: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h2>👈 Adjust the parameters in the sidebar</h2>
            <p style='font-size: 18px; color: #7f8c8d;'>
                Set the soil moisture, temperature, and air humidity values,<br>
                then click <b>"🚀 PREDICT PUMP STATUS"</b> to get recommendations.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Make predictions
    if predict_button:
        with st.spinner("🔮 Analyzing conditions with AI models..."):
            predictions = make_predictions(models, soil_moisture, temperature, humidity)
            st.session_state.predictions = predictions
            st.session_state.inputs = {
                'soil_moisture': soil_moisture,
                'temperature': temperature,
                'humidity': humidity
            }
    
    if 'predictions' in st.session_state:
        predictions = st.session_state.predictions
        
        # Find best model
        best_model = max(predictions.keys(), key=lambda k: predictions[k]['confidence'])
        best_confidence = predictions[best_model]['confidence']
        best_prediction = predictions[best_model]['prediction']
        
        # Overall recommendation
        st.markdown("## 🎯 Irrigation Recommendation")
        
        col_gauge, col_info = st.columns([1, 1])
        
        with col_gauge:
            fig_gauge = plot_prediction_gauge(best_prediction, best_confidence)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col_info:
            if best_prediction == 1:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            color: white; padding: 30px; border-radius: 15px; text-align: center;'>
                    <h1>💧 IRRIGATION NEEDED</h1>
                    <p style='font-size: 20px; margin: 20px 0;'>
                        Turn the pump <b>ON</b>
                    </p>
                    <p style='font-size: 16px; opacity: 0.9;'>
                        Best Model: <b>{best_model}</b><br>
                        Confidence: <b>{best_confidence*100:.1f}%</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                            color: white; padding: 30px; border-radius: 15px; text-align: center;'>
                    <h1>🚫 NO IRRIGATION</h1>
                    <p style='font-size: 20px; margin: 20px 0;'>
                        Keep the pump <b>OFF</b>
                    </p>
                    <p style='font-size: 16px; opacity: 0.9;'>
                        Best Model: <b>{best_model}</b><br>
                        Confidence: <b>{best_confidence*100:.1f}%</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Comparison Chart", "🔍 Detailed Results", "📈 Model Analysis"])
        
        with tab1:
            st.plotly_chart(plot_confidence_comparison(predictions), use_container_width=True)
            
            # Legend
            col_legend1, col_legend2 = st.columns(2)
            with col_legend1:
                st.markdown("🔴 **Quantum Models** - VQC (Variational Quantum Classifier)")
            with col_legend2:
                st.markdown("🔵 **Classical Models** - LR, SVM, Random Forest")
        
        with tab2:
            st.markdown("### 🤖 Individual Model Predictions")
            
            for model_name, pred_info in predictions.items():
                is_best = model_name == best_model
                
                if is_best:
                    st.markdown(f"""
                    <div class='highlight-best'>
                        🏆 BEST MODEL: {model_name}
                    </div>
                    """, unsafe_allow_html=True)
                
                with st.container():
                    col_model, col_pred, col_conf = st.columns([2, 2, 2])
                    
                    with col_model:
                        model_type_icon = "⚛️" if pred_info['type'] == 'quantum' else "🖥️"
                        st.markdown(f"### {model_type_icon} {model_name}")
                        st.caption(f"Type: {pred_info['type'].title()}")
                    
                    with col_pred:
                        if pred_info['prediction'] == 1:
                            st.markdown("### 💧 **ON**")
                            st.success("Irrigation Required")
                        else:
                            st.markdown("### 🚫 **OFF**")
                            st.error("No Irrigation")
                    
                    with col_conf:
                        st.markdown("### 📊 Confidence")
                        st.metric(
                            label="",
                            value=f"{pred_info['confidence']*100:.1f}%",
                            delta=f"{(pred_info['confidence'] - 0.5)*100:+.1f}% vs random"
                        )
                
                st.markdown("---")
        
        with tab3:
            st.markdown("### 📈 Statistical Analysis")
            
            # Create DataFrame for analysis
            df_results = pd.DataFrame([
                {
                    'Model': name,
                    'Type': info['type'].title(),
                    'Prediction': 'ON' if info['prediction'] == 1 else 'OFF',
                    'Confidence (%)': f"{info['confidence']*100:.2f}",
                    'Probability (%)': f"{info['probability']*100:.2f}"
                }
                for name, info in predictions.items()
            ])
            
            st.dataframe(
                df_results,
                use_container_width=True,
                hide_index=True
            )
            
            # Statistics
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                avg_confidence = np.mean([p['confidence'] for p in predictions.values()])
                st.metric("Average Confidence", f"{avg_confidence*100:.1f}%")
            
            with col_stat2:
                consensus = len([p for p in predictions.values() if p['prediction'] == best_prediction])
                st.metric("Model Consensus", f"{consensus}/{len(predictions)}")
            
            with col_stat3:
                quantum_models = len([p for p in predictions.values() if p['type'] == 'quantum'])
                st.metric("Quantum Models", f"{quantum_models}")
        
        # Environmental insights
        st.markdown("---")
        st.markdown("## 🌍 Environmental Analysis")
        
        col_insight1, col_insight2, col_insight3 = st.columns(3)
        
        with col_insight1:
            st.markdown("""
            <div class='info-box'>
            <b>💧 Soil Moisture Analysis</b><br>
            """, unsafe_allow_html=True)
            
            if soil_moisture < 500:
                st.markdown("🔴 **Critical:** Very low soil moisture - High irrigation priority")
            elif soil_moisture < 650:
                st.markdown("🟡 **Moderate:** Below optimal moisture - Consider irrigation")
            else:
                st.markdown("🟢 **Good:** Adequate soil moisture - Low irrigation priority")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_insight2:
            st.markdown("""
            <div class='info-box'>
            <b>🌡️ Temperature Analysis</b><br>
            """, unsafe_allow_html=True)
            
            if temperature > 33:
                st.markdown("🔴 **Hot:** High evaporation rate - Increased water needs")
            elif temperature > 27:
                st.markdown("🟡 **Warm:** Moderate water requirements")
            else:
                st.markdown("🟢 **Cool:** Lower evaporation - Reduced water needs")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_insight3:
            st.markdown("""
            <div class='info-box'>
            <b>💨 Humidity Analysis</b><br>
            """, unsafe_allow_html=True)
            
            if humidity < 50:
                st.markdown("🔴 **Dry:** Low humidity - Higher irrigation needs")
            elif humidity < 65:
                st.markdown("🟡 **Moderate:** Normal humidity levels")
            else:
                st.markdown("🟢 **Humid:** High moisture in air - Lower irrigation needs")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #95a5a6; padding: 20px;'>
        <p><b>Quantum-Enhanced Smart Irrigation System</b></p>
        <p>Powered by PennyLane, Scikit-learn & Streamlit</p>
        <p>🌾 Optimizing water usage for sustainable agriculture 🌍</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()