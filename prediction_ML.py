"""
Compatible Irrigation Prediction Script
======================================

This script can load VQC models trained with the fixed training script
and provides a simple interface for irrigation predictions.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
import pennylane as qml
from pennylane import numpy as pnp

# Import the VQC class definition (needed for pickle loading)
class FixedQuantumIrrigationVQC:
    """Fixed VQC class definition - needed for loading saved models"""
    
    def __init__(self, n_qubits=4, n_layers=3, n_features=3, learning_rate=0.01):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.weights = None
        self.scaler = MinMaxScaler(feature_range=(0, np.pi))
        self.training_costs = []
        self.is_trained = False
        self.training_time = 0
        self.quantum_circuit = qml.QNode(self._circuit, self.dev, diff_method="parameter-shift")
    
    def _feature_encoding(self, x):
        for i in range(min(self.n_features, self.n_qubits)):
            qml.RY(x[i], wires=i)
    
    def _variational_ansatz(self, weights):
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                qml.RY(weights[layer, qubit], wires=qubit)
            for qubit in range(self.n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])
            if self.n_qubits > 1:
                qml.CNOT(wires=[self.n_qubits - 1, 0])
    
    def _circuit(self, x, weights):
        self._feature_encoding(x)
        self._variational_ansatz(weights)
        return qml.expval(qml.PauliZ(0))
    
    def _prediction_from_expectation(self, expectation_val):
        probability = (expectation_val + 1) / 2
        return np.clip(probability, 0.001, 0.999)
    
    def predict_proba(self, X):
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
                print(f"Prediction error: {e}")
                probabilities.append(0.5)
        return np.array(probabilities)
    
    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)


class IrrigationPredictor:
    """Simple irrigation prediction interface"""
    
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.load_models()
        
        print("Smart Irrigation Prediction System Initialized")
        print(f"Available models: {list(self.models.keys())}")
    
    def load_models(self):
        """Load available models"""
        if not os.path.exists(self.model_dir):
            print(f"Model directory '{self.model_dir}' not found!")
            return
        
        # Try to load VQC
        vqc_path = os.path.join(self.model_dir, 'vqc_model.pkl')
        if os.path.exists(vqc_path):
            try:
                self.models['vqc'] = joblib.load(vqc_path)
                print("✓ VQC model loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load VQC: {e}")
        
        # Load classical models
        classical_models = ['logistic_regression', 'svm', 'random_forest']
        
        for model_name in classical_models:
            model_path = os.path.join(self.model_dir, f'{model_name}_model.pkl')
            scaler_path = os.path.join(self.model_dir, f'{model_name}_scaler.pkl')
            
            if os.path.exists(model_path):
                try:
                    self.models[model_name] = joblib.load(model_path)
                    print(f"✓ {model_name.replace('_', ' ').title()} model loaded")
                    
                    if os.path.exists(scaler_path):
                        self.scalers[model_name] = joblib.load(scaler_path)
                        print(f"✓ {model_name.replace('_', ' ').title()} scaler loaded")
                        
                except Exception as e:
                    print(f"❌ Failed to load {model_name}: {e}")
    
    def predict_single(self, soil_moisture, temperature, humidity):
        """Make predictions for single input"""
        X = np.array([[soil_moisture, temperature, humidity]])
        results = {}
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'vqc':
                    prediction = model.predict(X)[0]
                    probability = model.predict_proba(X)[0]
                else:
                    # Classical models
                    if model_name in self.scalers:
                        X_scaled = self.scalers[model_name].transform(X)
                        prediction = model.predict(X_scaled)[0]
                        probability = model.predict_proba(X_scaled)[0, 1]
                    else:
                        prediction = model.predict(X)[0]
                        probability = model.predict_proba(X)[0, 1]
                
                confidence = max(probability, 1 - probability)
                decision = 'IRRIGATE' if prediction == 1 else 'HOLD'
                
                results[model_name] = {
                    'prediction': int(prediction),
                    'probability': float(probability),
                    'confidence': float(confidence),
                    'decision': decision
                }
                
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                results[model_name] = {
                    'prediction': None,
                    'probability': None,
                    'confidence': None,
                    'decision': 'ERROR'
                }
        
        return results
    
    def get_recommendation(self, soil_moisture, temperature, humidity, results):
        """Get irrigation recommendation"""
        decisions = [r['decision'] for r in results.values() if r['decision'] != 'ERROR']
        
        if not decisions:
            return "⚠ No valid predictions available"
        
        irrigate_votes = sum(1 for d in decisions if d == 'IRRIGATE')
        consensus = irrigate_votes / len(decisions)
        
        if consensus >= 0.6:
            recommendation = "💧 RECOMMENDED: Start irrigation"
            if soil_moisture < 20:
                recommendation += "\n   🚨 URGENT: Very low soil moisture!"
            elif temperature > 35:
                recommendation += "\n   ⏰ Best time: Early morning or evening"
        else:
            recommendation = "✋ RECOMMENDED: Hold irrigation"
            if soil_moisture > 60:
                recommendation += "\n   💡 Reason: Adequate soil moisture"
        
        return recommendation


def interactive_mode():
    """Interactive prediction interface"""
    predictor = IrrigationPredictor()
    
    if not predictor.models:
        print("❌ No models available! Please run training first.")
        return
    
    print("\n" + "="*60)
    print("SMART IRRIGATION INTERACTIVE PREDICTION")
    print("="*60)
    print("Enter sensor values to get irrigation recommendations")
    print("Commands:")
    print("  - Enter 3 values separated by commas: soil_moisture, temperature, humidity")
    print("  - 'quit' or 'exit' to stop")
    print("  - 'demo' for sample predictions")
    print("  - 'help' for more information")
    print("-"*60)
    
    while True:
        try:
            user_input = input("\n📊 Enter values (or command): ").strip().lower()
            
            if user_input in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            elif user_input == 'help':
                print("\nHelp:")
                print("Format: soil_moisture, temperature, humidity")
                print("Example: 25.5, 35.2, 40.0")
                print("Ranges: Soil (0-100%), Temp (5-45°C), Humidity (15-95%)")
                continue
            
            elif user_input == 'demo':
                demo_scenarios = [
                    (15, 38, 25, "Dry & Hot"),
                    (80, 18, 85, "Wet & Cool"), 
                    (45, 28, 55, "Moderate"),
                    (8, 35, 20, "Critical Drought")
                ]
                
                for soil, temp, hum, desc in demo_scenarios:
                    print(f"\n📍 {desc}: {soil}% soil, {temp}°C, {hum}% humidity")
                    results = predictor.predict_single(soil, temp, hum)
                    
                    for model_name, result in results.items():
                        if result['decision'] != 'ERROR':
                            icon = "💧" if result['decision'] == 'IRRIGATE' else "✋"
                            print(f"   {icon} {model_name.replace('_', ' ').title()}: {result['decision']} (conf: {result['confidence']:.3f})")
                continue
            
            # Parse input values
            try:
                if ',' in user_input:
                    values = [float(v.strip()) for v in user_input.split(',')]
                    if len(values) != 3:
                        print("❌ Please enter exactly 3 values: soil_moisture, temperature, humidity")
                        continue
                    soil_moisture, temperature, humidity = values
                else:
                    print("❌ Please enter values separated by commas (e.g., 25.5, 35.2, 40.0)")
                    continue
            except ValueError:
                print("❌ Please enter valid numbers")
                continue
            
            # Make predictions
            print(f"\n🔍 Analyzing: {soil_moisture}% soil, {temperature}°C, {humidity}% humidity")
            
            results = predictor.predict_single(soil_moisture, temperature, humidity)
            
            print(f"\n📋 PREDICTIONS:")
            print("-" * 50)
            
            for model_name, result in results.items():
                if result['decision'] != 'ERROR':
                    icon = "💧" if result['decision'] == 'IRRIGATE' else "✋"
                    conf_bar = "█" * int(result['confidence'] * 10)
                    print(f"{icon} {model_name.replace('_', ' ').title():<20}: {result['decision']:<8} (conf: {result['confidence']:.3f} {conf_bar})")
                else:
                    print(f"❌ {model_name.replace('_', ' ').title():<20}: ERROR")
            
            # Get recommendation
            recommendation = predictor.get_recommendation(soil_moisture, temperature, humidity, results)
            print(f"\n🎯 RECOMMENDATION:")
            print(f"   {recommendation}")
            
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("🌱 SMART IRRIGATION PREDICTION SYSTEM")
    print("="*50)
    print("Quantum-Enhanced AI for Precision Agriculture")
    print("="*50)
    
    interactive_mode()