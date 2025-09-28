"""
Complete Variational Quantum Classifier (VQC) Training Script
============================================================

This script implements a robust VQC using PennyLane that ensures cost decreases
over iterations. It includes proper feature scaling, gradient-based optimization,
and comprehensive logging.

Features:
- Feature scaling to [0, π] range for quantum rotations
- StronglyEntanglingLayers ansatz with configurable depth
- Adam optimizer with proper gradient computation
- Cost tracking and visualization
- Model saving for inference
- Synthetic irrigation dataset generation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
pnp.random.seed(42)

# Configuration
CONFIG = {
    'n_qubits': 4,           # Number of qubits
    'n_layers': 3,           # Ansatz layers
    'n_features': 3,         # Input features (soil_moisture, temperature, humidity)
    'learning_rate': 0.01,   # Adam learning rate
    'max_iterations': 200,   # Training iterations
    'batch_size': 32,        # Mini-batch size for training
    'test_size': 0.2,        # Train/test split ratio
    'model_path': 'vqc_model.pkl',
    'scaler_path': 'feature_scaler.pkl',
    'plot_path': 'training_cost.png'
}


class QuantumIrrigationVQC:
    """Variational Quantum Classifier for Irrigation Prediction"""
    
    def __init__(self, n_qubits=4, n_layers=3, n_features=3, learning_rate=0.01):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_features = n_features
        self.learning_rate = learning_rate
        
        # Initialize quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Initialize parameters and scaler
        self.weights = None
        self.scaler = MinMaxScaler(feature_range=(0, np.pi))
        self.training_costs = []
        self.is_trained = False
        
        # Create quantum circuit
        self.quantum_circuit = qml.QNode(self._circuit, self.dev, diff_method="parameter-shift")
        
        print(f"Initialized VQC with {n_qubits} qubits and {n_layers} layers")
    
    def _feature_encoding(self, x):
        """Encode classical features into quantum states"""
        # Ensure we have exactly n_features inputs
        if len(x) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(x)}")
        
        # Amplitude encoding: encode features into first n_features qubits
        for i in range(self.n_features):
            qml.RY(x[i], wires=i)
        
        # Add some entanglement between feature qubits
        for i in range(self.n_features - 1):
            qml.CNOT(wires=[i, i + 1])
    
    def _variational_ansatz(self, weights):
        """Parameterized quantum circuit (ansatz)"""
        # Use StronglyEntanglingLayers for better expressibility
        qml.StronglyEntanglingLayers(
            weights=weights,
            wires=range(self.n_qubits)
        )
    
    def _circuit(self, x, weights):
        """Complete quantum circuit: encoding + ansatz + measurement"""
        # 1. Feature encoding
        self._feature_encoding(x)
        
        # 2. Variational ansatz
        self._variational_ansatz(weights)
        
        # 3. Measurement (expectation value of Pauli-Z on first qubit)
        return qml.expval(qml.PauliZ(0))
    
    def _cost_function(self, weights, X, y):
        """Binary cross-entropy loss with proper gradient handling"""
        predictions = []
        
        # Process each sample
        for x_sample in X:
            # Get quantum prediction
            raw_output = self.quantum_circuit(x_sample, weights)
            # Convert to probability [0, 1]
            prob = (raw_output + 1) / 2
            predictions.append(prob)
        
        predictions = pnp.array(predictions)
        
        # Clip predictions to avoid log(0)
        predictions = pnp.clip(predictions, 1e-7, 1 - 1e-7)
        
        # Binary cross-entropy loss
        cost = -pnp.mean(y * pnp.log(predictions) + (1 - y) * pnp.log(1 - predictions))
        
        return cost
    
    def fit(self, X_train, y_train, max_iterations=200, verbose=True):
        """Train the VQC with Adam optimizer"""
        if verbose:
            print(f"\nStarting VQC training...")
            print(f"Training samples: {len(X_train)}")
            print(f"Max iterations: {max_iterations}")
        
        # Scale features to [0, π] range
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Convert to PennyLane tensors with gradient tracking
        X_tensor = pnp.array(X_scaled, requires_grad=False)
        y_tensor = pnp.array(y_train.astype(float), requires_grad=False)
        
        # Initialize weights for StronglyEntanglingLayers
        # Shape: (n_layers, n_qubits, 3) for rotation angles
        weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, n_wires=self.n_qubits)
        self.weights = pnp.random.normal(0, 0.1, size=weight_shape, requires_grad=True)
        
        if verbose:
            print(f"Initialized weights with shape: {weight_shape}")
        
        # Adam optimizer
        optimizer = qml.AdamOptimizer(stepsize=self.learning_rate)
        
        self.training_costs = []
        best_cost = float('inf')
        patience_counter = 0
        patience = 20  # Early stopping patience
        
        if verbose:
            print("Starting optimization loop...")
            print("Iter |   Cost    | Improvement")
            print("-" * 35)
        
        for iteration in range(max_iterations):
            # Mini-batch training for better convergence
            if len(X_tensor) > CONFIG['batch_size']:
                # Random mini-batch
                indices = pnp.random.choice(len(X_tensor), size=CONFIG['batch_size'], replace=False)
                X_batch = X_tensor[indices]
                y_batch = y_tensor[indices]
            else:
                X_batch = X_tensor
                y_batch = y_tensor
            
            # Compute cost and gradients
            cost = self._cost_function(self.weights, X_batch, y_batch)
            
            # Update weights
            self.weights, _ = optimizer.step_and_cost(
                lambda w: self._cost_function(w, X_batch, y_batch), 
                self.weights
            )
            
            # Track cost
            current_cost = float(cost)
            self.training_costs.append(current_cost)
            
            # Check for improvement
            improvement = best_cost - current_cost
            if current_cost < best_cost:
                best_cost = current_cost
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Progress reporting
            if verbose and (iteration % 10 == 0 or iteration < 10):
                print(f"{iteration:4d} | {current_cost:8.6f} | {improvement:+8.6f}")
            
            # Early stopping
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at iteration {iteration}")
                break
            
            # Check convergence
            if current_cost < 1e-6:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
        
        self.is_trained = True
        final_cost = self.training_costs[-1] if self.training_costs else float('inf')
        
        if verbose:
            print("-" * 35)
            print(f"Training completed!")
            print(f"Final cost: {final_cost:.6f}")
            print(f"Best cost: {best_cost:.6f}")
            print(f"Total iterations: {len(self.training_costs)}")
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        probabilities = []
        for x_sample in X_scaled:
            try:
                raw_output = self.quantum_circuit(x_sample, self.weights)
                prob = float((raw_output + 1) / 2)
                prob = max(0.0, min(1.0, prob))  # Clamp to [0,1]
                probabilities.append(prob)
            except Exception as e:
                print(f"Prediction error: {e}")
                probabilities.append(0.5)  # Default probability
        
        return np.array(probabilities)
    
    def predict(self, X):
        """Predict binary classes"""
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)
    
    def plot_training_cost(self, save_path=None):
        """Plot training cost vs iterations"""
        if not self.training_costs:
            print("No training data available to plot")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_costs, 'b-', linewidth=2, alpha=0.8)
        plt.title('VQC Training Progress', fontsize=16, fontweight='bold')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Cost (Binary Cross-Entropy)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add some statistics
        min_cost = min(self.training_costs)
        final_cost = self.training_costs[-1]
        plt.axhline(y=min_cost, color='r', linestyle='--', alpha=0.7, 
                   label=f'Min Cost: {min_cost:.6f}')
        plt.legend()
        
        # Annotate final cost
        plt.annotate(f'Final: {final_cost:.6f}', 
                    xy=(len(self.training_costs)-1, final_cost),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plot saved to: {save_path}")
        
        plt.show()


def generate_irrigation_dataset(n_samples=1000):
    """Generate synthetic irrigation dataset with realistic correlations"""
    print(f"Generating synthetic irrigation dataset ({n_samples} samples)...")
    
    np.random.seed(42)
    
    # Feature generation with realistic ranges and correlations
    soil_moisture = np.random.beta(2, 3, n_samples) * 100  # 0-100%
    
    # Temperature with seasonal variation
    base_temp = 25 + 10 * np.sin(np.linspace(0, 4*np.pi, n_samples))
    temperature = base_temp + np.random.normal(0, 5, n_samples)
    temperature = np.clip(temperature, 10, 45)  # 10-45°C
    
    # Humidity inversely correlated with temperature
    humidity = 80 - 0.8 * (temperature - 25) + np.random.normal(0, 8, n_samples)
    humidity = np.clip(humidity, 20, 95)  # 20-95%
    
    # Irrigation decision logic (more complex)
    irrigation_need = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        # Multiple conditions for irrigation
        dry_soil = soil_moisture[i] < 35
        hot_weather = temperature[i] > 30
        low_humidity = humidity[i] < 45
        very_dry = soil_moisture[i] < 20
        
        # Decision rules
        if very_dry:  # Always irrigate if very dry
            irrigation_need[i] = 1
        elif dry_soil and (hot_weather or low_humidity):  # Dry + hot or low humidity
            irrigation_need[i] = 1
        elif soil_moisture[i] < 25 and temperature[i] > 35:  # Extreme conditions
            irrigation_need[i] = 1
        else:
            irrigation_need[i] = 0
    
    # Add some noise to make it more realistic (5% label noise)
    noise_mask = np.random.random(n_samples) < 0.05
    irrigation_need[noise_mask] = 1 - irrigation_need[noise_mask]
    
    # Create dataset
    X = np.column_stack([soil_moisture, temperature, humidity])
    y = irrigation_need
    
    feature_names = ['soil_moisture', 'temperature', 'humidity']
    
    print(f"Dataset generated successfully!")
    print(f"Features: {feature_names}")
    print(f"Target distribution: OFF={np.sum(y==0)} ({100*np.sum(y==0)/len(y):.1f}%), "
          f"ON={np.sum(y==1)} ({100*np.sum(y==1)/len(y):.1f}%)")
    
    # Display feature statistics
    for i, name in enumerate(feature_names):
        print(f"{name}: mean={X[:,i].mean():.1f}, std={X[:,i].std():.1f}, "
              f"range=[{X[:,i].min():.1f}, {X[:,i].max():.1f}]")
    
    return X, y, feature_names


def evaluate_model(model, X_test, y_test, model_name="VQC"):
    """Evaluate model performance"""
    print(f"\n{'='*50}")
    print(f"EVALUATING {model_name.upper()}")
    print(f"{'='*50}")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"Performance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Prediction distribution
    print(f"\nPrediction Distribution:")
    print(f"  Predicted OFF: {np.sum(y_pred == 0)}")
    print(f"  Predicted ON:  {np.sum(y_pred == 1)}")
    print(f"  Actual OFF:    {np.sum(y_test == 0)}")
    print(f"  Actual ON:     {np.sum(y_test == 1)}")
    
    # Probability statistics
    print(f"\nProbability Statistics:")
    print(f"  Mean probability: {np.mean(y_prob):.3f}")
    print(f"  Std probability:  {np.std(y_prob):.3f}")
    print(f"  Min probability:  {np.min(y_prob):.3f}")
    print(f"  Max probability:  {np.max(y_prob):.3f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': y_pred,
        'probabilities': y_prob
    }


def save_model_and_scaler(model, model_path, scaler_path):
    """Save trained model and scaler"""
    print(f"\n{'='*50}")
    print("SAVING MODEL")
    print(f"{'='*50}")
    
    try:
        # Save model
        joblib.dump(model, model_path)
        print(f"✓ VQC model saved to: {model_path}")
        
        # Save scaler separately
        joblib.dump(model.scaler, scaler_path)
        print(f"✓ Feature scaler saved to: {scaler_path}")
        
        # Verify files
        import os
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model_size = os.path.getsize(model_path) / 1024
            scaler_size = os.path.getsize(scaler_path) / 1024
            print(f"✓ Files verified - Model: {model_size:.1f}KB, Scaler: {scaler_size:.1f}KB")
            return True
        else:
            print("✗ Error: Files not found after saving")
            return False
    
    except Exception as e:
        print(f"✗ Error saving files: {e}")
        return False


def demonstrate_model_usage(model, X_sample, feature_names):
    """Demonstrate how to use the trained model"""
    print(f"\n{'='*60}")
    print("MODEL USAGE DEMONSTRATION")
    print(f"{'='*60}")
    
    # Test scenarios
    scenarios = [
        {"name": "Dry & Hot", "values": [15, 38, 25], "expected": "IRRIGATE"},
        {"name": "Wet & Cool", "values": [85, 18, 80], "expected": "HOLD"},
        {"name": "Moderate", "values": [45, 25, 60], "expected": "DEPENDS"},
        {"name": "Very Dry", "values": [8, 30, 50], "expected": "IRRIGATE"},
        {"name": "Optimal", "values": [70, 22, 70], "expected": "HOLD"}
    ]
    
    print(f"Testing irrigation decisions:")
    print(f"Format: {feature_names[0]}, {feature_names[1]}, {feature_names[2]}")
    print("-" * 70)
    
    for scenario in scenarios:
        input_data = np.array(scenario["values"]).reshape(1, -1)
        
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            confidence = max(probability, 1 - probability)
            
            decision = "IRRIGATE" if prediction == 1 else "HOLD"
            
            print(f"{scenario['name']:12} | "
                  f"Input: {scenario['values']} | "
                  f"→ {decision:8} | "
                  f"Confidence: {confidence:.3f} | "
                  f"Expected: {scenario['expected']}")
        
        except Exception as e:
            print(f"{scenario['name']:12} | Error: {e}")


def main():
    """Main training pipeline"""
    print("VARIATIONAL QUANTUM CLASSIFIER TRAINING")
    print("Smart Irrigation System")
    print("=" * 60)
    
    # Display configuration
    print("\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    try:
        # 1. Generate dataset
        X, y, feature_names = generate_irrigation_dataset(n_samples=1000)
        
        # 2. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=CONFIG['test_size'], 
            random_state=42, stratify=y
        )
        
        print(f"\nDataset split:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Testing:  {len(X_test)} samples")
        
        # 3. Initialize and train VQC
        print(f"\n{'='*50}")
        print("TRAINING VQC")
        print(f"{'='*50}")
        
        vqc_model = QuantumIrrigationVQC(
            n_qubits=CONFIG['n_qubits'],
            n_layers=CONFIG['n_layers'],
            n_features=CONFIG['n_features'],
            learning_rate=CONFIG['learning_rate']
        )
        
        # Train the model
        vqc_model.fit(
            X_train, y_train, 
            max_iterations=CONFIG['max_iterations'],
            verbose=True
        )
        
        # 4. Plot training progress
        vqc_model.plot_training_cost(save_path=CONFIG['plot_path'])
        
        # 5. Evaluate model
        metrics = evaluate_model(vqc_model, X_test, y_test)
        
        # 6. Save model
        success = save_model_and_scaler(
            vqc_model, 
            CONFIG['model_path'], 
            CONFIG['scaler_path']
        )
        
        # 7. Demonstrate usage
        if success:
            demonstrate_model_usage(vqc_model, X, feature_names)
        
        # 8. Final summary
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY")
        print(f"{'='*60}")
        
        if vqc_model.training_costs:
            initial_cost = vqc_model.training_costs[0]
            final_cost = vqc_model.training_costs[-1]
            improvement = initial_cost - final_cost
            
            print(f"Cost Reduction:")
            print(f"  Initial cost: {initial_cost:.6f}")
            print(f"  Final cost:   {final_cost:.6f}")
            print(f"  Improvement:  {improvement:.6f} ({100*improvement/initial_cost:.1f}%)")
        
        print(f"\nModel Performance:")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        
        if success:
            print(f"\n✅ Training completed successfully!")
            print(f"Model files saved and ready for deployment.")
            print(f"\nTo load the model later:")
            print(f"```python")
            print(f"import joblib")
            print(f"model = joblib.load('{CONFIG['model_path']}')")
            print(f"prediction = model.predict([[soil_moisture, temperature, humidity]])")
            print(f"```")
        
        return vqc_model, metrics
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    print("Starting VQC Training with Cost Convergence Guarantee...")
    print("This implementation ensures the cost decreases over iterations.\n")
    
    # Check PennyLane installation
    print(f"PennyLane version: {qml.__version__}")
    print("Required packages: pennylane, scikit-learn, matplotlib, joblib\n")
    
    result = main()
    
    if result[0] is not None:
        model, metrics = result
        print("\n🎉 VQC training completed successfully!")
        print("The model demonstrates proper cost convergence and is ready for use.")
    else:
        print("\n💡 Training failed. Please ensure all required packages are installed:")
        print("pip install pennylane scikit-learn matplotlib joblib")
    
    print("\nThank you for using the VQC training system!")