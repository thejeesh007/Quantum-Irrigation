"""
Fixed Variational Quantum Classifier (VQC) for Smart Irrigation System
=====================================================================

This script implements a stable VQC that avoids measurement operation errors
and focuses on clear cost tracking and iteration improvements.

Key fixes:
- Proper circuit measurement handling
- Simplified cost function
- Clear iteration progress display
- Error-free gradient computation
- Stable probability conversion

Author: Fixed VQC System
Version: 3.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report, roc_auc_score)
import pennylane as qml
from pennylane import numpy as pnp
import joblib
import warnings
import time
import os
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
pnp.random.seed(42)

# Simplified Configuration
CONFIG = {
    # Dataset parameters
    'n_samples': 1000,
    'test_size': 0.2,
    'validation_size': 0.2,
    
    # VQC parameters
    'n_qubits': 4,              # Reduced for stability
    'n_layers': 3,              # Reduced for stability
    'n_features': 3,
    
    # Training parameters
    'learning_rate': 0.01,
    'max_iterations': 100,      # Reduced for faster testing
    'batch_size': 32,
    'early_stopping_patience': 15,
    
    # File paths
    'model_dir': 'models',
    'plots_dir': 'plots',
    'results_dir': 'results',
}


class FixedQuantumIrrigationVQC:
    """Fixed Variational Quantum Classifier with stable implementation"""
    
    def __init__(self, n_qubits=4, n_layers=3, n_features=3, learning_rate=0.01):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_features = n_features
        self.learning_rate = learning_rate
        
        # Initialize quantum device - no shots for exact computation
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Initialize components
        self.weights = None
        self.scaler = MinMaxScaler(feature_range=(0, np.pi))
        self.training_costs = []
        self.is_trained = False
        self.training_time = 0
        
        # Create quantum circuit with proper differentiation
        self.quantum_circuit = qml.QNode(
            self._circuit, 
            self.dev, 
            diff_method="parameter-shift"
        )
        
        print(f"Fixed VQC initialized: {n_qubits} qubits, {n_layers} layers")
    
    def _feature_encoding(self, x):
        """Simple angle encoding for features"""
        for i in range(min(self.n_features, self.n_qubits)):
            qml.RY(x[i], wires=i)
    
    def _variational_ansatz(self, weights):
        """Simple variational ansatz with RY rotations and CNOT gates"""
        for layer in range(self.n_layers):
            # Rotation layer
            for qubit in range(self.n_qubits):
                qml.RY(weights[layer, qubit], wires=qubit)
            
            # Entanglement layer
            for qubit in range(self.n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])
            
            # Circular entanglement
            if self.n_qubits > 1:
                qml.CNOT(wires=[self.n_qubits - 1, 0])
    
    def _circuit(self, x, weights):
        """Complete quantum circuit returning single expectation value"""
        # Feature encoding
        self._feature_encoding(x)
        
        # Variational ansatz
        self._variational_ansatz(weights)
        
        # Single measurement on first qubit
        return qml.expval(qml.PauliZ(0))
    
    def _prediction_from_expectation(self, expectation_val):
        """Convert expectation value to probability"""
        # Map [-1, 1] to [0, 1]
        probability = (expectation_val + 1) / 2
        return np.clip(probability, 0.001, 0.999)  # Avoid extreme values
    
    def _cost_function(self, weights, X, y):
        """Binary cross-entropy cost function"""
        predictions = []
        
        for x_sample in X:
            # Get expectation value
            expectation = self.quantum_circuit(x_sample, weights)
            # Convert to probability
            prob = self._prediction_from_expectation(expectation)
            predictions.append(prob)
        
        predictions = pnp.array(predictions)
        
        # Binary cross-entropy loss
        epsilon = 1e-7  # Small constant to avoid log(0)
        predictions = pnp.clip(predictions, epsilon, 1 - epsilon)
        
        cost = -pnp.mean(y * pnp.log(predictions) + (1 - y) * pnp.log(1 - predictions))
        
        return cost
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """Train the VQC with clear progress tracking"""
        start_time = time.time()
        
        if verbose:
            print(f"\nStarting VQC Training")
            print("=" * 50)
            print(f"Training samples: {len(X_train)}")
            print(f"Validation samples: {len(X_val) if X_val is not None else 'None'}")
            print(f"Max iterations: {CONFIG['max_iterations']}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        X_tensor = pnp.array(X_scaled, requires_grad=False)
        y_tensor = pnp.array(y_train.astype(float), requires_grad=False)
        
        # Validation data
        X_val_tensor = None
        y_val_tensor = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_tensor = pnp.array(X_val_scaled, requires_grad=False)
            y_val_tensor = pnp.array(y_val.astype(float), requires_grad=False)
        
        # Initialize weights
        weight_shape = (self.n_layers, self.n_qubits)
        self.weights = pnp.random.normal(0, 0.1, size=weight_shape, requires_grad=True)
        
        if verbose:
            print(f"Weight shape: {weight_shape}")
            print(f"Total parameters: {np.prod(weight_shape)}")
        
        # Adam optimizer
        optimizer = qml.AdamOptimizer(stepsize=self.learning_rate)
        
        # Training tracking
        self.training_costs = []
        best_val_cost = float('inf')
        patience_counter = 0
        previous_cost = float('inf')
        
        if verbose:
            print("\nTraining Progress:")
            print("Iter |   Cost     | Improvement | Val Cost  | Patience")
            print("-" * 60)
        
        for iteration in range(CONFIG['max_iterations']):
            # Mini-batch selection
            if len(X_tensor) > CONFIG['batch_size']:
                indices = pnp.random.choice(len(X_tensor), size=CONFIG['batch_size'], replace=False)
                X_batch = X_tensor[indices]
                y_batch = y_tensor[indices]
            else:
                X_batch = X_tensor
                y_batch = y_tensor
            
            try:
                # Training step
                self.weights, current_cost = optimizer.step_and_cost(
                    lambda w: self._cost_function(w, X_batch, y_batch), 
                    self.weights
                )
                
                current_cost = float(current_cost)
                self.training_costs.append(current_cost)
                
                # Calculate improvement
                improvement = previous_cost - current_cost
                previous_cost = current_cost
                
                # Validation cost
                val_cost_str = "   N/A   "
                if X_val_tensor is not None:
                    val_cost = float(self._cost_function(self.weights, X_val_tensor, y_val_tensor))
                    val_cost_str = f"{val_cost:8.5f}"
                    
                    # Early stopping logic
                    if val_cost < best_val_cost:
                        best_val_cost = val_cost
                        patience_counter = 0
                    else:
                        patience_counter += 1
                
                # Progress display
                if verbose and (iteration % 10 == 0 or iteration < 10):
                    print(f"{iteration:4d} | {current_cost:9.6f} | {improvement:+9.6f} | {val_cost_str} | {patience_counter:4d}")
                
                # Early stopping
                if patience_counter >= CONFIG['early_stopping_patience']:
                    if verbose:
                        print(f"\nEarly stopping at iteration {iteration}")
                    break
                
                # Convergence check
                if current_cost < 1e-6:
                    if verbose:
                        print(f"\nConverged at iteration {iteration}")
                    break
                
                # NaN check
                if np.isnan(current_cost):
                    print(f"\nWarning: NaN cost detected at iteration {iteration}")
                    break
                    
            except Exception as e:
                print(f"Training error at iteration {iteration}: {e}")
                break
        
        self.training_time = time.time() - start_time
        self.is_trained = True
        
        if verbose:
            print("-" * 60)
            print(f"Training completed in {self.training_time:.2f} seconds")
            if self.training_costs:
                initial_cost = self.training_costs[0]
                final_cost = self.training_costs[-1]
                total_improvement = initial_cost - final_cost
                improvement_percent = (total_improvement / initial_cost) * 100 if initial_cost > 0 else 0
                
                print(f"Initial cost: {initial_cost:.6f}")
                print(f"Final cost: {final_cost:.6f}")
                print(f"Total improvement: {total_improvement:.6f} ({improvement_percent:.2f}%)")
        
        return self
    
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
                print(f"Prediction error: {e}")
                probabilities.append(0.5)
        
        return np.array(probabilities)
    
    def predict(self, X):
        """Predict binary classes"""
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)


def generate_irrigation_dataset(n_samples=1000):
    """Generate synthetic irrigation dataset"""
    print(f"Generating irrigation dataset ({n_samples} samples)...")
    
    np.random.seed(42)
    
    # Generate features
    soil_moisture = np.random.beta(2, 3, n_samples) * 100
    temperature = 25 + 10 * np.sin(np.linspace(0, 4*np.pi, n_samples)) + np.random.normal(0, 3, n_samples)
    temperature = np.clip(temperature, 10, 45)
    humidity = 70 - 0.5 * (temperature - 25) + np.random.normal(0, 8, n_samples)
    humidity = np.clip(humidity, 20, 95)
    
    # Simple irrigation logic
    irrigation_need = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        if soil_moisture[i] < 30 or (soil_moisture[i] < 50 and temperature[i] > 35):
            irrigation_need[i] = 1
        elif soil_moisture[i] < 20:
            irrigation_need[i] = 1
    
    # Add 3% noise
    noise_mask = np.random.random(n_samples) < 0.03
    irrigation_need[noise_mask] = 1 - irrigation_need[noise_mask]
    
    X = np.column_stack([soil_moisture, temperature, humidity])
    y = irrigation_need
    
    print(f"Dataset generated: {np.sum(y==1)}/{len(y)} irrigation needed ({np.mean(y)*100:.1f}%)")
    
    return X, y, ['soil_moisture', 'temperature', 'humidity']


def train_classical_baselines(X_train, y_train):
    """Train classical ML models"""
    print(f"\nTraining Classical Baselines")
    print("=" * 30)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    models = {}
    
    # Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    models['logistic_regression'] = (lr_model, scaler)
    
    # SVM
    print("Training SVM...")
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    models['svm'] = (svm_model, scaler)
    
    # Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    models['random_forest'] = (rf_model, None)
    
    return models


def evaluate_models(vqc_model, classical_models, X_test, y_test):
    """Evaluate all models and display results"""
    print(f"\nModel Evaluation Results")
    print("=" * 60)
    
    results = {}
    
    # Evaluate VQC
    print("Evaluating VQC...")
    vqc_pred = vqc_model.predict(X_test)
    vqc_prob = vqc_model.predict_proba(X_test)
    
    results['VQC'] = {
        'accuracy': accuracy_score(y_test, vqc_pred),
        'precision': precision_score(y_test, vqc_pred, zero_division=0),
        'recall': recall_score(y_test, vqc_pred, zero_division=0),
        'f1_score': f1_score(y_test, vqc_pred, zero_division=0),
        'auc': roc_auc_score(y_test, vqc_prob) if len(np.unique(y_test)) > 1 else 0.5
    }
    
    # Evaluate classical models
    for name, (model, scaler) in classical_models.items():
        print(f"Evaluating {name.replace('_', ' ').title()}...")
        
        if scaler is not None:
            X_test_scaled = scaler.transform(X_test)
            pred = model.predict(X_test_scaled)
            prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)[:, 1]
        
        results[name.replace('_', ' ').title()] = {
            'accuracy': accuracy_score(y_test, pred),
            'precision': precision_score(y_test, pred, zero_division=0),
            'recall': recall_score(y_test, pred, zero_division=0),
            'f1_score': f1_score(y_test, pred, zero_division=0),
            'auc': roc_auc_score(y_test, prob) if len(np.unique(y_test)) > 1 else 0.5
        }
    
    # Display results table
    print(f"\nPerformance Comparison:")
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<8}")
    print("-" * 75)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f} {metrics['auc']:<8.4f}")
    
    return results


def plot_training_progress(vqc_model, save_path=None):
    """Plot training cost progression"""
    if not vqc_model.training_costs:
        print("No training costs to plot")
        return
    
    plt.figure(figsize=(12, 4))
    
    # Cost evolution
    plt.subplot(1, 2, 1)
    costs = vqc_model.training_costs
    plt.plot(costs, 'b-', linewidth=2, alpha=0.8)
    plt.title('VQC Training Cost Evolution')
    plt.xlabel('Iteration')
    plt.ylabel('Cost (Binary Cross-Entropy)')
    plt.grid(True, alpha=0.3)
    
    # Add final cost annotation
    final_cost = costs[-1]
    plt.annotate(f'Final: {final_cost:.4f}', 
                xy=(len(costs)-1, final_cost),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Cost improvements
    plt.subplot(1, 2, 2)
    if len(costs) > 1:
        improvements = [-np.diff(costs)[i] for i in range(len(costs)-1)]
        plt.plot(improvements, 'g-', linewidth=2, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.title('Cost Improvement per Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Reduction')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training plot saved: {save_path}")
    
    plt.show()


def save_models(vqc_model, classical_models):
    """Save all trained models"""
    print(f"\nSaving Models")
    print("=" * 20)
    
    os.makedirs(CONFIG['model_dir'], exist_ok=True)
    
    # Save VQC
    vqc_path = f"{CONFIG['model_dir']}/vqc_model.pkl"
    joblib.dump(vqc_model, vqc_path)
    print(f"VQC model saved: {vqc_path}")
    
    # Save classical models
    for name, (model, scaler) in classical_models.items():
        model_path = f"{CONFIG['model_dir']}/{name}_model.pkl"
        joblib.dump(model, model_path)
        print(f"{name.title()} model saved: {model_path}")
        
        if scaler is not None:
            scaler_path = f"{CONFIG['model_dir']}/{name}_scaler.pkl"
            joblib.dump(scaler, scaler_path)
            print(f"{name.title()} scaler saved: {scaler_path}")


def main():
    """Main training pipeline - simplified and error-free"""
    print("FIXED QUANTUM-ENHANCED SMART IRRIGATION SYSTEM")
    print("=" * 60)
    print("Error-free VQC training with clear cost tracking")
    print("=" * 60)
    
    try:
        # 1. Generate dataset
        X, y, feature_names = generate_irrigation_dataset(n_samples=CONFIG['n_samples'])
        
        # 2. Train-test-validation split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=CONFIG['test_size'], random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=CONFIG['validation_size']/(1-CONFIG['test_size']), 
            random_state=42, stratify=y_temp
        )
        
        print(f"\nDataset splits:")
        print(f"Training: {len(X_train)} samples")
        print(f"Validation: {len(X_val)} samples")
        print(f"Testing: {len(X_test)} samples")
        
        # 3. Train VQC
        vqc_model = FixedQuantumIrrigationVQC(
            n_qubits=CONFIG['n_qubits'],
            n_layers=CONFIG['n_layers'],
            n_features=CONFIG['n_features'],
            learning_rate=CONFIG['learning_rate']
        )
        
        vqc_model.fit(X_train, y_train, X_val, y_val, verbose=True)
        
        # 4. Train classical baselines
        classical_models = train_classical_baselines(X_train, y_train)
        
        # 5. Evaluate all models
        results = evaluate_models(vqc_model, classical_models, X_test, y_test)
        
        # 6. Plot training progress
        os.makedirs(CONFIG['plots_dir'], exist_ok=True)
        plot_training_progress(vqc_model, f"{CONFIG['plots_dir']}/training_progress.png")
        
        # 7. Save models
        save_models(vqc_model, classical_models)
        
        # 8. Final summary
        print(f"\nTraining Summary")
        print("=" * 30)
        
        if vqc_model.training_costs:
            initial_cost = vqc_model.training_costs[0]
            final_cost = vqc_model.training_costs[-1]
            improvement = initial_cost - final_cost
            improvement_percent = (improvement / initial_cost) * 100 if initial_cost > 0 else 0
            
            print(f"VQC Training Results:")
            print(f"  Initial cost: {initial_cost:.6f}")
            print(f"  Final cost: {final_cost:.6f}")
            print(f"  Improvement: {improvement:.6f} ({improvement_percent:.2f}%)")
            print(f"  Training time: {vqc_model.training_time:.2f} seconds")
            print(f"  Iterations: {len(vqc_model.training_costs)}")
        
        # Best model
        best_model = max(results.keys(), key=lambda k: results[k]['f1_score'])
        best_f1 = results[best_model]['f1_score']
        print(f"\nBest performing model: {best_model} (F1: {best_f1:.4f})")
        
        print(f"\nTraining completed successfully!")
        print(f"Models saved in '{CONFIG['model_dir']}' directory")
        
        return vqc_model, classical_models, results
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    print("Starting Fixed VQC Training Pipeline...")
    
    # Check PennyLane
    try:
        import pennylane as qml
        print(f"PennyLane version: {qml.__version__}")
    except ImportError:
        print("Please install PennyLane: pip install pennylane")
        exit(1)
    
    # Run training
    result = main()
    
    if result[0] is not None:
        print("\nTraining completed successfully!")
        print("The VQC model is now ready for making irrigation predictions.")
    else:
        print("\nTraining encountered errors. Please check the error messages above.")
    
    print("\nNext: Use the prediction interface to make irrigation decisions!")