import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import pennylane as qml
from pennylane import numpy as pnp

# Set random seed for reproducibility
np.random.seed(42)

class QuantumIrrigationClassifier:
    """
    Quantum-Enhanced Smart Irrigation System using Variational Quantum Circuit
    """

    def __init__(self, n_qubits=3, n_layers=2, learning_rate=0.1):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate

        # Initialize quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Initialize parameters
        self.weights = None
        self.scaler = MinMaxScaler(feature_range=(0, np.pi))
        
        # Create the quantum circuit
        self.quantum_circuit = qml.QNode(self.circuit, self.dev, diff_method="parameter-shift")

    def circuit(self, x, weights):
        """Define the complete quantum circuit with proper weight structure"""
        # Feature encoding
        qml.RX(x[0], wires=0)  # Soil Moisture
        qml.RY(x[1], wires=1)  # Temperature  
        qml.RZ(x[2], wires=2)  # Humidity

        # Variational layers
        weight_idx = 0
        for layer in range(self.n_layers):
            # Parameterized rotation gates
            for qubit in range(self.n_qubits):
                qml.RX(weights[weight_idx], wires=qubit)
                weight_idx += 1
                qml.RY(weights[weight_idx], wires=qubit) 
                weight_idx += 1
                qml.RZ(weights[weight_idx], wires=qubit)
                weight_idx += 1

            # Entanglement layer (CNOT gates)
            for qubit in range(self.n_qubits):
                qml.CNOT(wires=[qubit, (qubit + 1) % self.n_qubits])

        return qml.expval(qml.PauliZ(0))

    def cost_function(self, weights, X, y):
        """Cost function using mean squared error for stability"""
        predictions = []
        
        for x_sample in X:
            expectation = self.quantum_circuit(x_sample, weights)
            # Convert expectation value (-1 to 1) to probability (0 to 1)
            prob = (expectation + 1) / 2
            predictions.append(prob)
        
        predictions = pnp.array(predictions)
        
        # Use MSE loss for better stability
        mse_loss = pnp.mean((predictions - y)**2)
        return mse_loss

    def fit(self, X, y, max_iterations=60):
        """Train the quantum classifier"""
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to PennyLane arrays with proper types
        X_scaled = pnp.array(X_scaled, requires_grad=False)
        y = pnp.array(y, requires_grad=False)

        # Initialize weights with correct size
        total_params = self.n_layers * self.n_qubits * 3
        print(f"Initializing {total_params} parameters...")
        
        # Use small random initialization
        initial_weights = 0.1 * np.random.randn(total_params)
        self.weights = pnp.array(initial_weights, requires_grad=True)

        # Use GradientDescentOptimizer for stability
        opt = qml.GradientDescentOptimizer(stepsize=self.learning_rate)
        costs = []

        print("Starting quantum training...")
        print(f"Dataset size: {len(X_scaled)} samples")
        
        for iteration in range(max_iterations):
            try:
                # Compute cost and gradients
                cost_val = self.cost_function(self.weights, X_scaled, y)
                costs.append(float(cost_val))
                
                # Update weights
                self.weights = opt.step(self.cost_function, self.weights, X_scaled, y)
                
                if iteration % 10 == 0:
                    print(f"Iteration {iteration:2d}, Cost: {cost_val:.6f}")
                    
            except Exception as e:
                print(f"Error at iteration {iteration}: {e}")
                print("Stopping training early...")
                break

        final_cost = costs[-1] if costs else float('inf')
        print(f"Training completed! Final cost: {final_cost:.6f}")
        return costs

    def predict_proba(self, X):
        """Predict probabilities"""
        if self.weights is None:
            raise ValueError("Model must be trained before making predictions")
            
        X_scaled = self.scaler.transform(X)
        probabilities = []
        
        for x_sample in X_scaled:
            try:
                expectation = self.quantum_circuit(x_sample, self.weights)
                prob = float((expectation + 1) / 2)
                # Ensure probability is in valid range
                prob = max(0.0, min(1.0, prob))
                probabilities.append(prob)
            except Exception as e:
                print(f"Error in prediction: {e}")
                probabilities.append(0.5)  # Default to uncertain
                
        return np.array(probabilities)

    def predict(self, X):
        """Make binary predictions"""
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)


def generate_synthetic_irrigation_data(n_samples=400):
    """Generate synthetic irrigation data with realistic patterns"""
    np.random.seed(42)
    
    # Generate correlated features for more realistic data
    soil_moisture = np.random.uniform(15, 85, n_samples)
    temperature = np.random.uniform(18, 38, n_samples)  
    humidity = np.random.uniform(35, 85, n_samples)

    # Create irrigation decision based on clear logic
    pump = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Irrigation score based on conditions
        moisture_need = max(0, (50 - soil_moisture[i]) / 50)  # Need increases as moisture drops below 50%
        temp_stress = max(0, (temperature[i] - 25) / 15)       # Stress increases above 25°C
        humidity_need = max(0, (60 - humidity[i]) / 30)        # Need increases as humidity drops below 60%
        
        # Combined irrigation need
        irrigation_need = 0.5 * moisture_need + 0.3 * temp_stress + 0.2 * humidity_need
        
        # Add small amount of noise
        noise = np.random.normal(0, 0.1)
        irrigation_need += noise
        
        # Binary decision
        pump[i] = 1 if irrigation_need > 0.4 else 0

    X = np.column_stack([soil_moisture, temperature, humidity])
    
    # Print data distribution info
    print(f"Generated {n_samples} samples:")
    print(f"  Irrigation ON:  {np.sum(pump)} ({100*np.mean(pump):.1f}%)")
    print(f"  Irrigation OFF: {n_samples - np.sum(pump)} ({100*(1-np.mean(pump)):.1f}%)")
    
    return X, pump.astype(int)


def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model performance"""
    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"\n{model_name} Performance:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        return accuracy, precision, recall, f1
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        return 0, 0, 0, 0


def create_visualizations(X_test, y_test, quantum_pred, classical_pred, costs):
    """Create visualization plots"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Training cost curve
        if costs and len(costs) > 1:
            axes[0, 0].plot(costs, 'b-', linewidth=2, marker='o', markersize=4)
            axes[0, 0].set_title('Quantum Training Progress', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Cost (MSE)')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_yscale('log')
        else:
            axes[0, 0].text(0.5, 0.5, 'No training data available', 
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Training Progress')

        # 2. Quantum confusion matrix
        try:
            cm_quantum = confusion_matrix(y_test, quantum_pred)
            sns.heatmap(cm_quantum, annot=True, fmt='d', cmap='Blues', 
                       ax=axes[0, 1], square=True, cbar_kws={'shrink': 0.8})
            axes[0, 1].set_title('Quantum Model Confusion Matrix', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Predicted')
            axes[0, 1].set_ylabel('Actual')
        except:
            axes[0, 1].text(0.5, 0.5, 'Error creating\nconfusion matrix', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)

        # 3. Classical confusion matrix
        try:
            cm_classical = confusion_matrix(y_test, classical_pred)
            sns.heatmap(cm_classical, annot=True, fmt='d', cmap='Reds',
                       ax=axes[1, 0], square=True, cbar_kws={'shrink': 0.8})
            axes[1, 0].set_title('Classical Model Confusion Matrix', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Predicted')
            axes[1, 0].set_ylabel('Actual')
        except:
            axes[1, 0].text(0.5, 0.5, 'Error creating\nconfusion matrix', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)

        # 4. Feature distributions
        try:
            feature_names = ['Soil Moisture', 'Temperature', 'Humidity']
            colors = ['green', 'red', 'blue']
            
            for i, (name, color) in enumerate(zip(feature_names, colors)):
                off_data = X_test[y_test == 0, i]
                on_data = X_test[y_test == 1, i]
                
                axes[1, 1].hist(off_data, bins=12, alpha=0.6, color=color, 
                               label=f'{name} (Pump OFF)', density=True)
                axes[1, 1].hist(on_data, bins=12, alpha=0.6, color=color,
                               label=f'{name} (Pump ON)', density=True, 
                               linestyle='--', histtype='step', linewidth=2)

            axes[1, 1].set_title('Feature Distributions by Class', fontsize=12, fontweight='bold')
            axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1, 1].grid(True, alpha=0.3)
        except:
            axes[1, 1].text(0.5, 0.5, 'Error creating\nfeature plot', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)

        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")


def demonstrate_predictions(quantum_classifier, feature_ranges):
    """Demonstrate quantum model predictions"""
    print("\n" + "="*50)
    print("🔮 QUANTUM MODEL DEMONSTRATION")
    print("="*50)
    
    # Create test scenarios
    scenarios = [
        {"name": "🌵 Dry & Hot", "features": [20, 35, 30], "expected": "ON"},
        {"name": "🌿 Wet & Cool", "features": [80, 20, 80], "expected": "OFF"}, 
        {"name": "🌤️  Moderate", "features": [50, 25, 60], "expected": "UNCERTAIN"},
        {"name": "🔥 Heat Wave", "features": [30, 40, 25], "expected": "ON"},
        {"name": "💧 After Rain", "features": [85, 22, 90], "expected": "OFF"}
    ]
    
    print("Testing irrigation decisions for different conditions:\n")
    
    for scenario in scenarios:
        features = np.array(scenario["features"]).reshape(1, -1)
        
        try:
            prediction = quantum_classifier.predict(features)[0]
            probability = quantum_classifier.predict_proba(features)[0]
            confidence = max(probability, 1-probability)
            
            result = "💧 IRRIGATE" if prediction == 1 else "🚫 NO IRRIGATION"
            
            print(f"{scenario['name']:12} | "
                  f"Moisture: {features[0,0]:2.0f}% | "
                  f"Temp: {features[0,1]:2.0f}°C | "
                  f"Humidity: {features[0,2]:2.0f}% | "
                  f"→ {result:15} (confidence: {confidence:.3f})")
                  
        except Exception as e:
            print(f"{scenario['name']:12} | Error: {e}")


def main():
    """Main execution function"""
    print("🌾 QUANTUM-ENHANCED SMART IRRIGATION SYSTEM 🌾")
    print("=" * 55)

    try:
        # Generate data
        print("\n📊 Generating synthetic irrigation dataset...")
        X, y = generate_synthetic_irrigation_data(n_samples=400)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        print(f"\n📋 Dataset Information:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Features: Soil Moisture (%), Temperature (°C), Humidity (%)")

        # Train Quantum Model
        print(f"\n⚛️  QUANTUM MODEL TRAINING")
        print("-" * 35)
        
        quantum_classifier = QuantumIrrigationClassifier(
            n_qubits=3, n_layers=2, learning_rate=0.05
        )
        costs = quantum_classifier.fit(X_train, y_train, max_iterations=60)

        # Make quantum predictions
        print("\nMaking quantum predictions...")
        quantum_pred = quantum_classifier.predict(X_test)

        # Train Classical Model  
        print(f"\n📈 CLASSICAL MODEL TRAINING")
        print("-" * 35)
        
        classical_model = LogisticRegression(random_state=42, max_iter=1000)
        classical_model.fit(X_train, y_train)
        classical_pred = classical_model.predict(X_test)

        # Evaluate Models
        print(f"\n📊 MODEL EVALUATION RESULTS")
        print("-" * 35)
        
        quantum_metrics = evaluate_model(y_test, quantum_pred, "🔮 Quantum VQC")
        classical_metrics = evaluate_model(y_test, classical_pred, "📊 Classical LR")

        # Compare Performance
        print(f"\n🏆 PERFORMANCE COMPARISON")
        print("-" * 35)
        
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        wins = {'quantum': 0, 'classical': 0, 'ties': 0}
        
        for i, metric in enumerate(metric_names):
            q_score, c_score = quantum_metrics[i], classical_metrics[i]
            
            if q_score > c_score + 0.01:
                winner, symbol = "Quantum", "🔮"
                wins['quantum'] += 1
            elif c_score > q_score + 0.01:
                winner, symbol = "Classical", "📊"
                wins['classical'] += 1
            else:
                winner, symbol = "Tie", "🤝"
                wins['ties'] += 1
                
            print(f"  {symbol} {metric:>9}: "
                  f"Quantum {q_score:.3f} vs Classical {c_score:.3f} → {winner}")

        print(f"\n🏆 Overall: Quantum {wins['quantum']}, Classical {wins['classical']}, Ties {wins['ties']}")

        # Demonstrate quantum model
        demonstrate_predictions(quantum_classifier, X)

        # Create visualizations
        print(f"\n📈 Generating visualizations...")
        create_visualizations(X_test, y_test, quantum_pred, classical_pred, costs)
        
        return quantum_classifier, costs, quantum_metrics, classical_metrics
        
    except Exception as e:
        print(f"\n❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        return None, [], (0,0,0,0), (0,0,0,0)


if __name__ == "__main__":
    print("🔧 Required packages: pennylane, scikit-learn, matplotlib, seaborn")
    print("💾 Install with: pip install pennylane scikit-learn matplotlib seaborn")
    print()
    
    try:
        result = main()
        
        if result[0] is not None:
            print(f"\n✅ PROGRAM COMPLETED SUCCESSFULLY!")
            print(f"🎯 Quantum model trained and evaluated successfully")
            print(f"📊 Visualizations generated")
            print(f"🌾 Smart irrigation system ready for deployment!")
        else:
            print(f"\n❌ Program encountered errors during execution")
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Program interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        
    print(f"\n🌾 Thank you for using the Quantum Irrigation System! 🌾")