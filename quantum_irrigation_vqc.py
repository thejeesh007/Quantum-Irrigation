import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import seaborn as sns
import pennylane as qml
from pennylane import numpy as pnp
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
pnp.random.seed(42)

# SET YOUR DATASET FILEPATH HERE
DATASET_FILEPATH = "Soil Moisture, Air Temperature and humidity, and Water Motor onoff Monitor data.AmritpalKaur.csv"

class QuantumIrrigationClassifier:
    def __init__(self, n_qubits=3, n_layers=2, learning_rate=0.1):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate

        # Initialize quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Initialize parameters
        self.weights = None
        self.scaler = StandardScaler()  
        
        # Create the quantum circuit
        self.quantum_circuit = qml.QNode(self.circuit, self.dev, diff_method="parameter-shift")

    def circuit(self, x, weights):
        """Simplified quantum circuit that works reliably"""
        # Ensure we have exactly 3 features for 3 qubits
        if len(x) != 3:
            raise ValueError(f"Expected 3 features, got {len(x)}")
            
        # Feature encoding - simple rotation encoding
        qml.RY(x[0], wires=0)  # Soil moisture
        qml.RY(x[1], wires=1)  # Temperature  
        qml.RY(x[2], wires=2)  # Humidity

        # Variational layers
        for layer in range(self.n_layers):
            # Single rotation per qubit per layer
            for qubit in range(self.n_qubits):
                weight_idx = layer * self.n_qubits + qubit
                if weight_idx < len(weights):
                    qml.RY(weights[weight_idx], wires=qubit)

            # Simple entanglement pattern
            for qubit in range(self.n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])

        # Measurement on the first qubit
        return qml.expval(qml.PauliZ(0))

    def cost_function(self, weights, X, y):
        """Binary cross-entropy loss"""
        total_loss = 0
        n_samples = len(X)
        
        for i, x_sample in enumerate(X):
            try:
                # Get quantum prediction
                expectation = self.quantum_circuit(x_sample, weights)
                # Convert to probability [0, 1]
                prob = (expectation + 1) / 2
                prob = pnp.clip(prob, 1e-7, 1 - 1e-7)
                
                # Binary cross-entropy
                loss = -y[i] * pnp.log(prob) - (1 - y[i]) * pnp.log(1 - prob)
                total_loss += loss
                
            except Exception as e:
                print(f"Circuit evaluation error for sample {i}: {e}")
                total_loss += 1.0  # Penalty for failed evaluation
        
        return total_loss / n_samples

    def fit(self, X, y, max_iterations=100):
        """Train the quantum classifier"""
        print("Preprocessing features...")
        
        # Ensure we have the right number of features
        if X.shape[1] != 3:
            raise ValueError(f"Expected 3 features, got {X.shape[1]}")
        
        # Scale features to [0, π] range for rotations
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = X_scaled * np.pi / 2  # Scale to [0, π/2] for RY gates
        
        # Convert to PennyLane arrays
        X_scaled = pnp.array(X_scaled, requires_grad=False)
        y = pnp.array(y.astype(float), requires_grad=False)

        # Initialize weights - one parameter per qubit per layer
        total_params = self.n_layers * self.n_qubits
        print(f"Initializing {total_params} quantum parameters...")
        
        # Small random initialization
        self.weights = pnp.array(0.1 * np.random.randn(total_params), requires_grad=True)

        # Use Adam optimizer which works better for quantum circuits
        opt = qml.AdamOptimizer(stepsize=self.learning_rate)
        costs = []

        print(f"Starting quantum training for {max_iterations} iterations...")
        print("This will take time as we're using classical optimization to minimize quantum cost function...")
        
        for iteration in range(max_iterations):
            try:
                # Calculate current cost
                cost_val = self.cost_function(self.weights, X_scaled, y)
                costs.append(float(cost_val))
                
                # Update weights using classical optimizer
                self.weights = opt.step(lambda w: self.cost_function(w, X_scaled, y), self.weights)
                
                # Progress reporting
                if iteration % 10 == 0:
                    print(f"Iteration {iteration:3d}/{max_iterations} | Cost: {cost_val:.6f}")
                    
                # Early stopping if cost becomes very small
                if cost_val < 1e-6:
                    print(f"Converged at iteration {iteration}")
                    break
                    
            except Exception as e:
                print(f"Training error at iteration {iteration}: {e}")
                break

        final_cost = costs[-1] if costs else float('inf')
        print(f"Training completed! Final cost: {final_cost:.6f}")
        return costs

    def predict_proba(self, X):
        """Predict class probabilities with proper error handling"""
        if self.weights is None:
            raise ValueError("Model must be trained before making predictions")
        
        if X.shape[1] != 3:
            raise ValueError(f"Expected 3 features, got {X.shape[1]}")
            
        # Apply same scaling as training
        X_scaled = self.scaler.transform(X) * np.pi / 2
        
        probabilities = []
        
        for i, x_sample in enumerate(X_scaled):
            try:
                # Ensure x_sample is the right shape and type
                x_sample = np.array(x_sample).flatten()
                if len(x_sample) != 3:
                    raise ValueError(f"Sample {i} has wrong shape: {x_sample.shape}")
                
                expectation = self.quantum_circuit(x_sample, self.weights)
                prob = float((expectation + 1) / 2)
                prob = max(0.0, min(1.0, prob))  # Clamp to [0,1]
                probabilities.append(prob)
                
            except Exception as e:
                print(f"Prediction error for sample {i}: {e}")
                probabilities.append(0.5)  # Default to uncertain
                
        return np.array(probabilities)

    def predict(self, X):
        """Predict binary class (0 or 1)"""
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)


def load_and_preprocess_data(filepath):
    """Load your specific dataset"""
    try:
        print(f"Loading dataset from: {filepath}")
        df = pd.read_csv(filepath)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found!")
        print("Please ensure the CSV file is in the same directory as this script.")
        raise
    except Exception as e:
        print(f"Error loading file: {e}")
        raise
    
    # Display first few rows to understand data structure
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # Clean column names
    df.columns = [col.strip().replace(' ', '_') for col in df.columns]
    print(f"\nCleaned columns: {df.columns.tolist()}")
    
    # Try to identify the correct columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Numeric columns: {numeric_cols}")
    
    if len(numeric_cols) < 4:
        raise ValueError(f"Need at least 4 numeric columns, found {len(numeric_cols)}")
    
    # Map to standard names - you may need to adjust these based on your CSV structure
    feature_cols = numeric_cols[:3]  # First 3 numeric columns as features
    target_col = numeric_cols[3]     # 4th numeric column as target
    
    print(f"\nUsing columns:")
    print(f"  Features: {feature_cols}")
    print(f"  Target: {target_col}")
    
    # Extract features and target
    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(int)
    
    # Clean data - remove any invalid values
    valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Ensure binary target
    y = (y > 0).astype(int)
    
    print(f"\nFinal dataset info:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Target distribution: OFF={np.sum(y==0)}, ON={np.sum(y==1)} ({np.mean(y)*100:.1f}% ON)")
    print(f"  Feature ranges:")
    for i, col in enumerate(feature_cols):
        print(f"    {col}: {X[:,i].min():.2f} - {X[:,i].max():.2f}")
    
    return X, y, feature_cols


def evaluate_model(y_true, y_pred, model_name):
    """Evaluate and print model metrics"""
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


def create_visualizations(X_test, y_test, quantum_pred, classical_pred, svm_pred, costs, quantum_proba, classical_proba, svm_proba, feature_names):
    """Create comprehensive visualizations"""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Quantum vs Classical Irrigation Models Comparison', fontsize=16, fontweight='bold')

    # 1. Training Cost Curve
    if len(costs) > 1:
        axes[0, 0].plot(costs, 'b-', linewidth=2, marker='o', markersize=3)
        axes[0, 0].set_title('Quantum Training Progress', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss (BCE)')
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'Insufficient training data', ha='center', va='center', 
                       transform=axes[0,0].transAxes, fontsize=12)

    # 2-4. Confusion Matrices
    models_data = [
        (quantum_pred, 'Quantum VQC', 'Blues'),
        (classical_pred, 'Logistic Regression', 'Greens'),
        (svm_pred, 'SVM', 'Oranges')
    ]
    
    for idx, (pred, name, color) in enumerate(models_data):
        ax = axes[0, 1] if idx == 0 else (axes[0, 2] if idx == 1 else axes[1, 0])
        try:
            cm = confusion_matrix(y_test, pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap=color, ax=ax, 
                       square=True, cbar_kws={'shrink': 0.8})
            ax.set_title(f'{name} Confusion Matrix', fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:20]}...', ha='center', va='center', 
                   transform=ax.transAxes)

    # 5. ROC Curves
    try:
        if len(np.unique(y_test)) > 1:  # Need both classes
            probas = [quantum_proba, classical_proba, svm_proba]
            names = ['Quantum VQC', 'Logistic Regression', 'SVM']
            colors = ['blue', 'green', 'orange']
            
            for proba, name, color in zip(probas, names, colors):
                fpr, tpr, _ = roc_curve(y_test, proba)
                roc_auc = auc(fpr, tpr)
                axes[1, 1].plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})', 
                               color=color, linewidth=2)
            
            axes[1, 1].plot([0,1], [0,1], 'k--', alpha=0.5, label='Random')
            axes[1, 1].set_title('ROC Curves Comparison', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('False Positive Rate')
            axes[1, 1].set_ylabel('True Positive Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Need both classes for ROC', ha='center', va='center', 
                           transform=axes[1,1].transAxes)
    except Exception as e:
        axes[1, 1].text(0.5, 0.5, f'ROC Error', ha='center', va='center', 
                       transform=axes[1,1].transAxes)

    # 6. Feature Distributions
    try:
        colors = ['red', 'blue', 'green']
        for i, (name, color) in enumerate(zip(feature_names, colors)):
            if len(np.unique(y_test)) > 1:
                off_data = X_test[y_test == 0, i] if np.sum(y_test == 0) > 0 else []
                on_data = X_test[y_test == 1, i] if np.sum(y_test == 1) > 0 else []
                
                if len(off_data) > 0:
                    axes[1, 2].hist(off_data, bins=15, alpha=0.5, label=f'{name} (OFF)', 
                                   density=True, color=color)
                if len(on_data) > 0:
                    axes[1, 2].hist(on_data, bins=15, alpha=0.7, label=f'{name} (ON)', 
                                   density=True, color=color, histtype='step', linewidth=2)
        
        axes[1, 2].set_title('Feature Distributions by Class', fontsize=12, fontweight='bold')
        axes[1, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 2].grid(True, alpha=0.3)
    except Exception as e:
        axes[1, 2].text(0.5, 0.5, f'Feature Plot Error', ha='center', va='center', 
                       transform=axes[1,2].transAxes)

    plt.tight_layout()
    plt.show()


def demonstrate_predictions(quantum_classifier, X, feature_names):
    """Demonstrate quantum model on realistic scenarios based on actual data"""
    print("\n" + "="*60)
    print("QUANTUM IRRIGATION DECISIONS — REAL-WORLD SCENARIOS")
    print("="*60)
    
    # Use actual data ranges
    feature_ranges = [(X[:,i].min(), X[:,i].max()) for i in range(3)]
    
    # Define test scenarios
    scenarios = [
        {
            "name": "Dry & Hot", 
            "features": [
                feature_ranges[0][0] + 0.1 * (feature_ranges[0][1] - feature_ranges[0][0]),  # Low moisture
                feature_ranges[1][1] - 0.1 * (feature_ranges[1][1] - feature_ranges[1][0]),  # High temp
                feature_ranges[2][0] + 0.2 * (feature_ranges[2][1] - feature_ranges[2][0])   # Low humidity
            ],
            "desc": "Low moisture, high temp → Should irrigate"
        },
        {
            "name": "Wet & Cool", 
            "features": [
                feature_ranges[0][1] - 0.1 * (feature_ranges[0][1] - feature_ranges[0][0]),  # High moisture
                feature_ranges[1][0] + 0.2 * (feature_ranges[1][1] - feature_ranges[1][0]),  # Low temp
                feature_ranges[2][1] - 0.1 * (feature_ranges[2][1] - feature_ranges[2][0])   # High humidity
            ],
            "desc": "High moisture, low temp → Hold irrigation"
        },
        {
            "name": "Moderate", 
            "features": [
                np.mean(feature_ranges[0]),  # Average moisture
                np.mean(feature_ranges[1]),  # Average temp
                np.mean(feature_ranges[2])   # Average humidity
            ],
            "desc": "Balanced conditions → Model decides"
        }
    ]
    
    print(f"\nTesting quantum model on irrigation scenarios:")
    print(f"(Using {feature_names[0]}, {feature_names[1]}, {feature_names[2]})\n")
    
    for scenario in scenarios:
        features_array = np.array(scenario["features"]).reshape(1, -1)
        
        try:
            prediction = quantum_classifier.predict(features_array)[0]
            probability = quantum_classifier.predict_proba(features_array)[0]
            confidence = max(probability, 1 - probability)
            
            decision = "IRRIGATE (ON)" if prediction == 1 else "HOLD (OFF)"
            
            print(f"{scenario['name']:12} | "
                  f"{feature_names[0][:8]}: {features_array[0,0]:.1f} | "
                  f"{feature_names[1][:4]}: {features_array[0,1]:.1f} | "
                  f"{feature_names[2][:8]}: {features_array[0,2]:.1f} | "
                  f"→ {decision:12} (confidence: {confidence:.3f})")
            print(f"               → {scenario['desc']}\n")
                  
        except Exception as e:
            print(f"{scenario['name']} | Error: {e}")


def main():
    print("QUANTUM SMART IRRIGATION SYSTEM")
    print("="*50)

    try:
        # Load your specific dataset
        print("\nLoading irrigation dataset...")
        X, y, feature_names = load_and_preprocess_data(DATASET_FILEPATH)

        # Check dataset size for appropriate split
        if len(X) < 100:
            test_size = 0.3
        else:
            test_size = 0.25

        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        except ValueError:
            # If stratification fails
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        print(f"\nTrain/Test Split:")
        print(f"  Training samples: {len(X_train)} (OFF: {np.sum(y_train==0)}, ON: {np.sum(y_train==1)})")
        print(f"  Test samples: {len(X_test)} (OFF: {np.sum(y_test==0)}, ON: {np.sum(y_test==1)})")

        # TRAIN QUANTUM MODEL
        print(f"\nTRAINING QUANTUM VQC MODEL")
        print("-"*40)
        quantum_classifier = QuantumIrrigationClassifier(
            n_qubits=3, 
            n_layers=2, 
            learning_rate=0.1
        )
        
        # Reduced iterations for faster testing - increase for better results
        costs = quantum_classifier.fit(X_train, y_train, max_iterations=50)

        # Quantum predictions
        print("\nMaking quantum predictions...")
        quantum_pred = quantum_classifier.predict(X_test)
        quantum_proba = quantum_classifier.predict_proba(X_test)

        # TRAIN CLASSICAL MODELS
        print(f"\nTRAINING CLASSICAL MODELS")
        print("-"*40)
        
        # Logistic Regression
        print("Training Logistic Regression...")
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        lr_proba = lr_model.predict_proba(X_test)[:, 1]
        
        # SVM
        print("Training SVM...")
        svm_model = SVC(random_state=42, probability=True)
        svm_model.fit(X_train, y_train)
        svm_pred = svm_model.predict(X_test)
        svm_proba = svm_model.predict_proba(X_test)[:, 1]

        # EVALUATE ALL MODELS
        print(f"\nMODEL EVALUATION")
        print("-"*40)
        quantum_metrics = evaluate_model(y_test, quantum_pred, "Quantum VQC")
        lr_metrics = evaluate_model(y_test, lr_pred, "Logistic Regression")
        svm_metrics = evaluate_model(y_test, svm_pred, "SVM")

        # Compare models
        print(f"\nPERFORMANCE COMPARISON (F1-Score)")
        print("-"*40)
        models = ["Quantum VQC", "Logistic Regression", "SVM"]
        scores = [quantum_metrics[3], lr_metrics[3], svm_metrics[3]]
        best_idx = np.argmax(scores)
        
        for i, (model, score) in enumerate(zip(models, scores)):
            star = "★" if i == best_idx else " "
            print(f"{star} {model:>20}: {score:.4f}")

        # VISUALIZATIONS
        print(f"\nGENERATING VISUALIZATIONS...")
        create_visualizations(X_test, y_test, quantum_pred, lr_pred, svm_pred, 
                            costs, quantum_proba, lr_proba, svm_proba, feature_names)

        # DEMONSTRATE PREDICTIONS
        demonstrate_predictions(quantum_classifier, X, feature_names)

        print(f"\nTRAINING COMPLETE!")
        print(f"Model summary:")
        print(f"  - Quantum parameters: {len(quantum_classifier.weights)}")
        print(f"  - Training iterations: {len(costs)}")
        print(f"  - Final training cost: {costs[-1]:.6f}" if costs else "  - No training cost recorded")
        
        # Training time note
        print(f"\nNOTE ON TRAINING TIME:")
        print(f"- Quantum training uses classical optimization (Adam) to minimize quantum cost function")
        print(f"- Each iteration evaluates quantum circuit for all training samples")
        print(f"- Training time scales as O(iterations × samples × circuit_depth)")
        print(f"- For faster results, reduce max_iterations or use smaller dataset")
        
        return quantum_classifier, costs, quantum_metrics, lr_metrics, svm_metrics
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None, [], (0,0,0,0), (0,0,0,0), (0,0,0,0)


if __name__ == "__main__":
    print("Starting Quantum Smart Irrigation System")
    print("This uses classical optimization to train quantum parameters...")
    print("Training time depends on dataset size and iterations.\n")
    
    # Run main function
    result = main()
    
    if result[0] is not None:
        quantum_classifier, costs, quantum_metrics, lr_metrics, svm_metrics = result
        print(f"\nSUCCESS! Quantum irrigation model trained.")
        print(f"Performance Summary:")
        print(f"  Quantum VQC F1-Score: {quantum_metrics[3]:.4f}")
        print(f"  Logistic Regression F1-Score: {lr_metrics[3]:.4f}")
        print(f"  SVM F1-Score: {svm_metrics[3]:.4f}")
        
        # Show example usage
        print(f"\nExample Usage:")
        print(f"quantum_classifier.predict([[soil_moisture, temperature, humidity]])")
        
    else:
        print(f"\nTraining failed. Please check:")
        print(f"1. Dataset file path is correct")
        print(f"2. CSV file has at least 4 numeric columns")
        print(f"3. Required packages are installed:")
        print(f"   pip install pennylane scikit-learn matplotlib seaborn pandas")
    
    print(f"\nThank you for using Quantum-Enhanced Smart Irrigation!")