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

# Configuration
CONFIG = {
    # Dataset parameters
    'csv_file': 'Soil Moisture, Air Temperature and humidity, and Water Motor onoff Monitor data.AmritpalKaur.csv',
    'feature_columns': ['Soil Moisture', 'Temperature', 'Air Humidity'],
    'label_column': 'Pump Data',
    'test_size': 0.2,
    'validation_size': 0.2,
    
    # VQC parameters
    'n_qubits': 4,
    'n_layers': 3,
    'n_features': 3,
    
    # Training parameters
    'learning_rate': 0.01,
    'max_iterations': 100,
    'batch_size': 32,
    'early_stopping_patience': 15,
    
    # File paths
    'model_dir': 'models',
    'plots_dir': 'plots',
    'results_dir': 'results',
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
        
        print(f"✓ VQC initialized: {n_qubits} qubits, {n_layers} layers")
    
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
    
    def _cost_function(self, weights, X, y):
        """Binary cross-entropy cost function"""
        predictions = []
        
        for x_sample in X:
            expectation = self.quantum_circuit(x_sample, weights)
            prob = self._prediction_from_expectation(expectation)
            predictions.append(prob)
        
        predictions = pnp.array(predictions)
        epsilon = 1e-7
        predictions = pnp.clip(predictions, epsilon, 1 - epsilon)
        cost = -pnp.mean(y * pnp.log(predictions) + (1 - y) * pnp.log(1 - predictions))
        
        return cost
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """Train the VQC"""
        start_time = time.time()
        
        if verbose:
            print(f"\n{'='*60}")
            print("TRAINING VQC MODEL")
            print('='*60)
            print(f"Training samples: {len(X_train)}")
            if X_val is not None:
                print(f"Validation samples: {len(X_val)}")
        
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
        
        optimizer = qml.AdamOptimizer(stepsize=self.learning_rate)
        
        # Training tracking
        self.training_costs = []
        best_val_cost = float('inf')
        patience_counter = 0
        previous_cost = float('inf')
        
        if verbose:
            print("\nTraining Progress:")
            print(f"{'Iter':<6} {'Cost':<12} {'Δ Cost':<12} {'Val Cost':<12} {'Patience':<8}")
            print("-" * 60)
        
        for iteration in range(CONFIG['max_iterations']):
            # Mini-batch training
            if len(X_tensor) > CONFIG['batch_size']:
                indices = pnp.random.choice(len(X_tensor), size=CONFIG['batch_size'], replace=False)
                X_batch = X_tensor[indices]
                y_batch = y_tensor[indices]
            else:
                X_batch = X_tensor
                y_batch = y_tensor
            
            try:
                self.weights, current_cost = optimizer.step_and_cost(
                    lambda w: self._cost_function(w, X_batch, y_batch), 
                    self.weights
                )
                
                current_cost = float(current_cost)
                self.training_costs.append(current_cost)
                improvement = previous_cost - current_cost
                previous_cost = current_cost
                
                # Validation
                val_cost_str = "N/A"
                if X_val_tensor is not None:
                    val_cost = float(self._cost_function(self.weights, X_val_tensor, y_val_tensor))
                    val_cost_str = f"{val_cost:.6f}"
                    
                    if val_cost < best_val_cost:
                        best_val_cost = val_cost
                        patience_counter = 0
                    else:
                        patience_counter += 1
                
                if verbose and (iteration % 10 == 0 or iteration < 10):
                    print(f"{iteration:<6} {current_cost:<12.6f} {improvement:+12.6f} {val_cost_str:<12} {patience_counter:<8}")
                
                # Early stopping
                if patience_counter >= CONFIG['early_stopping_patience']:
                    if verbose:
                        print(f"\n✓ Early stopping at iteration {iteration}")
                    break
                
                # Convergence check
                if current_cost < 1e-6:
                    if verbose:
                        print(f"\n✓ Converged at iteration {iteration}")
                    break
                
                # NaN check
                if np.isnan(current_cost):
                    print(f"\n⚠ Warning: NaN detected at iteration {iteration}")
                    break
                    
            except Exception as e:
                print(f"\n❌ Training error at iteration {iteration}: {e}")
                break
        
        self.training_time = time.time() - start_time
        self.is_trained = True
        
        if verbose:
            print("-" * 60)
            print(f"✓ Training completed in {self.training_time:.2f} seconds")
            print(f"  Final cost: {self.training_costs[-1]:.6f}")
            print(f"  Total iterations: {len(self.training_costs)}")
        
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


def load_dataset(csv_file):
    """Load and preprocess irrigation dataset"""
    print(f"\n{'='*60}")
    print("LOADING DATASET")
    print('='*60)
    print(f"File: {csv_file}")
    
    try:
        # Load CSV
        df = pd.read_csv(csv_file)
        print(f"✓ Dataset loaded successfully")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Check required columns
        required_features = CONFIG['feature_columns']
        required_label = CONFIG['label_column']
        
        missing_cols = []
        for col in required_features + [required_label]:
            if col not in df.columns:
                missing_cols.append(col)
        
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        print(f"✓ All required columns found")
        
        # Handle missing values
        missing_count = df[required_features + [required_label]].isnull().sum()
        if missing_count.sum() > 0:
            print(f"\n⚠ Missing values detected:")
            for col, count in missing_count.items():
                if count > 0:
                    print(f"  {col}: {count}")
            
            # Fill missing values
            for col in required_features:
                if df[col].isnull().sum() > 0:
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    print(f"  ✓ Filled {col} with median: {median_val:.2f}")
            
            if df[required_label].isnull().sum() > 0:
                mode_val = df[required_label].mode()[0]
                df[required_label].fillna(mode_val, inplace=True)
                print(f"  ✓ Filled {required_label} with mode: {mode_val}")
        else:
            print(f"✓ No missing values")
        
        # Extract features and labels
        X = df[required_features].values
        y = df[required_label].values
        
        # Validate labels (must be binary: 0 or 1)
        unique_labels = np.unique(y)
        if not set(unique_labels).issubset({0, 1}):
            print(f"⚠ Warning: Non-binary labels detected: {unique_labels}")
            print(f"  Converting to binary using median threshold...")
            threshold = np.median(y)
            y = (y > threshold).astype(int)
        
        print(f"\n{'='*60}")
        print("DATASET SUMMARY")
        print('='*60)
        print(f"Total samples: {len(X)}")
        print(f"Features: {required_features}")
        print(f"\nLabel distribution:")
        print(f"  Class 0 (No Irrigation): {np.sum(y==0)} ({np.mean(y==0)*100:.1f}%)")
        print(f"  Class 1 (Irrigation): {np.sum(y==1)} ({np.mean(y==1)*100:.1f}%)")
        
        print(f"\nFeature statistics:")
        for i, feature in enumerate(required_features):
            print(f"  {feature}:")
            print(f"    Min: {X[:, i].min():.2f}, Max: {X[:, i].max():.2f}")
            print(f"    Mean: {X[:, i].mean():.2f}, Std: {X[:, i].std():.2f}")
        
        return X, y, required_features
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: File '{csv_file}' not found!")
        print(f"\nPlease ensure the CSV file exists with these columns:")
        print(f"  Features: {CONFIG['feature_columns']}")
        print(f"  Label: {CONFIG['label_column']}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR loading dataset: {e}")
        raise


def train_classical_models(X_train, y_train):
    """Train classical ML baseline models"""
    print(f"\n{'='*60}")
    print("TRAINING CLASSICAL MODELS")
    print('='*60)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    models = {}
    
    # Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    models['logistic_regression'] = (lr_model, scaler)
    print("✓ Logistic Regression trained")
    
    # SVM
    print("Training Support Vector Machine...")
    svm_model = SVC(probability=True, random_state=42, kernel='rbf')
    svm_model.fit(X_train_scaled, y_train)
    models['svm'] = (svm_model, scaler)
    print("✓ SVM trained")
    
    # Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    models['random_forest'] = (rf_model, None)
    print("✓ Random Forest trained")
    
    return models


def evaluate_all_models(vqc_model, classical_models, X_test, y_test):
    """Evaluate and compare all models"""
    print(f"\n{'='*60}")
    print("MODEL EVALUATION")
    print('='*60)
    
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
        model_name = name.replace('_', ' ').title()
        print(f"Evaluating {model_name}...")
        
        if scaler is not None:
            X_test_scaled = scaler.transform(X_test)
            pred = model.predict(X_test_scaled)
            prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)[:, 1]
        
        results[model_name] = {
            'accuracy': accuracy_score(y_test, pred),
            'precision': precision_score(y_test, pred, zero_division=0),
            'recall': recall_score(y_test, pred, zero_division=0),
            'f1_score': f1_score(y_test, pred, zero_division=0),
            'auc': roc_auc_score(y_test, prob) if len(np.unique(y_test)) > 1 else 0.5
        }
    
    # Display comparison table
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print('='*60)
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<8}")
    print("-" * 78)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f} {metrics['auc']:<8.4f}")
    
    print('='*60)
    
    # Identify best model
    best_model = max(results.keys(), key=lambda k: results[k]['f1_score'])
    best_f1 = results[best_model]['f1_score']
    print(f"\n🏆 Best Model: {best_model} (F1-Score: {best_f1:.4f})")
    
    return results


def save_all_models(vqc_model, classical_models):
    """Save all trained models and scalers"""
    print(f"\n{'='*60}")
    print("SAVING MODELS")
    print('='*60)
    
    os.makedirs(CONFIG['model_dir'], exist_ok=True)
    
    # Save VQC
    vqc_path = os.path.join(CONFIG['model_dir'], 'vqc_model.pkl')
    joblib.dump(vqc_model, vqc_path)
    print(f"✓ VQC model saved: {vqc_path}")
    
    scaler_path = os.path.join(CONFIG['model_dir'], 'vqc_scaler.pkl')
    joblib.dump(vqc_model.scaler, scaler_path)
    print(f"✓ VQC scaler saved: {scaler_path}")
    
    # Save classical models
    for name, (model, scaler) in classical_models.items():
        model_path = os.path.join(CONFIG['model_dir'], f'{name}_model.pkl')
        joblib.dump(model, model_path)
        print(f"✓ {name.replace('_', ' ').title()} saved: {model_path}")
        
        if scaler is not None:
            scaler_path = os.path.join(CONFIG['model_dir'], f'{name}_scaler.pkl')
            joblib.dump(scaler, scaler_path)
            print(f"✓ {name.replace('_', ' ').title()} scaler saved: {scaler_path}")
    
    print(f"\n✓ All models saved in '{CONFIG['model_dir']}/' directory")


def plot_training_curve(vqc_model):
    """Plot VQC training cost evolution"""
    if not vqc_model.training_costs:
        print("⚠ No training costs to plot")
        return
    
    os.makedirs(CONFIG['plots_dir'], exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Cost evolution
    plt.subplot(1, 2, 1)
    costs = vqc_model.training_costs
    plt.plot(costs, 'b-', linewidth=2, alpha=0.8)
    plt.title('VQC Training Cost Evolution', fontsize=14, fontweight='bold')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cost (Binary Cross-Entropy)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    final_cost = costs[-1]
    plt.annotate(f'Final: {final_cost:.4f}', 
                xy=(len(costs)-1, final_cost),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Cost improvement
    plt.subplot(1, 2, 2)
    if len(costs) > 1:
        improvements = [-np.diff(costs)[i] for i in range(len(costs)-1)]
        plt.plot(improvements, 'g-', linewidth=2, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.title('Cost Improvement per Iteration', fontsize=14, fontweight='bold')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Cost Reduction', fontsize=12)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(CONFIG['plots_dir'], 'training_progress.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training plot saved: {save_path}")
    plt.close()


def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("QUANTUM-ENHANCED SMART IRRIGATION SYSTEM")
    print("="*60)
    print("Training Pipeline with Real Dataset")
    print("="*60)
    
    try:
        # Step 1: Load dataset
        X, y, feature_names = load_dataset(CONFIG['csv_file'])
        
        # Step 2: Split data
        print(f"\n{'='*60}")
        print("SPLITTING DATASET")
        print('='*60)
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=CONFIG['test_size'], random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=CONFIG['validation_size']/(1-CONFIG['test_size']), 
            random_state=42, stratify=y_temp
        )
        
        print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # Step 3: Train VQC
        vqc_model = QuantumIrrigationVQC(
            n_qubits=CONFIG['n_qubits'],
            n_layers=CONFIG['n_layers'],
            n_features=CONFIG['n_features'],
            learning_rate=CONFIG['learning_rate']
        )
        
        vqc_model.fit(X_train, y_train, X_val, y_val, verbose=True)
        
        # Step 4: Train classical models
        classical_models = train_classical_models(X_train, y_train)
        
        # Step 5: Evaluate all models
        results = evaluate_all_models(vqc_model, classical_models, X_test, y_test)
        
        # Step 6: Plot training curve
        plot_training_curve(vqc_model)
        
        # Step 7: Save all models
        save_all_models(vqc_model, classical_models)
        
        # Step 8: Final summary
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY")
        print('='*60)
        
        if vqc_model.training_costs:
            initial_cost = vqc_model.training_costs[0]
            final_cost = vqc_model.training_costs[-1]
            improvement = initial_cost - final_cost
            improvement_percent = (improvement / initial_cost) * 100 if initial_cost > 0 else 0
            
            print(f"\nVQC Training Statistics:")
            print(f"  Initial cost: {initial_cost:.6f}")
            print(f"  Final cost: {final_cost:.6f}")
            print(f"  Improvement: {improvement:.6f} ({improvement_percent:.2f}%)")
            print(f"  Training time: {vqc_model.training_time:.2f} seconds")
            print(f"  Total iterations: {len(vqc_model.training_costs)}")
        
        print(f"\n{'='*60}")
        print("✓ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print('='*60)
        print(f"All models saved in: {CONFIG['model_dir']}/")
        print(f"Plots saved in: {CONFIG['plots_dir']}/")
        
        return vqc_model, classical_models, results
        
    except FileNotFoundError as e:
        print(f"\n{'='*60}")
        print("❌ FILE NOT FOUND ERROR")
        print('='*60)
        print(f"Cannot find: {CONFIG['csv_file']}")
        print(f"\nRequired columns:")
        print(f"  Features: {CONFIG['feature_columns']}")
        print(f"  Label: {CONFIG['label_column']}")
        print(f"\nPlease check:")
        print("  1. File exists in the current directory")
        print("  2. File name is correct")
        print("  3. CSV has the required columns")
        return None, None, None
        
    except Exception as e:
        print(f"\n{'='*60}")
        print("❌ TRAINING ERROR")
        print('='*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    print("\nStarting training pipeline...")
    print("Checking dependencies...")
    
    try:
        import pennylane as qml
        print(f"✓ PennyLane version: {qml.__version__}")
    except ImportError:
        print("❌ PennyLane not found!")
        print("Install with: pip install pennylane")
        exit(1)
    
    try:
        import sklearn
        print(f"✓ scikit-learn version: {sklearn.__version__}")
    except ImportError:
        print("❌ scikit-learn not found!")
        print("Install with: pip install scikit-learn")
        exit(1)
    
    print("\nAll dependencies satisfied. Starting training...\n")
    
    # Run main training
    vqc_model, classical_models, results = main()
    
    if vqc_model is not None:
        print("\n" + "="*60)
        print("🎉 SUCCESS!")
        print("="*60)
        print("The quantum and classical models are trained and ready.")
        print("\nNext steps:")
        print("  1. Check the 'models/' folder for saved models")
        print("  2. Check the 'plots/' folder for training visualizations")
        print("  3. Use the models for irrigation predictions")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ TRAINING FAILED")
        print("="*60)
        print("Please check the error messages above and:")
        print("  1. Verify the CSV file exists")
        print("  2. Check column names match the configuration")
        print("  3. Ensure all dependencies are installed")
        print("="*60)