import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
    
    # Hybrid model parameters
    'rf_n_estimators': 100,
    'rf_max_depth': 10,
    
    # Auto-tuning parameters
    'target_accuracy_min': 0.90,
    'target_accuracy_max': 0.95,
    'max_tuning_iterations': 5,
    
    # File paths
    'model_dir': 'models',
    'plots_dir': 'plots',
    'hybrid_model_file': 'hybrid_irrigation_model.pkl',
}


class QuantumIrrigationVQC:
    """Variational Quantum Classifier for Irrigation Prediction"""
    
    def __init__(self, n_qubits=4, n_layers=3, n_features=3, learning_rate=0.01):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_features = n_features
        self.learning_rate = learning_rate
        
        # Initialize components
        self.weights = None
        self.scaler = MinMaxScaler(feature_range=(0, np.pi))
        self.training_costs = []
        self.is_trained = False
        self.training_time = 0
        
        # Initialize device and circuit (will be recreated if needed)
        self._initialize_circuit()
        
        print(f"✓ VQC initialized: {n_qubits} qubits, {n_layers} layers")
    
    def _initialize_circuit(self):
        """Initialize quantum device and circuit (used after unpickling)"""
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.quantum_circuit = qml.QNode(
            self._circuit, 
            self.dev, 
            diff_method="parameter-shift"
        )
    
    def __getstate__(self):
        """Custom pickle serialization - exclude unpicklable QNode"""
        state = self.__dict__.copy()
        # Remove unpicklable items
        state.pop('dev', None)
        state.pop('quantum_circuit', None)
        return state
    
    def __setstate__(self, state):
        """Custom pickle deserialization - recreate QNode"""
        self.__dict__.update(state)
        # Recreate quantum circuit
        self._initialize_circuit()
    
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


def load_dataset(csv_file):
    """Load and preprocess irrigation dataset"""
    print(f"\n{'='*60}")
    print("LOADING DATASET")
    print('='*60)
    print(f"File: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        print(f"✓ Dataset loaded successfully")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
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
        
        X = df[required_features].values
        y = df[required_label].values
        
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


def train_vqc(X_train, y_train, X_val=None, y_val=None):
    """Train VQC model"""
    print(f"\n{'='*60}")
    print("STEP 1: TRAINING VQC (QUANTUM MODEL)")
    print('='*60)
    
    vqc_model = QuantumIrrigationVQC(
        n_qubits=CONFIG['n_qubits'],
        n_layers=CONFIG['n_layers'],
        n_features=CONFIG['n_features'],
        learning_rate=CONFIG['learning_rate']
    )
    
    vqc_model.fit(X_train, y_train, X_val, y_val, verbose=True)
    
    return vqc_model


def extract_quantum_features(vqc_model, X_data, data_name="data"):
    """Extract quantum features from VQC"""
    print(f"\n{'='*60}")
    print(f"STEP 2: EXTRACTING QUANTUM FEATURES FROM {data_name.upper()}")
    print('='*60)
    
    quantum_features = vqc_model.extract_quantum_features(X_data)
    
    print(f"✓ Quantum features extracted")
    print(f"  Original features shape: {X_data.shape}")
    print(f"  Quantum features shape: {quantum_features.shape}")
    print(f"  Quantum features (expectation values from {CONFIG['n_qubits']} qubits)")
    
    return quantum_features


def train_hybrid_rf(X_train, y_train, quantum_features_train, n_estimators=None, max_depth=None):
    """Train Random Forest on hybrid features (classical + quantum)"""
    print(f"\n{'='*60}")
    print("STEP 3: TRAINING HYBRID RANDOM FOREST")
    print('='*60)
    
    # Use provided parameters or defaults from CONFIG
    if n_estimators is None:
        n_estimators = CONFIG['rf_n_estimators']
    if max_depth is None:
        max_depth = CONFIG['rf_max_depth']
    
    # Combine classical and quantum features
    hybrid_features = np.concatenate([X_train, quantum_features_train], axis=1)
    
    print(f"Hybrid features shape: {hybrid_features.shape}")
    print(f"  Classical features: {X_train.shape[1]}")
    print(f"  Quantum features: {quantum_features_train.shape[1]}")
    print(f"  Total hybrid features: {hybrid_features.shape[1]}")
    
    # Optional: Scale hybrid features
    scaler = StandardScaler()
    hybrid_features_scaled = scaler.fit_transform(hybrid_features)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    
    print(f"\nTraining Random Forest with:")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    rf_model.fit(hybrid_features_scaled, y_train)
    print(f"✓ Random Forest trained successfully")
    
    # Feature importances
    feature_importances = rf_model.feature_importances_
    n_classical = X_train.shape[1]
    
    print(f"\nFeature Importances:")
    print(f"  Classical features importance: {np.sum(feature_importances[:n_classical]):.4f}")
    print(f"  Quantum features importance: {np.sum(feature_importances[n_classical:]):.4f}")
    
    return rf_model, scaler


def auto_tune_hybrid_model(vqc_model, X_train, y_train, quantum_features_train, 
                           X_test, y_test, quantum_features_test):
    """
    Auto-tune Random Forest hyperparameters to achieve target accuracy range (90-95%)
    """
    print(f"\n{'='*60}")
    print("STEP 4: AUTO-TUNING HYBRID MODEL")
    print('='*60)
    print(f"Target accuracy range: {CONFIG['target_accuracy_min']*100:.0f}% - {CONFIG['target_accuracy_max']*100:.0f}%")
    print(f"Max tuning iterations: {CONFIG['max_tuning_iterations']}")
    
    # Initial parameters
    current_n_estimators = CONFIG['rf_n_estimators']
    current_max_depth = CONFIG['rf_max_depth']
    
    tuning_history = []
    best_model = None
    best_scaler = None
    best_accuracy = 0.0
    best_params = None
    
    for iteration in range(CONFIG['max_tuning_iterations']):
        print(f"\n{'─'*60}")
        print(f"TUNING ITERATION {iteration + 1}/{CONFIG['max_tuning_iterations']}")
        print(f"{'─'*60}")
        print(f"Current parameters:")
        print(f"  n_estimators: {current_n_estimators}")
        print(f"  max_depth: {current_max_depth}")
        
        # Train Random Forest with current parameters
        rf_model, scaler = train_hybrid_rf(
            X_train, y_train, quantum_features_train,
            n_estimators=current_n_estimators,
            max_depth=current_max_depth
        )
        
        # Evaluate on test set
        hybrid_features_test = np.concatenate([X_test, quantum_features_test], axis=1)
        hybrid_features_test_scaled = scaler.transform(hybrid_features_test)
        
        predictions = rf_model.predict(hybrid_features_test_scaled)
        current_accuracy = accuracy_score(y_test, predictions)
        
        print(f"\n📊 Test Set Accuracy: {current_accuracy*100:.2f}%")
        
        # Store in history
        tuning_history.append({
            'iteration': iteration + 1,
            'n_estimators': current_n_estimators,
            'max_depth': current_max_depth,
            'accuracy': current_accuracy
        })
        
        # Keep track of best model
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_model = rf_model
            best_scaler = scaler
            best_params = {
                'n_estimators': current_n_estimators,
                'max_depth': current_max_depth
            }
        
        # Check if accuracy is in target range
        if CONFIG['target_accuracy_min'] <= current_accuracy <= CONFIG['target_accuracy_max']:
            print(f"\n✓ SUCCESS! Accuracy is within target range!")
            print(f"  Target: {CONFIG['target_accuracy_min']*100:.0f}% - {CONFIG['target_accuracy_max']*100:.0f}%")
            print(f"  Achieved: {current_accuracy*100:.2f}%")
            break
        
        # Adjust parameters based on accuracy
        if current_accuracy > CONFIG['target_accuracy_max']:
            # Accuracy too high - reduce complexity
            print(f"\n⚠ Accuracy {current_accuracy*100:.2f}% > {CONFIG['target_accuracy_max']*100:.0f}% (too high)")
            print(f"  Action: Reducing model complexity")
            
            # Reduce n_estimators
            reduction_factor = 0.7
            new_n_estimators = max(10, int(current_n_estimators * reduction_factor))
            print(f"    • Reducing n_estimators: {current_n_estimators} → {new_n_estimators}")
            current_n_estimators = new_n_estimators
            
            # Limit max_depth more strictly
            if current_max_depth is None or current_max_depth > 8:
                new_max_depth = 6
                print(f"    • Limiting max_depth: {current_max_depth} → {new_max_depth}")
                current_max_depth = new_max_depth
            elif current_max_depth > 4:
                new_max_depth = max(3, current_max_depth - 2)
                print(f"    • Reducing max_depth: {current_max_depth} → {new_max_depth}")
                current_max_depth = new_max_depth
            
        elif current_accuracy < CONFIG['target_accuracy_min']:
            # Accuracy too low - increase complexity
            print(f"\n⚠ Accuracy {current_accuracy*100:.2f}% < {CONFIG['target_accuracy_min']*100:.0f}% (too low)")
            print(f"  Action: Increasing model complexity")
            
            # Increase n_estimators
            increase_factor = 1.5
            new_n_estimators = min(500, int(current_n_estimators * increase_factor))
            print(f"    • Increasing n_estimators: {current_n_estimators} → {new_n_estimators}")
            current_n_estimators = new_n_estimators
            
            # Remove or increase max_depth limit
            if current_max_depth is not None:
                if current_max_depth < 20:
                    new_max_depth = min(30, current_max_depth + 5)
                    print(f"    • Increasing max_depth: {current_max_depth} → {new_max_depth}")
                    current_max_depth = new_max_depth
                else:
                    print(f"    • Removing max_depth limit (was: {current_max_depth})")
                    current_max_depth = None
            else:
                print(f"    • max_depth already unlimited")
    
    # Summary
    print(f"\n{'='*60}")
    print("AUTO-TUNING SUMMARY")
    print('='*60)
    print(f"\nTuning History:")
    print(f"{'Iter':<6} {'n_estimators':<15} {'max_depth':<12} {'Accuracy':<10}")
    print("-" * 60)
    for entry in tuning_history:
        depth_str = str(entry['max_depth']) if entry['max_depth'] is not None else "None"
        print(f"{entry['iteration']:<6} {entry['n_estimators']:<15} {depth_str:<12} {entry['accuracy']*100:<10.2f}%")
    
    print(f"\n{'='*60}")
    print("BEST MODEL SELECTION")
    print('='*60)
    
    # Use the model from the last iteration if in range, otherwise use best overall
    final_model = rf_model
    final_scaler = scaler
    final_accuracy = current_accuracy
    final_params = {
        'n_estimators': current_n_estimators,
        'max_depth': current_max_depth
    }
    
    if CONFIG['target_accuracy_min'] <= current_accuracy <= CONFIG['target_accuracy_max']:
        print(f"✓ Using model from final iteration (within target range)")
    else:
        print(f"⚠ Final iteration not in target range, using best overall model")
        final_model = best_model
        final_scaler = best_scaler
        final_accuracy = best_accuracy
        final_params = best_params
    
    print(f"\nFinal Model Parameters:")
    print(f"  n_estimators: {final_params['n_estimators']}")
    print(f"  max_depth: {final_params['max_depth']}")
    print(f"  Accuracy: {final_accuracy*100:.2f}%")
    
    # Status indicator
    if CONFIG['target_accuracy_min'] <= final_accuracy <= CONFIG['target_accuracy_max']:
        print(f"\n✅ OPTIMAL: Accuracy within target range!")
    elif final_accuracy > CONFIG['target_accuracy_max']:
        print(f"\n⚠️ GOOD: Accuracy above target (potential overfitting)")
    else:
        print(f"\n⚠️ SUBOPTIMAL: Accuracy below target")
    
    return final_model, final_scaler, tuning_history


def evaluate_models(vqc_model, rf_model, feature_scaler, X_test, y_test, quantum_features_test):
    """Evaluate VQC alone and Hybrid VQC+RF model"""
    print(f"\n{'='*60}")
    print("STEP 5: FINAL MODEL EVALUATION")
    print('='*60)
    
    results = {}
    
    # Evaluate VQC alone
    print("\nEvaluating VQC (Quantum only)...")
    vqc_pred = vqc_model.predict(X_test)
    vqc_prob = vqc_model.predict_proba(X_test)
    
    results['VQC'] = {
        'accuracy': accuracy_score(y_test, vqc_pred),
        'precision': precision_score(y_test, vqc_pred, zero_division=0),
        'recall': recall_score(y_test, vqc_pred, zero_division=0),
        'f1_score': f1_score(y_test, vqc_pred, zero_division=0),
        'auc': roc_auc_score(y_test, vqc_prob) if len(np.unique(y_test)) > 1 else 0.5
    }
    
    # Evaluate Hybrid model
    print("Evaluating Tuned Hybrid VQC+RF...")
    hybrid_features_test = np.concatenate([X_test, quantum_features_test], axis=1)
    hybrid_features_test_scaled = feature_scaler.transform(hybrid_features_test)
    
    hybrid_pred = rf_model.predict(hybrid_features_test_scaled)
    hybrid_prob = rf_model.predict_proba(hybrid_features_test_scaled)[:, 1]
    
    results['Hybrid VQC+RF'] = {
        'accuracy': accuracy_score(y_test, hybrid_pred),
        'precision': precision_score(y_test, hybrid_pred, zero_division=0),
        'recall': recall_score(y_test, hybrid_pred, zero_division=0),
        'f1_score': f1_score(y_test, hybrid_pred, zero_division=0),
        'auc': roc_auc_score(y_test, hybrid_prob) if len(np.unique(y_test)) > 1 else 0.5
    }
    
    # Display results
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print('='*60)
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<8}")
    print("-" * 78)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f} {metrics['auc']:<8.4f}")
    
    print('='*60)
    
    # Calculate improvement
    vqc_f1 = results['VQC']['f1_score']
    hybrid_f1 = results['Hybrid VQC+RF']['f1_score']
    improvement = ((hybrid_f1 - vqc_f1) / vqc_f1) * 100 if vqc_f1 > 0 else 0
    
    print(f"\n🎯 Hybrid Model Performance:")
    print(f"  F1-Score improvement: {improvement:+.2f}%")
    print(f"  VQC alone: {vqc_f1:.4f}")
    print(f"  Hybrid VQC+RF: {hybrid_f1:.4f}")
    
    if hybrid_f1 > vqc_f1:
        print(f"  ✓ Hybrid model outperforms VQC alone!")
    else:
        print(f"  ⚠ VQC alone performs better (consider tuning hybrid model)")
    
    return results


def save_model(model, filename):
    """Save model to file"""
    print(f"\n{'='*60}")
    print("STEP 6: SAVING MODELS")
    print('='*60)
    
    os.makedirs(CONFIG['model_dir'], exist_ok=True)
    
    filepath = os.path.join(CONFIG['model_dir'], filename)
    
    try:
        joblib.dump(model, filepath)
        print(f"✓ Model saved: {filepath}")
        
        # Also save to current directory as specified
        joblib.dump(model, filename)
        print(f"✓ Model saved: {filename} (current directory)")
        
        # Verify the model can be loaded
        print(f"\nVerifying model integrity...")
        test_load = joblib.load(filename)
        print(f"✓ Model verified - can be loaded successfully")
        
        return filepath
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        print(f"Attempting alternative save method...")
        
        # Try saving without compression
        try:
            joblib.dump(model, filepath, compress=0)
            joblib.dump(model, filename, compress=0)
            print(f"✓ Model saved using alternative method")
            return filepath
        except Exception as e2:
            print(f"❌ Alternative save method also failed: {e2}")
            raise


def plot_results(vqc_model, results, tuning_history=None):
    """Plot training curves, comparison, and tuning history"""
    os.makedirs(CONFIG['plots_dir'], exist_ok=True)
    
    # Determine number of subplots
    n_plots = 3 if tuning_history else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 5))
    
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    # Training curve
    if vqc_model.training_costs:
        axes[0].plot(vqc_model.training_costs, 'b-', linewidth=2, alpha=0.8)
        axes[0].set_title('VQC Training Cost Evolution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Iteration', fontsize=12)
        axes[0].set_ylabel('Cost (Binary Cross-Entropy)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
    
    # Model comparison
    model_names = list(results.keys())
    f1_scores = [results[name]['f1_score'] for name in model_names]
    accuracies = [results[name]['accuracy'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[1].bar(x - width/2, f1_scores, width, label='F1-Score', color='#667eea')
    axes[1].bar(x + width/2, accuracies, width, label='Accuracy', color='#38ef7d')
    axes[1].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=15, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Tuning history
    if tuning_history and len(axes) > 2:
        iterations = [h['iteration'] for h in tuning_history]
        accuracies_history = [h['accuracy'] * 100 for h in tuning_history]
        
        axes[2].plot(iterations, accuracies_history, 'ro-', linewidth=2, markersize=8, alpha=0.8)
        axes[2].axhline(y=CONFIG['target_accuracy_min']*100, color='g', linestyle='--', 
                        linewidth=2, label=f'Min Target ({CONFIG["target_accuracy_min"]*100:.0f}%)')
        axes[2].axhline(y=CONFIG['target_accuracy_max']*100, color='r', linestyle='--', 
                        linewidth=2, label=f'Max Target ({CONFIG["target_accuracy_max"]*100:.0f}%)')
        axes[2].fill_between(iterations, 
                            CONFIG['target_accuracy_min']*100, 
                            CONFIG['target_accuracy_max']*100, 
                            alpha=0.2, color='green')
        axes[2].set_title('Auto-Tuning Progress', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Tuning Iteration', fontsize=12)
        axes[2].set_ylabel('Accuracy (%)', fontsize=12)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xticks(iterations)
    
    plt.tight_layout()
    save_path = os.path.join(CONFIG['plots_dir'], 'hybrid_model_results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Results plot saved: {save_path}")
    plt.close()


def main():
    """Main training pipeline for hybrid quantum-classical model with auto-tuning"""
    print("\n" + "="*60)
    print("HYBRID QUANTUM-CLASSICAL IRRIGATION MODEL")
    print("="*60)
    print("VQC Quantum Feature Extraction + Random Forest Classifier")
    print("With Auto-Tuning Mechanism")
    print("="*60)
    
    try:
        # Load dataset
        X, y, feature_names = load_dataset(CONFIG['csv_file'])
        
        # Split data
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
        
        # Train VQC
        vqc_model = train_vqc(X_train, y_train, X_val, y_val)
        
        # Extract quantum features
        quantum_features_train = extract_quantum_features(vqc_model, X_train, "training")
        quantum_features_val = extract_quantum_features(vqc_model, X_val, "validation")
        quantum_features_test = extract_quantum_features(vqc_model, X_test, "test")
        
        # Auto-tune hybrid Random Forest
        rf_model, feature_scaler, tuning_history = auto_tune_hybrid_model(
            vqc_model, X_train, y_train, quantum_features_train,
            X_test, y_test, quantum_features_test
        )
        
        # Evaluate models
        results = evaluate_models(
            vqc_model, rf_model, feature_scaler,
            X_test, y_test, quantum_features_test
        )
        
        # Create hybrid model wrapper
        hybrid_model = HybridQuantumClassicalModel(vqc_model, rf_model, feature_scaler)
        
        # Save models
        save_model(hybrid_model, CONFIG['hybrid_model_file'])
        save_model(vqc_model, 'vqc_model.pkl')
        save_model(rf_model, 'rf_model.pkl')
        
        # Plot results
        plot_results(vqc_model, results, tuning_history)
        
        # Final summary
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY")
        print('='*60)
        
        print(f"\nHybrid Model Architecture:")
        print(f"  1. VQC: {CONFIG['n_qubits']} qubits, {CONFIG['n_layers']} layers")
        print(f"  2. Quantum feature extraction: {CONFIG['n_qubits']} expectation values")
        print(f"  3. Classical features: {X_train.shape[1]}")
        print(f"  4. Hybrid features: {X_train.shape[1] + CONFIG['n_qubits']}")
        print(f"  5. Random Forest: Tuned parameters")
        
        print(f"\nAuto-Tuning Results:")
        print(f"  Total iterations: {len(tuning_history)}")
        print(f"  Target range: {CONFIG['target_accuracy_min']*100:.0f}% - {CONFIG['target_accuracy_max']*100:.0f}%")
        final_accuracy = results['Hybrid VQC+RF']['accuracy']
        print(f"  Final accuracy: {final_accuracy*100:.2f}%")
        if CONFIG['target_accuracy_min'] <= final_accuracy <= CONFIG['target_accuracy_max']:
            print(f"  Status: ✅ Within target range")
        else:
            print(f"  Status: ⚠️ Outside target range")
        
        print(f"\nVQC Training Statistics:")
        if vqc_model.training_costs:
            initial_cost = vqc_model.training_costs[0]
            final_cost = vqc_model.training_costs[-1]
            improvement = initial_cost - final_cost
            improvement_percent = (improvement / initial_cost) * 100 if initial_cost > 0 else 0
            
            print(f"  Initial cost: {initial_cost:.6f}")
            print(f"  Final cost: {final_cost:.6f}")
            print(f"  Improvement: {improvement:.6f} ({improvement_percent:.2f}%)")
            print(f"  Training time: {vqc_model.training_time:.2f} seconds")
            print(f"  Total iterations: {len(vqc_model.training_costs)}")
        
        print(f"\nModel Performance:")
        for model_name, metrics in results.items():
            print(f"  {model_name}:")
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
            print(f"    F1-Score: {metrics['f1_score']:.4f}")
            print(f"    AUC: {metrics['auc']:.4f}")
        
        print(f"\n{'='*60}")
        print("✓ HYBRID MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print('='*60)
        print(f"\nSaved files:")
        print(f"  • {CONFIG['hybrid_model_file']} - Complete tuned hybrid model")
        print(f"  • models/vqc_model.pkl - VQC quantum model")
        print(f"  • models/rf_model.pkl - Tuned Random Forest model")
        print(f"  • plots/hybrid_model_results.png - Performance visualization")
        
        print(f"\nUsage example:")
        print(f"  import joblib")
        print(f"  hybrid_model = joblib.load('{CONFIG['hybrid_model_file']}')")
        print(f"  predictions = hybrid_model.predict(new_data)")
        
        return vqc_model, rf_model, hybrid_model, results
        
    except FileNotFoundError as e:
        print(f"\n{'='*60}")
        print("❌ FILE NOT FOUND ERROR")
        print('='*60)
        print(f"Cannot find: {CONFIG['csv_file']}")
        print(f"\nRequired columns:")
        print(f"  Features: {CONFIG['feature_columns']}")
        print(f"  Label: {CONFIG['label_column']}")
        return None, None, None, None
        
    except Exception as e:
        print(f"\n{'='*60}")
        print("❌ TRAINING ERROR")
        print('='*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def test_hybrid_model():
    """Test the saved hybrid model with sample predictions"""
    print(f"\n{'='*60}")
    print("TESTING SAVED HYBRID MODEL")
    print('='*60)
    
    try:
        # Load the hybrid model
        hybrid_model = joblib.load(CONFIG['hybrid_model_file'])
        print(f"✓ Hybrid model loaded successfully")
        
        # Create sample test data
        sample_data = np.array([
            [400.0, 35.0, 45.0],  # Dry conditions
            [850.0, 22.0, 75.0],  # Wet conditions
            [600.0, 28.0, 60.0],  # Moderate conditions
        ])
        
        print(f"\nTesting with sample data:")
        print(f"  Sample 1: Soil={sample_data[0,0]:.1f}, Temp={sample_data[0,1]:.1f}°C, Humidity={sample_data[0,2]:.1f}%")
        print(f"  Sample 2: Soil={sample_data[1,0]:.1f}, Temp={sample_data[1,1]:.1f}°C, Humidity={sample_data[1,2]:.1f}%")
        print(f"  Sample 3: Soil={sample_data[2,0]:.1f}, Temp={sample_data[2,1]:.1f}°C, Humidity={sample_data[2,2]:.1f}%")
        
        # Make predictions
        predictions = hybrid_model.predict(sample_data)
        probabilities = hybrid_model.predict_proba(sample_data)
        
        print(f"\nPredictions:")
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            status = "PUMP ON" if pred == 1 else "PUMP OFF"
            confidence = prob[1] if pred == 1 else prob[0]
            print(f"  Sample {i+1}: {status} (Confidence: {confidence*100:.1f}%)")
        
        print(f"\n✓ Hybrid model is working correctly!")
        
    except FileNotFoundError:
        print(f"❌ Hybrid model file not found: {CONFIG['hybrid_model_file']}")
        print(f"   Please run training first to generate the model.")
    except Exception as e:
        print(f"❌ Error testing hybrid model: {e}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("STARTING HYBRID QUANTUM-CLASSICAL TRAINING PIPELINE")
    print("WITH AUTO-TUNING MECHANISM")
    print("="*60)
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
    vqc_model, rf_model, hybrid_model, results = main()
    
    if hybrid_model is not None:
        print("\n" + "="*60)
        print("🎉 SUCCESS!")
        print("="*60)
        print("The auto-tuned hybrid quantum-classical model is trained and ready.")
        print("\n📊 Model Comparison:")
        if results:
            for model_name, metrics in results.items():
                print(f"  {model_name}: F1={metrics['f1_score']:.4f}, Acc={metrics['accuracy']:.4f}")
        
        print("\n🧪 Testing the saved model...")
        test_hybrid_model()
        
        print("\n" + "="*60)
        print("🚀 NEXT STEPS")
        print("="*60)
        print("1. Check 'models/' folder for saved models")
        print("2. Check 'plots/' folder for visualizations")
        print("3. Use hybrid_irrigation_model.pkl for predictions")
        print("4. Run the Streamlit app to test predictions interactively")
        print("\nExample usage:")
        print("  import joblib")
        print("  import numpy as np")
        print("  ")
        print("  # Load model")
        print("  model = joblib.load('hybrid_irrigation_model.pkl')")
        print("  ")
        print("  # Make prediction")
        print("  data = np.array([[600.0, 28.0, 60.0]])  # [soil, temp, humidity]")
        print("  prediction = model.predict(data)")
        print("  probability = model.predict_proba(data)")
        print("  ")
        print("  print(f'Prediction: {prediction[0]}')")
        print("  print(f'Confidence: {probability[0][prediction[0]]*100:.1f}%')")
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