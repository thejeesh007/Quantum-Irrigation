

import numpy as np
import joblib
import os
import sys
from typing import Tuple, Optional


class IrrigationPredictor:
    """Quantum-Enhanced Irrigation Prediction System"""
    
    def __init__(self, model_path: str = "quantum_irrigation_model.pkl", 
                 scaler_path: str = "scaler.pkl"):
        """
        Initialize the irrigation predictor
        
        Args:
            model_path (str): Path to the trained quantum model
            scaler_path (str): Path to the trained scaler
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        
    def load_models(self) -> bool:
        """
        Load the trained quantum model and scaler
        
        Returns:
            bool: True if models loaded successfully, False otherwise
        """
        try:
            # Check if model files exist
            if not os.path.exists(self.model_path):
                print(f"❌ Error: Model file '{self.model_path}' not found!")
                print("Please ensure you have trained and saved the quantum model first.")
                return False
                
            if not os.path.exists(self.scaler_path):
                print(f"❌ Error: Scaler file '{self.scaler_path}' not found!")
                print("Please ensure you have trained and saved the scaler first.")
                return False
            
            # Load model
            print("Loading quantum irrigation model...")
            self.model = joblib.load(self.model_path)
            print(f"✅ Quantum model loaded successfully from {self.model_path}")
            
            # Load scaler
            print("Loading feature scaler...")
            self.scaler = joblib.load(self.scaler_path)
            print(f"✅ Scaler loaded successfully from {self.scaler_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("Please check that the model files are valid and not corrupted.")
            return False
    
    def validate_input(self, soil_moisture: float, temperature: float, 
                      humidity: float) -> Tuple[bool, str]:
        """
        Validate user input values
        
        Args:
            soil_moisture (float): Soil moisture percentage
            temperature (float): Temperature in Celsius
            humidity (float): Humidity percentage
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        # Validate soil moisture (0-100%)
        if not (0 <= soil_moisture <= 100):
            return False, "Soil moisture must be between 0% and 100%"
        
        # Validate temperature (reasonable range for agriculture: -10°C to 50°C)
        if not (-10 <= temperature <= 50):
            return False, "Temperature must be between -10°C and 50°C"
        
        # Validate humidity (0-100%)
        if not (0 <= humidity <= 100):
            return False, "Humidity must be between 0% and 100%"
        
        return True, ""
    
    def get_user_input(self) -> Optional[Tuple[float, float, float]]:
        """
        Get and validate user input for environmental conditions
        
        Returns:
            Optional[Tuple[float, float, float]]: (soil_moisture, temperature, humidity) 
                                                 or None if input is invalid
        """
        print("\n" + "="*60)
        print("🌱 QUANTUM SMART IRRIGATION SYSTEM")
        print("="*60)
        print("Please enter the current environmental conditions:")
        
        try:
            # Get soil moisture
            soil_moisture = float(input("Soil Moisture (%): "))
            
            # Get temperature
            temperature = float(input("Temperature (°C): "))
            
            # Get humidity
            humidity = float(input("Humidity (%): "))
            
            # Validate inputs
            is_valid, error_message = self.validate_input(soil_moisture, temperature, humidity)
            
            if not is_valid:
                print(f"❌ Invalid input: {error_message}")
                return None
            
            return soil_moisture, temperature, humidity
            
        except ValueError:
            print("❌ Invalid input: Please enter numeric values only")
            return None
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return None
    
    def make_prediction(self, soil_moisture: float, temperature: float, 
                       humidity: float) -> Tuple[int, float]:
        """
        Make irrigation prediction using the quantum model
        
        Args:
            soil_moisture (float): Soil moisture percentage
            temperature (float): Temperature in Celsius
            humidity (float): Humidity percentage
            
        Returns:
            Tuple[int, float]: (prediction, probability)
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Prepare input data
        input_data = np.array([[soil_moisture, temperature, humidity]])
        
        # Scale the input using the loaded scaler
        try:
            scaled_input = self.scaler.transform(input_data)
        except Exception as e:
            print(f"❌ Error scaling input: {e}")
            raise
        
        # Make prediction
        try:
            prediction = self.model.predict(scaled_input)[0]
            probability = self.model.predict_proba(scaled_input)[0]
            
            return prediction, probability
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            raise
    
    def display_prediction(self, prediction: int, probability: float,
                          soil_moisture: float, temperature: float, humidity: float):
        """
        Display the irrigation prediction with formatted output
        
        Args:
            prediction (int): Binary prediction (0 or 1)
            probability (float): Prediction probability
            soil_moisture (float): Input soil moisture
            temperature (float): Input temperature
            humidity (float): Input humidity
        """
        print("\n" + "-"*60)
        print("📊 PREDICTION RESULTS")
        print("-"*60)
        
        # Display input summary
        print("Input Conditions:")
        print(f"  🌱 Soil Moisture: {soil_moisture:.1f}%")
        print(f"  🌡️  Temperature:   {temperature:.1f}°C")
        print(f"  💨 Humidity:      {humidity:.1f}%")
        print()
        
        # Display prediction
        if prediction == 1:
            print("🔮 Quantum Prediction: IRRIGATION NEEDED")
            print("💧 Irrigation Needed")
            recommendation = "Turn on irrigation system"
        else:
            print("🔮 Quantum Prediction: NO IRRIGATION REQUIRED")
            print("✅ No Irrigation Required")
            recommendation = "Keep irrigation system off"
        
        # Display probability and confidence
        confidence = max(probability, 1 - probability)
        print(f"📈 Irrigation Probability: {probability:.3f} ({probability*100:.1f}%)")
        print(f"🎯 Prediction Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
        print(f"💡 Recommendation: {recommendation}")
        
        # Add interpretation
        print("\n📋 Interpretation:")
        if confidence >= 0.8:
            print("🟢 High confidence - Reliable prediction")
        elif confidence >= 0.6:
            print("🟡 Moderate confidence - Consider additional factors")
        else:
            print("🟠 Low confidence - Manual assessment recommended")
    
    def run_interactive_mode(self):
        """Run the system in interactive mode"""
        print("Starting Quantum-Enhanced Smart Irrigation System...")
        
        # Load models
        if not self.load_models():
            print("Failed to load models. Exiting...")
            return False
        
        print(f"\n✅ System ready for irrigation predictions!")
        print("Enter environmental conditions to get irrigation recommendations.")
        
        while True:
            print(f"\n{'='*60}")
            
            # Get user input
            user_input = self.get_user_input()
            if user_input is None:
                continue
            
            soil_moisture, temperature, humidity = user_input
            
            try:
                # Make prediction
                print("\n🔄 Processing with quantum model...")
                prediction, probability = self.make_prediction(soil_moisture, temperature, humidity)
                
                # Display results
                self.display_prediction(prediction, probability, soil_moisture, temperature, humidity)
                
            except Exception as e:
                print(f"❌ Prediction failed: {e}")
                continue
            
            # Ask if user wants to continue
            print(f"\n{'='*60}")
            try:
                continue_choice = input("Make another prediction? (y/n): ").lower().strip()
                if continue_choice in ['n', 'no', 'quit', 'exit']:
                    break
            except KeyboardInterrupt:
                break
        
        print("\n👋 Thank you for using the Quantum Smart Irrigation System!")
        return True


def run_single_prediction(soil_moisture: float, temperature: float, 
                         humidity: float, model_path: str = "quantum_irrigation_model.pkl",
                         scaler_path: str = "scaler.pkl") -> Optional[Tuple[int, float]]:
    """
    Run a single prediction (useful for API or batch processing)
    
    Args:
        soil_moisture (float): Soil moisture percentage
        temperature (float): Temperature in Celsius
        humidity (float): Humidity percentage
        model_path (str): Path to the trained model
        scaler_path (str): Path to the trained scaler
        
    Returns:
        Optional[Tuple[int, float]]: (prediction, probability) or None if failed
    """
    predictor = IrrigationPredictor(model_path, scaler_path)
    
    if not predictor.load_models():
        return None
    
    # Validate input
    is_valid, error_message = predictor.validate_input(soil_moisture, temperature, humidity)
    if not is_valid:
        print(f"Invalid input: {error_message}")
        return None
    
    try:
        return predictor.make_prediction(soil_moisture, temperature, humidity)
    except Exception as e:
        print(f"Prediction failed: {e}")
        return None


def main():
    """Main function"""
    print("QUANTUM-ENHANCED SMART IRRIGATION SYSTEM")
    print("Prediction Interface v1.0")
    print("="*60)
    
    # Check command line arguments for batch mode
    if len(sys.argv) == 4:
        try:
            # Batch mode: python irrigation_predictor.py soil_moisture temperature humidity
            soil_moisture = float(sys.argv[1])
            temperature = float(sys.argv[2])
            humidity = float(sys.argv[3])
            
            print(f"Batch mode: Processing input ({soil_moisture}, {temperature}, {humidity})")
            
            result = run_single_prediction(soil_moisture, temperature, humidity)
            if result is not None:
                prediction, probability = result
                if prediction == 1:
                    print("💧 Irrigation Needed")
                else:
                    print("✅ No Irrigation Required")
                print(f"Probability: {probability:.3f}")
            else:
                print("❌ Prediction failed")
                sys.exit(1)
                
        except ValueError:
            print("❌ Error: Invalid command line arguments")
            print("Usage: python irrigation_predictor.py <soil_moisture> <temperature> <humidity>")
            sys.exit(1)
    else:
        # Interactive mode
        predictor = IrrigationPredictor()
        success = predictor.run_interactive_mode()
        
        if not success:
            print("❌ System encountered errors during execution")
            sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 System interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your model files and try again.")
        sys.exit(1)