import torch
import numpy as np
import joblib
from model import Autoencoder, load_model

model_file_path = "model/autoencoder_model.pth"
scaler_file_path = "model/scaler.pkl"
threshold_path = "model/anomaly_threshold.npy"
ANOMALY_THRESHOLD = np.load(threshold_path)

# Load the trained model and scaler once at the start
def initialize_model_and_scaler():
    in_features = 3 
    hidden1 = 32
    hidden2 = 16
    
    # Initialize and load the model
    model = Autoencoder(in_features, hidden1, hidden2)
    load_model(model, model_file_path)
    
    # Load the scaler
    try:
        scaler = joblib.load(scaler_file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Scaler file '{scaler_file_path}' does not exist. Ensure the scaler is saved during training.")
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    return model, scaler, device

# Initialize model, scaler, and device
model, scaler, device = initialize_model_and_scaler()

def predict_anomaly(input_array):
    # Ensure input_array is a numpy array
    input_array = np.array(input_array).reshape(1, -1)  # Reshape to match (1, num_features)
    
    # Normalize or standardize the input data using the same scaler
    input_normalized = scaler.transform(input_array)    
    input_tensor = torch.tensor(input_normalized, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # Get the reconstructed output from the model
        reconstructed = model(input_tensor)        
        reconstruction_error = torch.mean((input_tensor - reconstructed) ** 2).cpu().numpy()
        
    is_anomaly = reconstruction_error > ANOMALY_THRESHOLD
    return is_anomaly

# example usage:
input_data = [22, 90, 90]
is_anomaly = predict_anomaly(input_data)
print(f"Anomaly detection result for the input {input_data}: {is_anomaly}")
