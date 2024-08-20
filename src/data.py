import os
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def preprocess(data, scaler=None):
    """
    Preprocess the data by applying normalization or standardization.

    Parameters:
    - data: The data to preprocess (DataFrame or numpy array).
    - scaler: An optional pre-fitted scaler. If None, a new StandardScaler will be fitted.

    Returns:
    - Preprocessed data as a numpy array.
    """
    if scaler is None:
        # Initialize and fit a new scaler
        scaler = StandardScaler()  # or MinMaxScaler()
        data_normalized = scaler.fit_transform(data)
    else:
        # Apply the pre-fitted scaler
        data_normalized = scaler.transform(data)
    
    return data_normalized, scaler

from sklearn.model_selection import train_test_split
import torch

from sklearn.model_selection import train_test_split
import torch

def split_dataset(data):
    # Extract features (assuming all columns are features)
    X = data
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    # Convert feature matrices to PyTorch tensors
    torch.manual_seed(42)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    
    return X_train, X_test


def plot_training_loss(train_loss_values, test_loss_values, save_path='results/training_loss_plot.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure()  # Create a new figure
    plt.plot(train_loss_values, label='Training Loss')
    plt.plot(test_loss_values, label='Test Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()  # Close the figure to free up memory

def plot_reconstruction_errors(train_errors, threshold, save_path='results/r-errors.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(train_errors, bins=50, alpha=0.75, label='Reconstruction Errors')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold (95th percentile): {threshold:.4f}')
    plt.title("Distribution of Reconstruction Errors and Anomaly Threshold")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()  #