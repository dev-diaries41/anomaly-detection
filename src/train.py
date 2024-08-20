import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from data import preprocess, split_dataset, plot_training_loss, plot_reconstruction_errors
from model import Autoencoder, save_model
import joblib
import numpy as np

DATA_PATH = "data/synthetic_data_500.csv"
MODEL_PATH = 'model/autoencoder_model.pth'
SCALER_PATH = 'model/scaler.pkl'

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Preprocess the data
data_normalized, scaler = preprocess(df)

# Split the dataset into training and test sets
X_train, X_test = split_dataset(data_normalized)

# Define model parameters
in_features = X_train.shape[1]  # Number of input features
hidden1 = 32
hidden2 = 16

# Instantiate model
model = Autoencoder(in_features, hidden1, hidden2)

# Check if GPU is available and move model to GPU if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
X_train, X_test = X_train.to(device), X_test.to(device)

learning_rate = 0.0025
num_epochs = 100
criterion = nn.MSELoss()  # Use MSE loss for autoencoder
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Optimizer

def train(model, criterion, optimizer, X_train, X_test, num_epochs):
    train_loss_values = []
    test_loss_values = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        train_loss = criterion(outputs, X_train)  # Reconstruction loss
        train_loss.backward()
        optimizer.step()
        train_loss_values.append(train_loss.item())

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, X_test)  # Reconstruction loss
            test_loss_values.append(test_loss.item())

        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

    return train_loss_values, test_loss_values

# Train the model
train_loss_values, test_loss_values = train(model, criterion, optimizer, X_train, X_test, num_epochs)

# Save the model and scaler
save_model(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

# Calculate and save the threshold
model.eval()
with torch.no_grad():
    train_outputs = model(X_train)
    train_errors = torch.mean((X_train - train_outputs) ** 2, dim=1).cpu().numpy()

# Choose a percentile for the threshold. 95th is chosen as an example. 
# For most cases it will detect outliers but in prod use validation set to fine tune with f1 score
threshold = np.percentile(train_errors, 95)
print(f"Determined anomaly threshold: {threshold}")

# Save the threshold for future use
np.save('model/anomaly_threshold.npy', threshold)

# save plots
plot_training_loss(train_loss_values, test_loss_values)
plot_reconstruction_errors(train_errors, threshold)