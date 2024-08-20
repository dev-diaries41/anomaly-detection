import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self, in_features, hidden1, hidden2):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, in_features)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(model, filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file '{filepath}' doesn't exist. Create it by running train_model.py.")
    
    # Load the saved model state dictionary into the model
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set the model to evaluation mode
    