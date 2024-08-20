import numpy as np
import pandas as pd
import os

# Parameters for the dataset
num_normal_points = 450
num_anomalies = 50
num_features = 3  # Number of features
normal_mean = 0.0
normal_std = 1.0
anomaly_mean = 10.0
anomaly_std = 1.0

DATA_PATH = "data/synthetic_data_500.csv"
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

try:
    # Generate normal data
    normal_data = np.random.normal(loc=normal_mean, scale=normal_std, size=(num_normal_points, num_features))

    # Generate anomalous data
    anomalous_data = np.random.normal(loc=anomaly_mean, scale=anomaly_std, size=(num_anomalies, num_features))

    # Combine data
    data = np.vstack((normal_data, anomalous_data))

    # Create DataFrame without normalization
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(num_features)])

    # Save to CSV
    df.to_csv(DATA_PATH, index=False)
    
    print(f"Synthetic dataset with {num_normal_points + num_anomalies} data points created and saved to '{DATA_PATH}'")

except Exception as e:
    print(f"An error occurred: {e}")
