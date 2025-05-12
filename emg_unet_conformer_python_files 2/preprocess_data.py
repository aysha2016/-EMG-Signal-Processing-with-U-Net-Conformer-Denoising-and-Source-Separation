
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Define paths
data_path = '/content/drive/MyDrive/emg_data'  # Update if needed

# Define activities
activities = ['sitting', 'standing', 'walking']
emg_data, labels = [], []

# Load and preprocess data
for activity in activities:
    file_path = os.path.join(data_path, f'{activity}.csv')
    df = pd.read_csv(file_path)
    scaler = MinMaxScaler()
    norm_data = scaler.fit_transform(df.values)
    emg_data.append(norm_data)
    labels.append(np.full((norm_data.shape[0],), activity))

# Concatenate the data and labels
X = np.concatenate(emg_data, axis=0)
y = np.concatenate(labels, axis=0)
X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshaping for model input

print("Data Preprocessing Complete")
