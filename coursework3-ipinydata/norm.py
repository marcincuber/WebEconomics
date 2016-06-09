import numpy as np
import pandas as pd

data_train_normalised = pd.read_csv('train_set_final.csv', delimiter=',', header=None, dtype=float)  # Load data into a numpy array
data_train_normalised = data_train_normalised.as_matrix()

data_test_normalised = pd.read_csv('test_set_final.csv', delimiter=',', header=None, dtype=float)  # Load data into a numpy array
data_test_normalised = data_test_normalised.as_matrix()

''' Normalise Training data '''
for i in range(data_train_normalised.shape[1]):  # Normalise data by subtracting feature means from each value
    feature_train = data_train_normalised[:, i]
    mean = np.mean(feature_train)  # Compute feature mean
    std = np.std(feature_train)  # Compute feature standard deviation
    feature_train = (feature_train - mean) / std  # Compute z-score based on computed mean and std
    data_train_normalised[:, i] = feature_train

''' Normalise Test data '''
for i in range(data_test_normalised.shape[1]):  # Normalise data by subtracting feature means from each value
    feature_test = data_test_normalised[:, i]
    mean = np.mean(feature_test)  # Compute feature mean
    std = np.std(feature_test)  # Compute feature standard deviation
    feature_test = (feature_test - mean) / std  # Compute z-score based on computed mean and std
    data_test_normalised[:, i] = feature_test

train_df = pd.DataFrame(data_train_normalised)
train_df.to_csv("train_norm.csv", index=False, header=False)

test_df = pd.DataFrame(data_test_normalised)
test_df.to_csv("test_norm.csv", index=False, header=False)