# data_preprocessing.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def load_binary_iris():
    data = load_iris()
    X = data.data[data.target != 2]  # Take only Setosa vs Versicolor
    y = data.target[data.target != 2]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X[:, :2])  # use 2 features for 2 qubits

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test
