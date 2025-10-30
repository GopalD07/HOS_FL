"""
Utility functions for Federated Learning system
FIXED: Keras 3 compatible
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pickle
from cryptography.fernet import Fernet

# Fixed encryption key (in production, use secure key management)
FERNET_KEY = b'wk5MHjHYjj7Cm8Xn5VXnYkBkJI_nN3MKa57UNPjW4SE='
FERNET = Fernet(FERNET_KEY)

def make_synthetic_regression(n_samples=1000, n_features=10, noise=0.1, random_state=None):
    """Generate synthetic regression dataset"""
    X, y = make_regression(n_samples=n_samples, n_features=n_features, 
                          noise=noise, random_state=random_state)
    y = (y - y.min()) / (y.max() - y.min() + 1e-12)
    return train_test_split(X.astype(np.float32), y.astype(np.float32), 
                           test_size=0.2, random_state=random_state)

def build_regression_model(input_dim, hidden=[64, 32]):
    """Build neural network - FIXED for Keras 3"""
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for h in hidden:
        model.add(layers.Dense(h, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))
    
    # FIXED: Use Adam optimizer without legacy
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), 
                 loss=losses.MeanSquaredError(),
                 metrics=['mae'])
    return model

def get_weights_as_numpy(model):
    """Extract model weights as numpy arrays"""
    return [w.copy() for w in model.get_weights()]

def set_weights_from_numpy(model, weights):
    """Set model weights from numpy arrays"""
    model.set_weights([np.array(w) for w in weights])

def serialize_weights(weights):
    """Convert weights to bytes for transmission"""
    return pickle.dumps(weights)

def deserialize_weights(b):
    """Convert bytes back to weights"""
    return pickle.loads(b)

def encrypt_bytes(b: bytes) -> bytes:
    """Encrypt data using Fernet symmetric encryption"""
    return FERNET.encrypt(b)

def decrypt_bytes(b: bytes) -> bytes:
    """Decrypt data using Fernet symmetric encryption"""
    return FERNET.decrypt(b)

def calculate_model_accuracy(model, X_test, y_test):
    """Calculate test loss and MAE for model evaluation"""
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    return loss, mae