"""
Client node for Federated Learning (represents a hospital)
Fetches global model, trains locally, submits encrypted updates
Supports FedProx (proximal term for handling heterogeneous data)
"""
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers  # FIXED: Added this import
from utils import (build_regression_model, serialize_weights, deserialize_weights,
                   encrypt_bytes, decrypt_bytes, get_weights_as_numpy, 
                   set_weights_from_numpy, make_synthetic_regression, calculate_model_accuracy)
import argparse

class ClientNode:
    def __init__(self, hospital_id, server_url="http://localhost:5000", 
                 input_dim=10, n_samples=500):
        """
        Initialize a client node (hospital)
        
        Args:
            hospital_id: Unique identifier for this hospital
            server_url: URL of the federated learning server
            input_dim: Number of features in the data
            n_samples: Number of samples in this hospital's dataset
        """
        self.hospital_id = hospital_id
        self.server_url = server_url
        self.input_dim = input_dim
        
        # Generate synthetic patient data for this hospital
        print(f"[{hospital_id}] Generating synthetic dataset ({n_samples} samples)...")
        X_train, X_test, y_train, y_test = make_synthetic_regression(
            n_samples=n_samples, n_features=input_dim, random_state=hash(hospital_id) % 1000
        )
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # Initialize local model
        self.model = build_regression_model(input_dim)
        self.global_weights = get_weights_as_numpy(self.model)
        
        print(f"[{hospital_id}] Initialized with {len(X_train)} training samples, {len(X_test)} test samples")

    def fetch_global(self):
        """Download global model from server"""
        try:
            print(f"[{self.hospital_id}] Fetching global model from server...")
            response = requests.get(f"{self.server_url}/get_global", timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Decrypt and deserialize weights
            enc = data["weights"].encode('latin1')
            plain = decrypt_bytes(enc)
            weights = deserialize_weights(plain)
            
            # Update local model
            self.global_weights = weights
            set_weights_from_numpy(self.model, weights)
            
            version = data.get("version", 0)
            print(f"[{self.hospital_id}] ✓ Downloaded global model (version {version})")
            return version
            
        except Exception as e:
            print(f"[{self.hospital_id}] ✗ Error fetching global model: {e}")
            return 0

    def local_train_and_submit(self, epochs=2, batch_size=32, lr=1e-3, mu=0.0):
        """
        Train model on local data and submit update to server
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            mu: FedProx proximal term coefficient (0 = standard training)
        """
        print(f"[{self.hospital_id}] Starting local training...")
        print(f"  - Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {lr}")
        if mu > 0:
            print(f"  - Using FedProx with mu={mu}")
        
        # Setup optimizer and loss
        opt = optimizers.Adam(learning_rate=lr)
        loss_fn = tf.keras.losses.MeanSquaredError()
        
        # Convert global weights to tensors for FedProx
        gw = [tf.convert_to_tensor(w) for w in self.global_weights]
        
        # Create dataset
        ds = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        ds = ds.shuffle(1000).batch(batch_size)
        
        # Training loop
        for epoch in range(epochs):
            epoch_losses = []
            for xb, yb in ds:
                with tf.GradientTape() as tape:
                    preds = self.model(xb, training=True)
                    loss = loss_fn(yb, preds)
                    
                    # FedProx: Add proximal term to keep model close to global
                    if mu > 0.0:
                        prox = 0.0
                        for w_var, w_glob in zip(self.model.trainable_weights, 
                                                gw[:len(self.model.trainable_weights)]):
                            prox += tf.reduce_sum(tf.square(w_var - tf.cast(w_glob, tf.float32)))
                        loss = loss + (mu / 2.0) * prox
                    
                    epoch_losses.append(float(loss))
                
                grads = tape.gradient(loss, self.model.trainable_weights)
                opt.apply_gradients(zip(grads, self.model.trainable_weights))
            
            avg_loss = np.mean(epoch_losses)
            print(f"  Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
        
        # Evaluate on test set
        test_loss, test_mae = calculate_model_accuracy(self.model, self.X_test, self.y_test)
        print(f"[{self.hospital_id}] Training complete - Test Loss: {test_loss:.4f}, MAE: {test_mae:.4f}")
        
        # Serialize and encrypt model weights
        weights = get_weights_as_numpy(self.model)
        ser = serialize_weights(weights)
        enc = encrypt_bytes(ser)
        
        # Submit to server
        payload = {
            "hospital_id": self.hospital_id,
            "weights": enc.decode('latin1'),
            "size": len(self.X_train)
        }
        
        try:
            print(f"[{self.hospital_id}] Submitting encrypted update to server...")
            response = requests.post(f"{self.server_url}/submit", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            print(f"[{self.hospital_id}] ✓ Update submitted successfully: {result}")
            return result
        except Exception as e:
            print(f"[{self.hospital_id}] ✗ Error submitting update: {e}")
            return {"error": str(e)}

def run_federated_round(hospital_id, epochs=2, mu=0.0, server_url="http://localhost:5000"):
    """Helper function to run one complete federated learning round"""
    print("\n" + "=" * 60)
    print(f"HOSPITAL: {hospital_id}")
    print("=" * 60)
    
    node = ClientNode(hospital_id, server_url=server_url)
    version = node.fetch_global()
    result = node.local_train_and_submit(epochs=epochs, mu=mu)
    
    print("=" * 60)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Client Node")
    parser.add_argument("--id", default="hospital_1", help="Hospital ID")
    parser.add_argument("--server", default="http://localhost:5000", help="Server URL")
    parser.add_argument("--epochs", type=int, default=2, help="Local training epochs")
    parser.add_argument("--mu", type=float, default=0.0, help="FedProx mu parameter")
    parser.add_argument("--samples", type=int, default=500, help="Number of training samples")
    
    args = parser.parse_args()
    
    run_federated_round(
        hospital_id=args.id,
        epochs=args.epochs,
        mu=args.mu,
        server_url=args.server
    )