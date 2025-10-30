"""
Graph Total Variation (GTV) Simulation
Decentralized federated learning where clients share with neighbors only
Uses graph smoothness constraint to regularize model differences
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers  # FIXED: Added this import
from utils import (make_synthetic_regression, build_regression_model, 
                   get_weights_as_numpy, set_weights_from_numpy)

class GTVSimulator:
    def __init__(self, n_clients=6, input_dim=10, samples_per_client=400, 
                 lam=0.5, neighbors=None, local_steps=1, lr=1e-3):
        """
        Initialize GTV simulation
        
        Args:
            n_clients: Number of hospitals/clients
            input_dim: Number of features
            samples_per_client: Dataset size per client
            lam: Graph regularization parameter (higher = more smoothing)
            neighbors: Dict mapping client_id -> list of neighbor ids
            local_steps: Number of local gradient steps per round
            lr: Learning rate
        """
        self.n_clients = n_clients
        self.input_dim = input_dim
        self.lam = lam
        self.local_steps = local_steps
        self.lr = lr
        
        # Generate separate datasets for each client
        print(f"Initializing {n_clients} clients with separate datasets...")
        self.clients_data = []
        for i in range(n_clients):
            X_train, X_test, y_train, y_test = make_synthetic_regression(
                n_samples=samples_per_client, 
                n_features=input_dim, 
                random_state=42 + i
            )
            self.clients_data.append((X_train, X_test, y_train, y_test))
        
        # Initialize models for each client
        self.local_models = [build_regression_model(input_dim) for _ in range(n_clients)]
        
        # Initialize all models with same weights for fair comparison
        base_weights = get_weights_as_numpy(self.local_models[0])
        for model in self.local_models:
            set_weights_from_numpy(model, base_weights)
        
        # Define neighbor graph (default: ring topology)
        if neighbors is None:
            # Ring topology: each client connected to 2 neighbors
            self.neighbors = {
                i: [(i - 1) % n_clients, (i + 1) % n_clients] 
                for i in range(n_clients)
            }
        else:
            self.neighbors = neighbors
        
        print(f"Configuration:")
        print(f"  - Clients: {n_clients}")
        print(f"  - Features: {input_dim}")
        print(f"  - Samples per client: {samples_per_client}")
        print(f"  - Lambda (graph regularization): {lam}")
        print(f"  - Local steps: {local_steps}")
        print(f"  - Learning rate: {lr}")
        print(f"  - Topology: {self._topology_name()}")

    def _topology_name(self):
        """Get human-readable topology name"""
        n_neighbors = len(self.neighbors[0])
        if n_neighbors == 2:
            return "Ring"
        elif n_neighbors == self.n_clients - 1:
            return "Fully Connected"
        else:
            return f"Custom ({n_neighbors} neighbors per client)"

    def local_objective_step(self, client_idx):
        """
        Perform one gradient descent step for a client
        Objective: local_loss + lambda * ||w_i - avg(w_neighbors)||^2
        """
        # Get client data and model
        X_train, X_test, y_train, y_test = self.clients_data[client_idx]
        model = self.local_models[client_idx]
        
        # Get neighbor models and compute average weights
        neighbor_ids = self.neighbors[client_idx]
        neighbor_weights = [get_weights_as_numpy(self.local_models[n]) for n in neighbor_ids]
        
        # Average neighbor weights per parameter
        neighbor_avg = []
        for params in zip(*neighbor_weights):
            neighbor_avg.append(np.mean(params, axis=0))
        
        # Convert data to tensors
        Xb = tf.convert_to_tensor(X_train, dtype=tf.float32)
        yb = tf.convert_to_tensor(y_train, dtype=tf.float32)
        
        # Compute gradients and update
        with tf.GradientTape() as tape:
            preds = model(Xb, training=True)
            loss = tf.keras.losses.MeanSquaredError()(yb, preds)
            
            # Graph regularization: penalize difference from neighbors
            prox = 0.0
            for w_var, w_neigh in zip(model.trainable_weights, 
                                     neighbor_avg[:len(model.trainable_weights)]):
                diff = w_var - tf.convert_to_tensor(w_neigh, dtype=tf.float32)
                prox += tf.reduce_sum(tf.square(diff))
            
            total_loss = loss + (self.lam / 2.0) * prox
        
        grads = tape.gradient(total_loss, model.trainable_weights)
        
        # Apply gradients
        opt = optimizers.SGD(learning_rate=self.lr)
        opt.apply_gradients(zip(grads, model.trainable_weights))

    def run(self, rounds=10):
        """
        Run GTV simulation for specified number of rounds
        
        Args:
            rounds: Number of communication rounds
        """
        print("\n" + "=" * 60)
        print("STARTING GTV SIMULATION")
        print("=" * 60)
        
        for r in range(rounds):
            # Each client performs local_steps gradient updates
            for i in range(self.n_clients):
                for _ in range(self.local_steps):
                    self.local_objective_step(i)
            
            # Evaluate all clients on test data
            losses = []
            maes = []
            for i, model in enumerate(self.local_models):
                X_test = self.clients_data[i][1]
                y_test = self.clients_data[i][3]
                loss, mae = model.evaluate(X_test, y_test, verbose=0)
                losses.append(loss)
                maes.append(mae)
            
            # Print statistics
            avg_loss = np.mean(losses)
            std_loss = np.std(losses)
            print(f"Round {r + 1}/{rounds}")
            print(f"  Avg Test Loss: {avg_loss:.4f} (±{std_loss:.4f})")
            print(f"  Avg MAE: {np.mean(maes):.4f}")
            print(f"  Per-client losses: {[f'{l:.4f}' for l in losses[:5]]}")
        
        print("=" * 60)
        print("GTV SIMULATION COMPLETED")
        print("=" * 60)
        print(f"Final avg loss: {np.mean(losses):.4f}")
        print(f"Final loss std: {np.std(losses):.4f}")
        
        return self.local_models

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GRAPH TOTAL VARIATION (GTV) FEDERATED LEARNING")
    print("Decentralized learning with neighbor communication")
    print("=" * 70)
    
    # Create and run simulation
    sim = GTVSimulator(
        n_clients=6,
        input_dim=10,
        samples_per_client=300,
        lam=0.2,  # Graph regularization strength
        local_steps=1,
        lr=1e-3
    )
    
    # Run for 5 rounds
    models = sim.run(rounds=5)
    
    print("\n✓ Simulation complete. Trained models saved in memory.")