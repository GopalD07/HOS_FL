"""
Flask server for Federated Learning
Supports FedAvg and FedOpt aggregation algorithms
Receives encrypted model updates from clients
"""
from flask import Flask, request, jsonify
import threading
import numpy as np
from utils import (build_regression_model, serialize_weights, deserialize_weights,
                   encrypt_bytes, decrypt_bytes, get_weights_as_numpy, set_weights_from_numpy)

app = Flask(__name__)

class FedServer:
    def __init__(self, input_dim=10, algo='fedavg', agg_every=2, lr=1e-3):
        """
        Initialize Federated Learning Server
        
        Args:
            input_dim: Number of input features
            algo: 'fedavg' or 'fedopt'
            agg_every: Aggregate after receiving this many updates
            lr: Learning rate for FedOpt
        """
        self.global_model = build_regression_model(input_dim)
        self.lock = threading.Lock()
        self.buffer = []  # Buffer for incoming updates
        self.algo = algo
        self.agg_every = agg_every
        
        # FedOpt (Adam) state variables
        self.m = None  # First moment
        self.v = None  # Second moment
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.t = 0  # Time step
        self.lr = lr
        
        self.version = 0
        print(f"[Server] Initialized with algorithm: {algo}, aggregate every: {agg_every}")

    def add_update(self, hospital_id, encrypted_bytes, size):
        """Add client update to buffer and trigger aggregation if ready"""
        with self.lock:
            self.buffer.append((hospital_id, encrypted_bytes, int(size)))
            print(f"[Server] Received update from {hospital_id} (size: {size}). Buffer: {len(self.buffer)}/{self.agg_every}")
            
            if len(self.buffer) >= self.agg_every:
                to_agg = self.buffer.copy()
                self.buffer.clear()
                # Run aggregation in separate thread to not block
                threading.Thread(target=self.aggregate, args=(to_agg,), daemon=True).start()

    def weighted_fedavg(self, client_weights, sizes):
        """
        FedAvg: Weighted average of client models
        Weight by dataset size: larger datasets have more influence
        """
        total = float(sum(sizes))
        avg = []
        
        for params in zip(*client_weights):
            acc = np.zeros_like(params[0])
            for p, n in zip(params, sizes):
                acc += (n / total) * p
            avg.append(acc)
        
        return avg

    def fedopt_aggregate(self, client_weights, sizes):
        """
        FedOpt: Apply adaptive optimization (Adam) on server
        Treats client updates as pseudo-gradients
        """
        gw = get_weights_as_numpy(self.global_model)
        total = float(sum(sizes))
        
        # Compute weighted average of deltas (client_weight - global_weight)
        avg_delta = [np.zeros_like(p) for p in gw]
        for w, n in zip(client_weights, sizes):
            for i, (wi, gwi) in enumerate(zip(w, gw)):
                avg_delta[i] += (n / total) * (wi - gwi)
        
        # Initialize Adam moments on first call
        if self.m is None:
            self.m = [np.zeros_like(p) for p in gw]
            self.v = [np.zeros_like(p) for p in gw]
        
        self.t += 1
        new_gw = []
        
        # Apply Adam update
        for i, g in enumerate(avg_delta):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Compute update
            update = (self.lr * m_hat) / (np.sqrt(v_hat) + self.eps)
            new_gw.append(gw[i] + update)
        
        return new_gw

    def aggregate(self, updates):
        """
        Aggregate client updates into new global model
        
        Args:
            updates: List of (hospital_id, encrypted_bytes, size)
        """
        client_weights = []
        sizes = []
        ids = []
        
        # Decrypt and deserialize all updates
        for hid, enc, size in updates:
            try:
                plain = decrypt_bytes(enc)
                weights = deserialize_weights(plain)
                client_weights.append(weights)
                sizes.append(size)
                ids.append(hid)
            except Exception as e:
                print(f"[Server] Error processing update from {hid}: {e}")
                continue
        
        if not client_weights:
            print("[Server] No valid updates to aggregate")
            return
        
        # Perform aggregation
        with self.lock:
            if self.algo == 'fedopt':
                new_weights = self.fedopt_aggregate(client_weights, sizes)
            else:
                new_weights = self.weighted_fedavg(client_weights, sizes)
            
            set_weights_from_numpy(self.global_model, new_weights)
            self.version += 1
            print(f"[Server] ✓ Aggregated {len(client_weights)} updates from {ids}")
            print(f"[Server] New global model version: {self.version} (algo={self.algo})")

# Global server instance
server = FedServer(input_dim=10, algo='fedavg', agg_every=2, lr=1e-3)

@app.route("/set_algo", methods=["POST"])
def set_algo():
    """Endpoint to change aggregation algorithm"""
    data = request.get_json() or {}
    algo = data.get("algo", "fedavg")
    agg_every = int(data.get("agg_every", 2))
    lr = float(data.get("lr", 1e-3))
    
    server.algo = algo
    server.agg_every = agg_every
    server.lr = lr
    
    print(f"[Server] Algorithm changed to: {algo}, agg_every={agg_every}, lr={lr}")
    return jsonify({"status": "ok", "algo": server.algo, "agg_every": server.agg_every})

@app.route("/get_global", methods=["GET"])
def get_global():
    """Endpoint for clients to download global model"""
    w = get_weights_as_numpy(server.global_model)
    ser = serialize_weights(w)
    enc = encrypt_bytes(ser)
    
    # Return encrypted weights as latin1 string (JSON-safe)
    return jsonify({"version": server.version, "weights": enc.decode('latin1')})

@app.route("/submit", methods=["POST"])
def submit():
    """Endpoint for clients to submit local updates"""
    data = request.get_json() or {}
    hid = data.get("hospital_id")
    weights_str = data.get("weights")
    size = int(data.get("size", 1))
    
    if hid is None or weights_str is None:
        return jsonify({"error": "missing fields"}), 400
    
    enc_bytes = weights_str.encode('latin1')
    server.add_update(hid, enc_bytes, size)
    
    return jsonify({"status": "queued", "hospital_id": hid})

@app.route("/status", methods=["GET"])
def status():
    """Endpoint to check server status"""
    return jsonify({
        "version": server.version,
        "algo": server.algo,
        "buffer_size": len(server.buffer),
        "agg_every": server.agg_every
    })

if __name__ == "__main__":
    print("=" * 60)
    print("FEDERATED LEARNING SERVER")
    print("=" * 60)
    print("Starting server on http://localhost:5000")
    print(f"Algorithm: {server.algo}")
    print(f"Aggregation threshold: {server.agg_every} clients")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)