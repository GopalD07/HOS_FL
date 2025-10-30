# Helper script to run: 1) start Flask server (in a thread) - demo, 2) run two client submissions, 3) run GTV simulation.
# Note: Running server in this script is only for demo/testing; in real use start server independently.
import threading, time, requests, os, sys, subprocess
from server_flask import app, server
import multiprocessing


def start_server():
    app.run(port=5000, debug=False, use_reloader=False)

def run_demo_clients():
    # Run two local client submissions using client_flask module via subprocesses
    import subprocess, sys
    cmds = [
        [sys.executable, "client_flask.py", "--id", "hosp_A", "--epochs", "2", "--mu", "0.0"],
        [sys.executable, "client_flask.py", "--id", "hosp_B", "--epochs", "2", "--mu", "0.01"]
    ]
    procs = []
    for c in cmds:
        p = subprocess.Popen(c)
        procs.append(p)
    for p in procs:
        p.wait()

def run_gtv():
    from gtv_simulation import GTVSimulator
    sim = GTVSimulator(n_clients=6, input_dim=10, samples_per_client=300, lam=0.2, local_steps=1, lr=1e-3)
    sim.run(rounds=5)

if __name__ == '__main__':
    # Start server in background process to avoid blocking
    srv = multiprocessing.Process(target=start_server, daemon=True)
    srv.start()
    time.sleep(2)
    print("Server started. Set server algo to fedopt for demo...")
    try:
        requests.post("http://localhost:5000/set_algo", json={"algo":"fedopt","agg_every":2,"lr":0.001}, timeout=5)
    except Exception as e:
        print("Warning: could not set algo via HTTP:", e)
    # Run demo clients
    run_demo_clients()
    # Run GTV simulation
    print("Running GTV simulation...")
    run_gtv()
    print("Demo complete. Killing server.")
    srv.terminate()
