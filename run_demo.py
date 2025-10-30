"""
Complete demonstration of Federated Learning System
Runs server and multiple clients to simulate federated training
"""
import multiprocessing
import time
import requests
import subprocess
import sys

def start_server():
    """Start Flask server in subprocess"""
    print("\n" + "=" * 70)
    print("STARTING FEDERATED LEARNING SERVER")
    print("=" * 70)
    subprocess.run([sys.executable, "server_flask.py"])

def wait_for_server(url="http://localhost:5000/status", max_attempts=10):
    """Wait for server to be ready"""
    print("\nWaiting for server to start...", end="")
    for i in range(max_attempts):
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(" ✓ Server is ready!")
                return True
        except:
            pass
        print(".", end="", flush=True)
        time.sleep(1)
    print(" ✗ Server failed to start")
    return False

def configure_server(algo="fedavg", agg_every=2, lr=0.001):
    """Configure server algorithm"""
    try:
        response = requests.post(
            "http://localhost:5000/set_algo",
            json={"algo": algo, "agg_every": agg_every, "lr": lr},
            timeout=5
        )
        if response.status_code == 200:
            print(f"✓ Server configured: {algo}, aggregate every {agg_every} clients")
            return True
    except Exception as e:
        print(f"✗ Failed to configure server: {e}")
    return False

def run_client(hospital_id, epochs=3, mu=0.0):
    """Run a single client"""
    subprocess.run([
        sys.executable, "client_flask.py",
        "--id", hospital_id,
        "--epochs", str(epochs),
        "--mu", str(mu)
    ])

def run_federated_learning_demo():
    """Run complete federated learning demo with multiple clients"""
    print("\n" + "=" * 70)
    print("FEDERATED LEARNING DEMO")
    print("=" * 70)
    print("\nThis demo simulates 4 hospitals training a model collaboratively")
    print("without sharing their raw data!\n")
    
    # Configuration
    hospitals = ["Hospital_A", "Hospital_B", "Hospital_C", "Hospital_D"]
    rounds = 2  # Number of federated learning rounds
    epochs_per_round = 3  # Local training epochs
    
    print(f"Configuration:")
    print(f"  - Hospitals: {len(hospitals)}")
    print(f"  - Rounds: {rounds}")
    print(f"  - Epochs per round: {epochs_per_round}")
    print(f"  - Algorithm: FedAvg")
    
    # Configure server for FedAvg
    if not configure_server(algo="fedavg", agg_every=len(hospitals)):
        return
    
    # Run federated learning rounds
    for round_num in range(1, rounds + 1):
        print("\n" + "=" * 70)
        print(f"FEDERATED ROUND {round_num}/{rounds}")
        print("=" * 70)
        
        # All hospitals train in parallel
        processes = []
        for hospital in hospitals:
            p = multiprocessing.Process(
                target=run_client,
                args=(hospital, epochs_per_round, 0.0)
            )
            p.start()
            processes.append(p)
        
        # Wait for all hospitals to finish
        for p in processes:
            p.join()
        
        print(f"\n✓ Round {round_num} complete!")
        time.sleep(2)  # Wait for aggregation
    
    print("\n" + "=" * 70)
    print("FEDERATED LEARNING DEMO COMPLETED")
    print("=" * 70)
    print("✓ All hospitals trained collaboratively without sharing data!")

def run_fedprox_demo():
    """Run FedProx demo (helps with heterogeneous data)"""
    print("\n" + "=" * 70)
    print("FEDPROX DEMO (with Proximal Term)")
    print("=" * 70)
    print("\nFedProx adds a proximal term to handle heterogeneous data")
    print("mu > 0 keeps local models closer to global model\n")
    
    hospitals = ["Hospital_X", "Hospital_Y"]
    
    # Configure for FedAvg with 2 clients
    configure_server(algo="fedavg", agg_every=2)
    
    print("\nRunning clients with FedProx (mu=0.01)...")
    processes = []
    for hospital in hospitals:
        p = multiprocessing.Process(
            target=run_client,
            args=(hospital, 3, 0.01)  # mu=0.01 for FedProx
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    print("\n✓ FedProx demo completed!")

def run_fedopt_demo():
    """Run FedOpt demo (adaptive server-side optimization)"""
    print("\n" + "=" * 70)
    print("FEDOPT DEMO (Adaptive Server Optimization)")
    print("=" * 70)
    print("\nFedOpt uses Adam optimizer on the server side")
    print("Better convergence for non-uniform data\n")
    
    hospitals = ["Hospital_P", "Hospital_Q"]
    
    # Configure for FedOpt
    configure_server(algo="fedopt", agg_every=2, lr=0.001)
    
    print("\nRunning clients with FedOpt...")
    processes = []
    for hospital in hospitals:
        p = multiprocessing.Process(
            target=run_client,
            args=(hospital, 3, 0.0)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    print("\n✓ FedOpt demo completed!")

def run_gtv_simulation():
    """Run decentralized GTV simulation"""
    print("\n" + "=" * 70)
    print("RUNNING GTV (GRAPH TOTAL VARIATION) SIMULATION")
    print("=" * 70)
    print("\nDecentralized approach: clients share only with neighbors")
    print("No central server needed!\n")
    
    subprocess.run([sys.executable, "gtv_simulation.py"])
    
    print("\n✓ GTV simulation completed!")

def main():
    """Main demo orchestrator"""
    print("\n" + "=" * 70)
    print("FEDERATED LEARNING SYSTEM - COMPLETE DEMONSTRATION")
    print("=" * 70)
    print("\nChoose a demo to run:")
    print("  1. Standard FedAvg Demo (4 hospitals, 2 rounds)")
    print("  2. FedProx Demo (handles heterogeneous data)")
    print("  3. FedOpt Demo (adaptive server optimization)")
    print("  4. GTV Simulation (decentralized, no server)")
    print("  5. Run ALL demos sequentially")
    print("  6. Quick test (2 hospitals, 1 round)")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    # Start server in background (except for GTV)
    if choice != "4":
        server_process = multiprocessing.Process(target=start_server, daemon=True)
        server_process.start()
        
        if not wait_for_server():
            print("Failed to start server. Exiting.")
            return
        
        time.sleep(1)
    
    try:
        if choice == "1":
            run_federated_learning_demo()
        elif choice == "2":
            run_fedprox_demo()
        elif choice == "3":
            run_fedopt_demo()
        elif choice == "4":
            run_gtv_simulation()
        elif choice == "5":
            print("\nRunning all demos...\n")
            run_federated_learning_demo()
            time.sleep(2)
            run_fedprox_demo()
            time.sleep(2)
            run_fedopt_demo()
            time.sleep(2)
            
            # Terminate server for GTV
            if 'server_process' in locals():
                server_process.terminate()
            
            run_gtv_simulation()
        elif choice == "6":
            print("\nQuick test...")
            configure_server(algo="fedavg", agg_every=2)
            processes = []
            for h in ["Test_A", "Test_B"]:
                p = multiprocessing.Process(target=run_client, args=(h, 2, 0.0))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            print("\n✓ Quick test completed!")
        else:
            print("Invalid choice!")
            return
        
        print("\n" + "=" * 70)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    finally:
        # Cleanup
        if 'server_process' in locals() and choice != "4":
            print("\nShutting down server...")
            server_process.terminate()
            server_process.join(timeout=2)

if __name__ == "__main__":
    main()