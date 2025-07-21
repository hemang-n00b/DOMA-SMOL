import socket
import json
import sys
import threading
import time
import numpy as np
import matplotlib.pyplot as plt

from evaluate_response import evaluate_with_gpt4
from cabspotting_simulation import CabspottingUserFactory,TDriveUserFactory  # Your simulation module

# --- Router Configuration ---
ROUTER_IP = '127.0.0.1'  # Update as needed
ROUTER_PORT = 9090

# --- Loading Requests for Three Functions ---
def load_requests():
    """
    Loads requests and ideal responses from a comma-separated list of files.
    Each file corresponds to one type:
      - 'medical': uses 'merged_input' and 'output'
      - 'mental': uses 'input' and 'output'
      - 'fitness': uses 'instruction' and 'completion'
    Returns:
      requests_dict: dict mapping type -> list of requests
      responses_dict: dict mapping type -> list of ideal responses
    """
    # Expect files as comma-separated string, e.g., "medical.jsonl,mental.jsonl,fitness.jsonl"
    files = sys.argv[1].split(',')
    requests_dict = {"medical": [], "mental": [], "fitness": []}
    responses_dict = {"medical": [], "mental": [], "fitness": []}
    
    for f in files:
        f = f.strip()
        with open(f, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line.strip())
                if "fitness" in f.lower():
                    requests_dict["fitness"].append(data["instruction"])
                    responses_dict["fitness"].append(data["completion"])
                elif "mental" in f.lower():
                    requests_dict["mental"].append(data["input"])
                    responses_dict["mental"].append(data["output"])
                elif "medical" in f.lower():
                    requests_dict["medical"].append(data["merged_input"])
                    responses_dict["medical"].append(data["output"])
    return requests_dict, responses_dict

# --- Sending a Request ---
def send_request(message, ideal_response):
    """
    Sends a request to the router and handles response evaluation.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((ROUTER_IP, ROUTER_PORT))
            client_socket.sendall(message.encode('utf-8'))
            
            # Receive response from the router
            response = client_socket.recv(1024).decode('utf-8')
            print(f"Received response: {response}")
            
            evaluation_score = evaluate_with_gpt4(message, response, ideal_response)
            print(f"Evaluated response with GPT-4: {evaluation_score}")
            
            # Send evaluation score back to the router
            client_socket.sendall(str(evaluation_score).encode('utf-8'))
            
            # Log interaction in a combined JSONL file
            log_entry = {
                "request": message,
                "response": response,
                "ideal_response": ideal_response,
                "evaluation_score": evaluation_score
            }
            with open('results_combined.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            return response, evaluation_score
    except Exception as e:
        print(f"Error communicating with the router: {e}")
        return "Error", "Error"

# --- Cabspotting Simulation Setup ---
# Define node coordinates (adjust as needed)
node_coordinates = np.array([
    [0.25, 0.25],
])

# Define a function distribution for three request types: medical, mental, fitness.
# For example, here we use probabilities [0.3, 0.4, 0.3]
functions = np.array([0.3, 0.4, 0.3])
functions = functions / functions.sum()

# Path to the Cabspotting dataset (adjust the path as needed)
DATASET_DIR = "."
user_factory = TDriveUserFactory(DATASET_DIR, node_coordinates, functions)

# Define simulation timestamps (e.g., 100 evenly spaced time steps between 0 and 1)
SIMULATION_TIMESTAMPS = np.linspace(0, 1, num=100)

# A lock to safely access shared request lists
requests_lock = threading.Lock()

# A global list to record workload data: each entry is (timestamp, medical, mental, fitness)
workload_records = []
workload_records_lock = threading.Lock()

# --- Simulation Worker ---
def simulation_worker(client_id, timestamps, requests_dict, responses_dict):
    """
    At each simulation timestamp, get the workload from the Cabspotting simulation.
    The workload is a matrix with shape [n_nodes, 3], where each column corresponds to:
      0: medical, 1: mental, 2: fitness.
    For each function type, send as many requests as indicated by the workload.
    Also record the computed workload per timestamp.
    """
    # Map function index to type key
    function_mapping = {0: "medical", 1: "mental", 2: "fitness"}
    
    for t in timestamps:
        # Get workload matrix at simulation time t
        workload = user_factory.get_user_workload(t)  # shape: (n_nodes, 3)
        # Sum over nodes to get total requests per function type
        workload_sum = np.sum(workload, axis=0)
        ##random permutation
        np.random.shuffle(workload_sum)
        print(f"[Client {client_id}] Time {t:.2f} - Workload: {workload_sum}")
        
        # Record the workload for visualization
        with workload_records_lock:
            workload_records.append((t, workload_sum[0], workload_sum[1], workload_sum[2]))
        
        # For each function type, send as many requests as indicated by the workload.
        for func_idx in range(3):
            req_type = function_mapping[func_idx]
            num_requests = int(workload_sum[func_idx])
            for _ in range(num_requests):
                with requests_lock:
                    if not requests_dict[req_type]:
                        print(f"[Client {client_id}] No more {req_type} requests available.")
                        continue
                    request = requests_dict[req_type].pop(0)
                    ideal_response = responses_dict[req_type].pop(0)
                # Construct a message that includes metadata about the type and timestamp.
                message_data = {
                    "type": req_type,
                    "timestamp": float(t),
                    "payload": request
                }
                message = json.dumps(message_data)
                print(f"[Client {client_id}] Sending {req_type} request: {message}")
                # Uncomment the following line if you want to actually send the request
                # response, score = send_request(message, ideal_response)
        
        # Wait a short interval (1 second) to simulate time passing
        time.sleep(1)

# --- Visualization Function ---
def visualize_workload(workload_records):
    """
    Sorts the recorded workload by simulation time and plots the number of requests per type.
    """
    # Convert workload_records (list of tuples) to a numpy array and sort by timestamp (index 0)
    print("Visualizing workload over time...")
    data = np.array(workload_records)
    data = data[data[:, 0].argsort()]
    times = data[:, 0]
    medical = data[:, 1]
    mental = data[:, 2]
    fitness = data[:, 3]
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, medical, label="Medical", marker='o')
    plt.plot(times, mental, label="Mental", marker='x')
    plt.plot(times, fitness, label="Fitness", marker='s')
    plt.xlabel("Simulation Time")
    plt.ylabel("Number of Requests")
    plt.title("User Requests Over Time")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig("workload_over_time_tdrive.png")

# --- Main Execution ---
if __name__ == "__main__":
    # Number of client threads provided as a command-line argument
    num_threads = int(sys.argv[2])
    
    requests_dict, responses_dict = load_requests()
    print("Loaded requests:")
    for key in requests_dict:
        print(f"  {key}: {len(requests_dict[key])} requests")
    
    # Divide simulation timestamps among threads
    timestamps_per_thread = np.array_split(SIMULATION_TIMESTAMPS, num_threads)
    
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(
            target=simulation_worker, 
            args=(i, timestamps_per_thread[i], requests_dict, responses_dict)
        )
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    print("All client threads have completed simulation.")
    
    # Visualize the workload over time after simulation is done
    visualize_workload(workload_records)
