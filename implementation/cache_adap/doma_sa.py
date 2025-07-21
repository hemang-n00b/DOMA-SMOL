import socket
import random
import threading
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import time 
import sys
import os
from collections import deque

from model_decider import model_decider , load_decider

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

registered_nodes = {}
torch._inductor.config.triton.cudagraphs = False
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
# latency_values = {}
# confidence_values = {}
embedding_model = load_decider()
embedding_model.to("cuda")
# The port on which the router listens for incoming user requests
ROUTER_PORT = 8081
# The port for node registration of SMOL nodes
REGISTRATION_PORT = 8080

OUTPUT_PATH = sys.argv[1]

latency_values = {}
confidence_values = {}
energy_values = {}

# Handles node registration of a single SMOL node and adds it to the distionary of registered nodes
def register_node(client_socket):
    global latency_values
    global confidence_values
    global energy_values
    data = client_socket.recv(1024).decode('utf-8')
    ip, port, model_name, description = data.split('|')
    port = int(port)
    latency_values[model_name] = deque(maxlen=100)
    confidence_values[model_name] = deque(maxlen=100)
    confidence_values[model_name].append(5.0)  # Initialize with a default confidence value
    energy_values[model_name] = deque(maxlen=100)
    # Store with (ip, port, model_name) as key
    registered_nodes[(ip, port, model_name)] = {"model_name": model_name, "description": description}
    print(f"Registered node: {ip}:{port} - {model_name}")
    client_socket.sendall("Registration successful".encode('utf-8'))
    client_socket.close()

# Starts the server to accept node registrations.
def start_registration_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reg_socket:
        reg_socket.bind(('0.0.0.0', REGISTRATION_PORT))
        reg_socket.listen(10)
        print(f"Registration server listening on port {REGISTRATION_PORT}...")

        while True:
            client_socket, addr = reg_socket.accept()
            threading.Thread(target=register_node, args=(client_socket,)).start()
         
def compute_similarities(request_text, registered_nodes, embedding_model):
    """Compute similarities between request and all model descriptions"""
    model_descriptions = {f"{model}": v["description"]  # Just use model name as key
                         for (ip, port, model), v in registered_nodes.items()}
    model_names = list(model_descriptions.keys())
    descriptions = list(model_descriptions.values())
    
    # Compute embeddings
    request_embedding = embedding_model.encode(request_text, convert_to_tensor=True)
    desc_embeddings = embedding_model.encode(descriptions, convert_to_tensor=True)
    
    # Compute cosine similarities
    similarities = util.cos_sim(request_embedding, desc_embeddings)[0]
    
    return {model_names[i]: float(similarities[i]) for i in range(len(model_names))}

def get_user_request(client_socket):
    global latency_values
    global confidence_values
    global energy_values
    try:
        request = client_socket.recv(4096).decode('utf-8').strip()
        if not request:
            return None
            
        # Compute similarities at router
        similarities = compute_similarities(request, registered_nodes, embedding_model)
        
        # Safely calculate metrics with proper type conversion
        model_metrics = {}
        for model in latency_values:
            try:
                avg_latency = float(np.mean(latency_values[model])) if latency_values[model] else float('inf')
                avg_confidence = float(np.mean(confidence_values[model])) if confidence_values[model] else 0.0
                avg_energy = float(np.mean(energy_values[model])) if energy_values[model] else 0.0
                
                model_metrics[model] = {
                    "avg_latency": avg_latency,
                    "avg_confidence": avg_confidence,
                    "avg_energy": avg_energy
                }
            except Exception as e:
                print(f"Error calculating metrics for {model}: {e}")
                model_metrics[model] = {
                    "avg_latency": float('inf'),
                    "avg_confidence": 0.0,
                    "avg_energy": 0.0
                }
        
        return {
            "request": request,
            "similarities": similarities,
            "model_metrics": model_metrics
        }
        
    except Exception as e:
        print(f"Error in get_user_request: {e}")
        return None

def route_request(user_request):
    # Get any available node
    matching_nodes = list(registered_nodes.items())
    if not matching_nodes:
        raise ValueError("No available nodes to handle request")
    
    node_key, node_info = random.choice(matching_nodes)
    ip, port, model_name = node_key
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as target_socket:
        target_socket.connect((ip, port))
        
        # Prepare payload with string keys
        payload = {
            "request": user_request["request"],
            "similarities": user_request.get("similarities", {}),
            "confidence":user_request.get("confidence",{}),
            "model_metrics": user_request.get("model_metrics", {}),
            "node_info": {
                "ip": ip,
                "port": port,
                "model_name": model_name
            }
        }
        
        # Send as JSON
        latency_start = time.time()
        target_socket.sendall(json.dumps(payload).encode('utf-8'))

        # Receive response
        response_data = target_socket.recv(4096).decode('utf-8')
        model_name, energy_consumption, response = response_data.split('|', 2)
        latency = time.time() - latency_start
        
    return response, latency, float(energy_consumption), model_name

# handles each request
def handle_client(client_socket):
    global latency_values
    global confidence_values
    global energy_values
    with client_socket:
        user_request = get_user_request(client_socket)
        if user_request is None:
            client_socket.sendall("An error occurred. Please try again later.".encode('utf-8'))
            return

        response,latency,energy_consumption,model_name = route_request(user_request)
        client_socket.sendall(response.encode('utf-8'))
        response_data = client_socket.recv(5000).decode('utf-8')
        ideal_response , evaluation_score = response_data.split('|')
        
        log_entry = {
            "request": user_request['request'],
            "response": response,
            "ideal_response": ideal_response,
            "latency": latency,
            "confidence_score": evaluation_score,
            "energy_consumption": float(energy_consumption),
            "model": model_name
        }

        latency_values[model_name].append(float(latency))
        energy_values[model_name].append(float(energy_consumption))
        confidence_values[model_name].append(float(evaluation_score))
        
        results_file = f"{OUTPUT_PATH}"
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
        
# Starts the router to handle incoming user requests
def start_router():
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as router_socket:
        router_socket.bind(('0.0.0.0', ROUTER_PORT))
        router_socket.listen(20)
        print(f"Router is listening for requests on port {ROUTER_PORT}...")
        
        while True:
            client_socket, addr = router_socket.accept()
            print(f"Connection from {addr}")
            thread=threading.Thread(target=handle_client, args=(client_socket,)).start()
        
        

if __name__ == "__main__":       
    threading.Thread(target=start_registration_server, daemon=True).start()
    start_router()