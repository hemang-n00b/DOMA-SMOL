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
# os.environ["TORCH_COMPILE_DISABLE"] = "1"

# Dictionary to store registered nodes
# Format: { (ip, port): {"model_name": str, "description": str} }
registered_nodes = {}
torch._inductor.config.triton.cudagraphs = False

torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
latency_values = {}
confidence_values = {}

model_latencies = {}
model_confidences = {}
model_gpu_usage = {}

model_max_latency = {}
model_min_latency = {}

model_max_confidence = {}
model_min_confidence = {}

# The port on which the router listens for incoming user requests
ROUTER_PORT = 9090

# The port for node registration
REGISTRATION_PORT = 9091

def register_node(client_socket):
    """Handles node registration."""
    try:
        data = client_socket.recv(1024).decode('utf-8')
        ip, port, model_name, description = data.split(';')
        port = int(port)
        registered_nodes[(ip, port, model_name)] = {"model_name": model_name, "description": description}
        ##initalize min and max latency and confidence values
        model_max_latency[model_name] = 0.0
        model_min_latency[model_name] = 0.0
        model_max_confidence[model_name] = 0.0
        model_min_confidence[model_name] = 0.0
        print(f"Registered node: {ip}:{port} - {model_name} ({description})")
        client_socket.sendall("Registration successful".encode('utf-8'))
    except Exception as e:
        print(f"Error during registration: {e}")
        client_socket.sendall("Registration failed".encode('utf-8'))
    finally:
        client_socket.close()

def blackbox_model_decider(request):
    """Determines the best matching model based on cosine similarity."""
    
    if not registered_nodes:
        return "default_model"
    
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    with open("descriptions.json", "r", encoding="utf-8") as file:
        model_descriptions = json.load(file)
        
    model_keys = list(model_descriptions.keys())
    model_embeddings = embedding_model.encode(list(model_descriptions.values()), convert_to_tensor=True)
    user_embedding = embedding_model.encode(request, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(user_embedding, model_embeddings)[0]
    best_model_idx = np.argmax(similarities.cpu().numpy())
    best_model = model_keys[best_model_idx]

    return best_model
    
    
    # Randomly select a model from the registered nodes
    # print("Selecting a random model...")
    # print(registered_nodes)
    # random_node = random.choice(list(registered_nodes.values()))
    # return random_node["model_name"]

def get_user_request(client_socket):
    """Receives a user request from the client and decides the model."""
    try:
        request = client_socket.recv(1024).decode('utf-8')
        model_string = blackbox_model_decider(request)
        # print(f"Model string: {model_string}")
        return {"request": request, "model": model_string}
    except Exception as e:
        print(f"Error receiving request: {e}")
        return None

def route_request(user_request):
    """Routes the request to a random registered node that matches the model and gets the response."""
    try:
        matching_nodes = [(ip, port, info) for (ip, port, model_name), info in registered_nodes.items() if info["model_name"] == user_request["model"]]
        
        if not matching_nodes:
            return "No matching nodes available."

        # Pick a random matching node
        ip, port, node_info = random.choice(matching_nodes)
        print(f"Routing to {ip}:{port} ({node_info['model_name']})")

        # Connect to the selected node
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as target_socket:
            target_socket.connect((ip, port))
            user_msg = f"{user_request['model']}|{user_request['request']}"
            latency_start = time.time()
            target_socket.sendall(user_msg.encode('utf-8'))

            # Wait for response from the target
            response = target_socket.recv(1024).decode('utf-8')
            latency_end = time.time()
            latency = latency_end - latency_start
            if latency_values.get(node_info['model_name']) and len(latency_values[node_info['model_name']]) < 10:
                latency_values[node_info['model_name']].append(latency)
            elif latency_values.get(node_info['model_name']) and len(latency_values[node_info['model_name']]) == 10:
                latency_values[node_info['model_name']].popleft()
                latency_values[node_info['model_name']].append(latency)
                
            elif not latency_values.get(node_info['model_name']):
                latency_values[node_info['model_name']] = deque([latency], maxlen=100)
            # print(f"Received response from {ip}:{port}: {response}")
        return response
    except Exception as e:
        print(f"Error routing request: {e}")
        return "Error processing request."

def handle_client(client_socket):
    """Handles interaction with a connected client."""
    with client_socket:
        user_request = get_user_request(client_socket)
        if user_request:
            # print(f"Received request: {user_request}")
            response = route_request(user_request)
            client_socket.sendall(response.encode('utf-8'))
            ##receive evaluation score from client
            evaluation_score = client_socket.recv(1024).decode('utf-8')
            # print(f"Received evaluation score: {evaluation_score}")
            
            ##map evaluation score to model
            if confidence_values.get(user_request['model']) and len(confidence_values[user_request['model']]) < 100:
                confidence_values[user_request['model']].append(int(evaluation_score))
            elif confidence_values.get(user_request['model']) and len(confidence_values[user_request['model']]) == 100:
                confidence_values[user_request['model']].popleft()
                confidence_values[user_request['model']].append(int(evaluation_score))
            elif not confidence_values.get(user_request['model']):
                confidence_values[user_request['model']] = deque([int(evaluation_score)], maxlen=100)
            
        else:
            print("Invalid request received.")

def start_router():
    """Starts the router to listen for incoming user requests."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as router_socket:
        router_socket.bind(('0.0.0.0', ROUTER_PORT))
        router_socket.listen(5)
        print(f"Router is listening for requests on port {ROUTER_PORT}...")
        active_threads = []  # List to track active threads

        while True:
            client_socket, addr = router_socket.accept()
            print(f"Connection from {addr}")
            thread=threading.Thread(target=handle_client, args=(client_socket,)).start()
        #     active_threads.append(thread)

        # # Cleanup finished threads to avoid memory leaks
        #     active_threads[:] = [t for t in active_threads if t.is_alive()]

def start_registration_server():
    """Starts the server to accept node registrations."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reg_socket:
        reg_socket.bind(('0.0.0.0', REGISTRATION_PORT))
        reg_socket.listen(5)
        print(f"Registration server listening on port {REGISTRATION_PORT}...")

        while True:
            client_socket, addr = reg_socket.accept()
            print(f"Node registering from {addr}")
            threading.Thread(target=register_node, args=(client_socket,)).start()

def get_metrics():
    """Listens for metrics from model servers on port 9092."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as metrics_socket:
        metrics_socket.bind(('0.0.0.0', 9092))
        metrics_socket.listen(5)
        print(f"Metrics server listening on port 9092...")

        while True:
            client_socket, addr = metrics_socket.accept()
            try:
                with client_socket:
                    data = client_socket.recv(1024).decode('utf-8')
                    
                    print("\033[93m" + f"Received metrics from {addr}: {data}")
                    print("Current metrics:")
                    local_ip, gpu_usage, throughput,inf_time_str = data.split(',')
                    print(f"Local IP: {local_ip}")
                    print(f"GPU Usage: {gpu_usage}")
                    print(f"Throughput: {throughput}")
                    print(f"Inf time: {inf_time_str}")
                    
                    for model_name in list(latency_values.keys()):
                        avg_latency = sum(latency_values[model_name])/len(latency_values[model_name]) if len(latency_values[model_name]) else 0.0
                        print(f"Model : {model_name} Latency: {avg_latency}")
                        print(f"Model : {model_name} Confidence: {sum(confidence_values[model_name])/len(confidence_values[model_name]) if len(confidence_values[model_name]) else 0.0}")
                    
                    metrics_json = {
                        "local_ip": local_ip,
                        "gpu_usage": gpu_usage,
                        "throughput": throughput,
                        "inf_time": inf_time_str,
                        "latency": {model_name: sum(latency_values[model_name])/len(latency_values[model_name]) if len(latency_values[model_name]) else 0.0 for model_name in list(latency_values.keys())},
                        "confidence": {model_name: sum(confidence_values[model_name])/len(confidence_values[model_name]) if len(confidence_values[model_name]) else 0.0 for model_name in list(confidence_values.keys())}
                    }
                    
                    ##add to model_latencies
                    for model_name in list(latency_values.keys()):
                        if model_latencies.get(model_name) and len(model_latencies[model_name]) < 100:
                            model_latencies[model_name].append(sum(latency_values[model_name])/len(latency_values[model_name]))
                        elif model_latencies.get(model_name) and len(model_latencies[model_name]) == 100:
                            model_latencies[model_name].popleft()
                            model_latencies[model_name].append(sum(latency_values[model_name])/len(latency_values[model_name]))
                        elif not model_latencies.get(model_name):
                            model_latencies[model_name] = deque([sum(latency_values[model_name])/len(latency_values[model_name])], maxlen=100)
                            
                    ##add to model_confidences
                    for model_name in list(confidence_values.keys()):
                        if model_confidences.get(model_name) and len(model_confidences[model_name]) < 100:
                            model_confidences[model_name].append(sum(confidence_values[model_name])/len(confidence_values[model_name]))
                        elif model_confidences.get(model_name) and len(model_confidences[model_name]) == 100:
                            model_confidences[model_name].popleft()
                            model_confidences[model_name].append(sum(confidence_values[model_name])/len(confidence_values[model_name]))
                        elif not model_confidences.get(model_name):
                            model_confidences[model_name] = deque([sum(confidence_values[model_name])/len(confidence_values[model_name])], maxlen=100)
                            
                    # ##add to model_gpu_usage
                    # if model_gpu_usage.get(local_ip) and len(model_gpu_usage[local_ip]) < 100:
                    #     model_gpu_usage[local_ip].append(float(gpu_usage))
                    # elif model_gpu_usage.get(local_ip) and len(model_gpu_usage[local_ip]) == 100:
                    #     model_gpu_usage[local_ip].popleft()
                    #     model_gpu_usage[local_ip].append(float(gpu_usage))
                    # elif not model_gpu_usage.get(local_ip):
                    #     model_gpu_usage[local_ip] = deque([float(gpu_usage)], maxlen=100)
                    
                    ##add to model_max_latency. For each model, keep track of max latency
                    if model_name in model_max_latency:
                        model_max_latency[model_name] = max(model_max_latency[model_name], sum(latency_values[model_name])/len(latency_values[model_name]))
                    else:
                        model_max_latency[model_name] = sum(latency_values[model_name])/len(latency_values[model_name])
                        
                    ##add to model_min_latency. For each model, keep track of min latency
                    if model_name in model_min_latency:
                        model_min_latency[model_name] = min(model_min_latency[model_name], sum(latency_values[model_name])/len(latency_values[model_name]))
                    else:
                        model_min_latency[model_name] = sum(latency_values[model_name])/len(latency_values[model_name])
                        
                    ##add to model_max_confidence. For each model, keep track of max confidence
                    if model_name in model_max_confidence:
                        model_max_confidence[model_name] = max(model_max_confidence[model_name], sum(confidence_values[model_name])/len(confidence_values[model_name]))
                    else:
                        model_max_confidence[model_name] = sum(confidence_values[model_name])/len(confidence_values[model_name])
                        
                    ##add to model_min_confidence. For each model, keep track of min confidence
                    if model_name in model_min_confidence:
                        model_min_confidence[model_name] = min(model_min_confidence[model_name], sum(confidence_values[model_name])/len(confidence_values[model_name]))
                    else:
                        model_min_confidence[model_name] = sum(confidence_values[model_name])/len(confidence_values[model_name])
                    
                    
                    
                    jsonl_file = "metrics.jsonl"
                    with open(jsonl_file, "a", encoding="utf-8") as file:
                        file.write(json.dumps(metrics_json) + "\n")    
                    
                    print("\033[0m")
                        
            except Exception as e:
                print(f"Error receiving metrics: {e}")

if __name__ == "__main__":
    threading.Thread(target=start_registration_server, daemon=True).start()
    threading.Thread(target=get_metrics, daemon=True).start()
    start_router()
