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

from model_decider import model_decider , load_decider, model_decider_maxsim,model_decider_cross_encoder

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
    data = client_socket.recv(1024).decode('utf-8')
    ip, port, model_name, description = data.split('|')
    port = int(port)
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
         
         
# Gets user request and tries to find the best node to handle that request
def get_user_request(client_socket):
    try:
        request = client_socket.recv(4096).decode('utf-8')
        start_decision = time.time()
        model_string = model_decider(request, registered_nodes, embedding_model)
        end_decision = time.time()
        latency = end_decision - start_decision
        with open("latency_values.jsonl", 'a') as f:
            json.dump({"latency": latency}, f)
            f.write('\n')
        return {"request": request, "model": model_string}
    except Exception as e:
        print(f"Error in get_user_request: {e}")
        return None

def route_request(user_request):
    
    matching_nodes = [(ip, port, info) for (ip, port, model_name), info in registered_nodes.items() if info["model_name"] == user_request["model"]]
    ip, port, node_info = random.choice(matching_nodes)
    print(f"Routing to {ip}:{port} ({node_info['model_name']})")
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as target_socket:
        target_socket.connect((ip, port))
        user_msg = f"{user_request['model']}|{user_request['request']}"
        latency_start = time.time()
        target_socket.sendall(user_msg.encode('utf-8'))

        response_data = target_socket.recv(4096).decode('utf-8')
        energy_consumption,response = response_data.split('|',1)
        latency_end = time.time()
        latency = latency_end - latency_start
    return response, latency, energy_consumption

# handles each request
def handle_client(client_socket):
    with client_socket:
        user_request = get_user_request(client_socket)
        if user_request is None:
            client_socket.sendall("An error occurred. Please try again later.".encode('utf-8'))
            return

        response,latency,energy_consumption = route_request(user_request)
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
            "model": user_request['model']
        }
        
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