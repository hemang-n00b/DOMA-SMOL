import torch
import gc
import socket
import threading
import json
import sys
import os
import matplotlib.pyplot as plt
import time
from queue import Queue
import pynvml

from hf_model import HFModel

router_host = sys.argv[1]
doma_host = sys.argv[2]
ROOT_MODEL_PATH = sys.argv[3]
MAX_NEW_TOKENS = int(sys.argv[4])
router_port = 8080 
listen_port = 5000

with open("descriptions.json", "r", encoding="utf-8") as file:
    model_description = json.load(file)

model_lookup = {
    "gemma-fitness": HFModel(f"{ROOT_MODEL_PATH}/gemma-2-2b-it-fitness", model_description=model_description["gemma-fitness"], max_tokens=MAX_NEW_TOKENS, gpu_ip=router_host, gpu_port=listen_port),
    "gemma-medical": HFModel(f"{ROOT_MODEL_PATH}/gemma-2-2b-it-medical-instruct", model_description=model_description["gemma-medical"], max_tokens=MAX_NEW_TOKENS, gpu_ip=router_host, gpu_port=listen_port),
    "gemma-mental": HFModel(f"{ROOT_MODEL_PATH}/gemma-2-2b-it-mental", model_description=model_description["gemma-mental"], max_tokens=MAX_NEW_TOKENS, gpu_ip=router_host, gpu_port=listen_port),
    "llama-fitness": HFModel(f"{ROOT_MODEL_PATH}/Llama-3.2-3B-Instruct-fitness", model_description=model_description["llama-fitness"], max_tokens=MAX_NEW_TOKENS, gpu_ip=router_host, gpu_port=listen_port),
    "llama-medical": HFModel(f"{ROOT_MODEL_PATH}/Llama-3.2-3B-Instruct-medical-instruct", model_description=model_description["llama-medical"], max_tokens=MAX_NEW_TOKENS, gpu_ip=router_host, gpu_port=listen_port),
    # "llama-mental": HFModel(f"{ROOT_MODEL_PATH}/Llama-3.2-3B-Instruct-mental", model_description=model_description["llama-mental"], max_tokens=MAX_NEW_TOKENS, gpu_ip=router_host, gpu_port=listen_port)
}

for model in model_lookup.values():
    model.load()

start_time = time.time()

# This flag indicates whether an inference is in progress.
inference_in_progress = False
active_model = None

request_queues = {model_name: Queue() for model_name in model_lookup.keys()}

# Create a common FIFO queue for all incoming requests.
# request_queue = Queue()
inference_times = {model_name: [] for model_name in model_lookup.keys()}
avg_inf = {model_name: 0 for model_name in model_lookup.keys()}

pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def get_gpu_energy():
    energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle)
    return energy

def process_requests():
    global inference_in_progress
    global active_model
    while True:
        sorted_avg_inf = dict(sorted(avg_inf.items(), key=lambda item: item[1]))
        for model_ in sorted_avg_inf.keys():
            if not request_queues[model_].empty():
                client_socket, model_name, request_text = request_queues[model_].get()
                inference_in_progress = True
                active_model = model_lookup[model_name]
                start = time.time() - start_time
                energy_before = get_gpu_energy()
                response = model_lookup[model_name].predict(request_text)
                end = time.time() - start_time
                energy_after = get_gpu_energy()
                energy_consumption = (energy_after - energy_before) 
                inference_in_progress = False
                response_msg = f"{energy_consumption}|{response}"
                inference_time = end-start 
                # log_entry={"model":model_name,"inf_time":inference_time}
                # with open("inference_times_20_512.jsonl", 'a', encoding='utf-8') as f:
                #     f.write(json.dumps(log_entry) + '\n')
                inference_times[model_name].append(inference_time)
                avg_inf[model_name] = sum(inference_times[model_name])/len(inference_times[model_name])
                client_socket.sendall(response_msg.encode('utf-8'))
                client_socket.close()
                break

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception as e:
        print(f"Error obtaining local IP: {e}")
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

local_ip = get_ip()

def handle_client(client_socket):
    message = client_socket.recv(4096).decode('utf-8')
    if '|' in message:
        model_name, request_text = message.split('|', 1)
        # request_queue.put((client_socket, model_name, request_text))
        request_queues[model_name].put((client_socket,model_name , request_text))
    else:
        client_socket.sendall("Invalid request format.".encode('utf-8'))
        client_socket.close()

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', listen_port))
    server_socket.listen(20)
    print(f"Server listening on port {listen_port}...")
    while True:
        client_socket, addr = server_socket.accept()
        threading.Thread(target=handle_client, args=(client_socket,), daemon=True).start()

def register_model(model_name):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        print(router_host, router_port)
        sock.connect((doma_host, router_port))
        dummy_description = model_description.get(model_name, "No description available.")
        registration_msg = f"{local_ip}|{listen_port}|{model_name}|{dummy_description}"
        sock.sendall(registration_msg.encode('utf-8'))
        response = sock.recv(1024).decode('utf-8')
        print(f"Registration response for '{model_name}': {response}")

def register_all_models():
    for model in model_lookup.keys():
        register_model(model)

# Start energy monitoring
# energy_monitor = EnergyMonitor(start_time)

if __name__ == "__main__":
    register_all_models()
    
    # Start the inference processing thread (sequential FIFO worker)
    threading.Thread(target=process_requests, daemon=True).start()
    
    # Start the server that accepts client connections
    start_server()
    
    # When shutting down, stop the energy monitor
    # energy_monitor.stop()
