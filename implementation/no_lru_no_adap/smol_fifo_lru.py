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
from collections import OrderedDict
import pynvml

from hf_model import HFModel

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Uncomment or adjust these if needed:
# torch.set_default_dtype(torch.float32)
# torch._dynamo.config.suppress_errors = True
# torch._inductor.config.triton.cudagraphs = False  
# torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

router_host = sys.argv[1]
doma_host = sys.argv[2]
ROOT_MODEL_PATH = sys.argv[3]
MAX_NEW_TOKENS = int(sys.argv[4])
OUTPUT_DIR = sys.argv[5]
router_port = 8080 
listen_port = 5000
cache_hits = 0
cache_miss = 0

with open("descriptions.json", "r", encoding="utf-8") as file:
    model_description = json.load(file)
    
# model_energy_ratio = {
#     "gemma-fitness": 0.17,
#     "gemma-medical": 0.1483,
#     "gemma-mental": 0.1319,
#     "llama-fitness": 0.1785,
#     "llama-medical": 0.1835,
#     "llama-mental": 0.1884
# }

# Create model instances but do not load them initially.
model_lookup = {
    # "qwen-fitness": HFModel(f"{ROOT_MODEL_PATH}/Qwen2.5-3B-Instruct-fitness", model_description=model_description["qwen-fitness"], max_tokens=MAX_NEW_TOKENS, gpu_ip=router_host, gpu_port=listen_port),
    "phi-medical": HFModel(f"{ROOT_MODEL_PATH}/Phi-3-mini-4k-instruct-medical-instruct", model_description=model_description["phi-medical"], max_tokens=MAX_NEW_TOKENS, gpu_ip=router_host, gpu_port=listen_port),
    "llama-mental-1b": HFModel(f"{ROOT_MODEL_PATH}/Llama-3.2-1B-Instruct-mental", model_description=model_description["llama-mental-1b"], max_tokens=MAX_NEW_TOKENS, gpu_ip=router_host, gpu_port=listen_port),
    "gemma-fitness": HFModel(f"{ROOT_MODEL_PATH}/gemma-2-2b-it-fitness", model_description=model_description["gemma-fitness"], max_tokens=MAX_NEW_TOKENS, gpu_ip=router_host, gpu_port=listen_port),
    "gemma-medical": HFModel(f"{ROOT_MODEL_PATH}/gemma-2-2b-it-medical-instruct", model_description=model_description["gemma-medical"], max_tokens=MAX_NEW_TOKENS, gpu_ip=router_host, gpu_port=listen_port),
    # "gemma-mental": HFModel(f"{ROOT_MODEL_PATH}/gemma-2-2b-it-mental", model_description=model_description["gemma-mental"], max_tokens=MAX_NEW_TOKENS, gpu_ip=router_host, gpu_port=listen_port),
    # "llama-fitness": HFModel(f"{ROOT_MODEL_PATH}/Llama-3.2-3B-Instruct-fitness", model_description=model_description["llama-fitness"], max_tokens=MAX_NEW_TOKENS, gpu_ip=router_host, gpu_port=listen_port),
    # "llama-medical": HFModel(f"{ROOT_MODEL_PATH}/Llama-3.2-3B-Instruct-medical-instruct", model_description=model_description["llama-medical"], max_tokens=MAX_NEW_TOKENS, gpu_ip=router_host, gpu_port=listen_port),
    "llama-mental": HFModel(f"{ROOT_MODEL_PATH}/Llama-3.2-3B-Instruct-mental", model_description=model_description["llama-mental"], max_tokens=MAX_NEW_TOKENS, gpu_ip=router_host, gpu_port=listen_port),
    "llama-fitness-1b": HFModel(f"{ROOT_MODEL_PATH}/Llama-3.2-1B-Instruct-fitness", model_description=model_description["llama-fitness"], max_tokens=MAX_NEW_TOKENS, gpu_ip=router_host, gpu_port=listen_port),
}

# LRU cache to track loaded models.
# Key: model name; Value: HFModel instance.
loaded_models = OrderedDict()
MAX_LOADED_MODELS = 3  # Adjust capacity as needed

# Global start time (for energy monitoring)
start_time = time.time()

inference_in_progress = False
active_model = None

# Create a common FIFO queue for all incoming requests.
request_queue = Queue()
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def get_gpu_energy():
    energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle)
    return energy

def ensure_model_loaded(model_name):
    """
    Ensures the model is loaded.
    Unloads the least-recently used model if the cache capacity is exceeded.
    """
    global loaded_models
    global cache_hits   
    global cache_miss
    # If the model is already loaded, update its recency and return the instance.
    if model_name in loaded_models:
        loaded_models.move_to_end(model_name)
        cache_hits += 1
        return loaded_models[model_name]

    cache_miss += 1
    # If not loaded, unload LRU if needed.
    while len(loaded_models) >= MAX_LOADED_MODELS:
        lru_model_name, lru_model = loaded_models.popitem(last=False)
        print(f"Unloading model {lru_model_name} due to LRU policy.")
        lru_model.unload()  # Ensure your HFModel class implements unload()
    # Load the requested model.
    model = model_lookup[model_name]
    print(f"Loading model {model_name}.")
    model.load()
    loaded_models[model_name] = model
    return model

def process_requests():
    global inference_in_progress
    global active_model
    while True:
        client_socket, model_name, request_text = request_queue.get()
        inference_in_progress = True
        # Ensure the correct model is loaded (and update LRU order).
        active_model = ensure_model_loaded(model_name)
        start = time.time() - start_time
        energy_before = get_gpu_energy()
        response = active_model.predict(request_text)
        end = time.time() - start_time
        energy_after = get_gpu_energy()
        energy_consumption = (energy_after - energy_before)
        inference_in_progress = False
        response_msg = f"{energy_consumption}|{response}"
        inference_time = end - start
        cache_h_ratio = cache_hits / (cache_hits + cache_miss) if (cache_hits + cache_miss) > 0 else 0
        log_entry = {"model": model_name, "inf_time": inference_time,"cache_hits":cache_hits,"cache_hit_ratio":cache_h_ratio}
        with open(OUTPUT_DIR, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
        client_socket.sendall(response_msg.encode('utf-8'))
        client_socket.close()

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
    # Expected message format: "model_name|request_text"
    if '|' in message:
        model_name, request_text = message.split('|', 1)
        request_queue.put((client_socket, model_name, request_text))
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

if __name__ == "__main__":
    register_all_models()
    
    # Start the inference processing thread (sequential FIFO worker)
    threading.Thread(target=process_requests, daemon=True).start()
    
    # Start the server that accepts client connections
    start_server()
    
    # When shutting down, you might want to unload any remaining loaded models.
    for model in loaded_models.values():
        model.unload()
