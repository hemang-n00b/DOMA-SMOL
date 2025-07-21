import torch
import socket
import threading
import json
import sys
import time
from queue import Queue
from collections import deque
import pynvml

from hf_model import HFModel
from model_decider import model_decider_sa_latency,model_decider_sa_confidence

# Configuration
router_host = sys.argv[1]
doma_host = sys.argv[2]
ROOT_MODEL_PATH = sys.argv[3]
MAX_NEW_TOKENS = int(sys.argv[4])
router_port = 8080 
listen_port = 5000

# Initialize data structures
inference_times = {}
request_queue = Queue()
inference_in_progress = False
active_model = None
loaded_models = OrderedDict()
MAX_LOADED_MODELS = 3
cache_hits = 0
cache_miss = 0

# Load model descriptions
with open("descriptions.json", "r", encoding="utf-8") as file:
    model_description = json.load(file)

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

model_lookup = {
#     "gemma-fitness": HFModel(f"{ROOT_MODEL_PATH}/gemma-2-2b-it-fitness", model_description=model_description["gemma-fitness"], max_tokens=MAX_NEW_TOKENS, gpu_ip=router_host, gpu_port=listen_port),
#     "gemma-medical": HFModel(f"{ROOT_MODEL_PATH}/gemma-2-2b-it-medical-instruct", model_description=model_description["gemma-medical"], max_tokens=MAX_NEW_TOKENS, gpu_ip=router_host, gpu_port=listen_port),
#     "gemma-mental": HFModel(f"{ROOT_MODEL_PATH}/gemma-2-2b-it-mental", model_description=model_description["gemma-mental"], max_tokens=MAX_NEW_TOKENS, gpu_ip=router_host, gpu_port=listen_port),
#     "llama-fitness": HFModel(f"{ROOT_MODEL_PATH}/Llama-3.2-3B-Instruct-fitness", model_description=model_description["llama-fitness"], max_tokens=MAX_NEW_TOKENS, gpu_ip=router_host, gpu_port=listen_port),
#     "llama-medical": HFModel(f"{ROOT_MODEL_PATH}/Llama-3.2-3B-Instruct-medical-instruct", model_description=model_description["llama-medical"], max_tokens=MAX_NEW_TOKENS, gpu_ip=router_host, gpu_port=listen_port),
#     "llama-mental": HFModel(f"{ROOT_MODEL_PATH}/Llama-3.2-3B-Instruct-mental", model_description=model_description["llama-mental"], max_tokens=MAX_NEW_TOKENS, gpu_ip=router_host, gpu_port=listen_port)
      "gemma-medical": HFModel(f"{ROOT_MODEL_PATH}/gemma-2-2b-it-medical-instruct", model_description=model_description["gemma-medical"], max_tokens=MAX_NEW_TOKENS, gpu_ip=router_host, gpu_port=listen_port),
      "gemma-legal": HFModel(f"{ROOT_MODEL_PATH}/gemma-2-2b-it-legal", model_description=model_description["gemma-legal"], max_tokens=MAX_NEW_TOKENS, gpu_ip=router_host, gpu_port=listen_port),
      "llama-finance": HFModel(f"{ROOT_MODEL_PATH}/Llama-3.2-1B-Instruct-finance", model_description=model_description["llama-finance"], max_tokens=MAX_NEW_TOKENS, gpu_ip=router_host, gpu_port=listen_port),
      "phi-medical": HFModel(f"{ROOT_MODEL_PATH}/Phi-3-mini-4k-instruct-medical-instruct", model_description=model_description["phi-medical"], max_tokens=MAX_NEW_TOKENS, gpu_ip=router_host, gpu_port=listen_port),
      "phi-legal": HFModel(f"{ROOT_MODEL_PATH}/Phi-3-mini-4k-instruct-legal", model_description=model_description["phi-legal"], max_tokens=MAX_NEW_TOKENS, gpu_ip=router_host, gpu_port=listen_port),
      "phi-finance": HFModel(f"{ROOT_MODEL_PATH}/Phi-3-mini-4k-instruct-finance", model_description=model_description["phi-finance"], max_tokens=MAX_NEW_TOKENS, gpu_ip=router_host, gpu_port=listen_port),
}



# Load all models
for model in model_lookup.values():
    model.load()
start_time = time.time()

# GPU energy monitoring
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def get_gpu_energy():
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle)

def process_requests():
    global inference_in_progress, active_model
    while True:
        client_socket, request_text, similarities,model_metrics = request_queue.get()
        inference_in_progress = True
        
        try:
            model_name = model_decider_sa_confidence(
                request_text,
                {k: {"model_name": k, "description": v.model_description} 
                 for k, v in model_lookup.items()},
                similarities=similarities,
            )
            
            active_model = ensure_model_loaded(model_name)
            energy_before = get_gpu_energy()
            inf_time_start = time.time()
            response = active_model.predict(request_text)
            inf_time = time.time() - inf_time_start
            energy_consumption = get_gpu_energy() - energy_before
            
            # Update inference times
            if model_name not in inference_times:
                inference_times[model_name] = deque(maxlen=100)
            inference_times[model_name].append(inf_time)

            log_entry = {"model":model_name,"inf_time":inf_time}
            with open("jul4_gen_nolru_fifo_20_256.txt", 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            client_socket.sendall(f"{model_name}|{energy_consumption}|{response}".encode('utf-8'))
            
        except Exception as e:
            print(f"Error processing request: {e}")
            client_socket.sendall(f"error|0|Error: {str(e)}".encode('utf-8'))
        finally:
            client_socket.close()
            inference_in_progress = False

def register_all_models():
    for model in model_lookup.keys():
        register_model(model)

def handle_client(client_socket):
    try:
        raw_data = client_socket.recv(4096).decode('utf-8').strip()
        
        # Initialize defaults
        request_text = ""
        similarities = {}
        model_metrics = {}
        
        # Try to parse as JSON
        try:
            data = json.loads(raw_data)
            if isinstance(data, dict):
                request_text = data.get("request", "")
                similarities = data.get("similarities", {})
                model_metrics = data.get("model_metrics", {})
                # confidence_values = data.get("confidence",{})
            else:
                request_text = str(data)
        except json.JSONDecodeError:
            request_text = raw_data
        
        # Put the parsed data in the queue
        request_queue.put((
            client_socket, 
            request_text,
            similarities,
            model_metrics
        ))
        
    except Exception as e:
        print(f"Error handling client request: {e}")
        client_socket.sendall("error|0|Request processing failed".encode('utf-8'))
        client_socket.close()

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', listen_port))
    server.listen(20)
    print(f"Server listening on port {listen_port}...")
    
    while True:
        client_socket, addr = server.accept()
        threading.Thread(target=handle_client, args=(client_socket,)).start()

def register_model(model_name):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((doma_host, router_port))
        description = model_description.get(model_name, "No description")
        sock.sendall(f"{get_ip()}|{listen_port}|{model_name}|{description}".encode('utf-8'))
        print(f"Registered {model_name}: {sock.recv(1024).decode('utf-8')}")
        inference_times[model_name] = deque(maxlen=100)

def get_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"

if __name__ == "__main__":
    # Initialize inference times for all models
    for model in model_lookup:
        inference_times[model] = deque(maxlen=100)
    
    register_all_models()
    threading.Thread(target=process_requests, daemon=True).start()
    start_server()