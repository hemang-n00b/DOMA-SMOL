import torch
import gc
from collections import OrderedDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from transformers import GemmaTokenizer
import os
import sys
import time
import socket
import threading
import sys

os.environ['TRANSFORMERS_CACHE'] = ''
os.environ['HF_HOME'] = ''
os.environ['TORCH_HOME'] = ''

device_id = sys.argv[1]
os.environ['CUDA_VISIBLE_DEVICES'] = f'{device_id}'

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")



class HFModel:
    def __init__(self, model_name, gpu_ip, gpu_port):
        """
        model_name: Hugging Face model identifier (or path) e.g., "./finetune/models/gemma_overtrained_fitness"
        gpu_ip, gpu_port: Identifiers for the GPU hosting this model (for logging/management)
        """
        self.model_name = model_name
        self.gpu_ip = gpu_ip
        self.gpu_port = gpu_port
        self.model = None  # Will hold the actual transformer model
        self.tokenizer = None

    def load(self):
        """Load the model and tokenizer, move the model to GPU."""
        print(f"Loading model '{self.model_name}' onto GPU at {self.gpu_ip}:{self.gpu_port}...")
        load_start = time.time()
        # Load the model and tokenizer from Hugging Face
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map="auto")
        if "gemma" in self.model_name:
            self.tokenizer = GemmaTokenizer.from_pretrained(self.model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Move the model to GPU (this is usually already done via device_map, but we ensure it)
        self.model = self.model.to('cuda')
        self.model.eval()  # Optional: set to evaluation mode
        print(f"Model loaded in {time.time() - load_start:.2f} seconds.")

    def unload(self):
        """
        Unload the model from GPU by moving it back to CPU, deleting references,
        and clearing the CUDA cache.
        """
        unload_start = time.time()
        if self.model is not None:
            print(f"Unloading model '{self.model_name}' from GPU at {self.gpu_ip}:{self.gpu_port}...")
            # Move model to CPU first
            self.model.cpu()
            # Delete the model and tokenizer to free memory
            del self.model
            self.model = None
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            # Run garbage collection and clear the CUDA cache
            gc.collect()
            torch.cuda.empty_cache()
            
            print(f"Model unloaded in {time.time() - unload_start:.2f} seconds.")
        else:  
            print(f"Model '{self.model_name}' is already unloaded.")


class ModelCache:
    def __init__(self, capacity=4, policy="LRU", demand_window=10):
        """
        capacity: Maximum number of models to hold in GPU memory.
        policy: Eviction policy, "LRU" or "demand".
        demand_window: Time window (in seconds) for demand tracking (used only if policy != "LRU").
        """
        self.capacity = capacity
        self.policy = policy
        self.cache = OrderedDict()
        # For demand-based policy, we track request timestamps for each model.
        if self.policy != "LRU":
            self.request_history = {}  # {model_key: [timestamp1, timestamp2, ...]}
            self.demand_window = demand_window

    def get_model(self, model_key, model_lookup):
        """
        Retrieve a model by key.
        model_lookup: Dictionary mapping model keys to HFModel instances.
        """
        now = time.time()
        # If using demand-based eviction, record the request timestamp.
        if self.policy != "LRU":
            if model_key not in self.request_history:
                self.request_history[model_key] = []
            self.request_history[model_key].append(now)
        
        if model_key in self.cache:
            # Model is already in cache.
            if self.policy == "LRU":
                self.cache.move_to_end(model_key)
            print(f"Model '{model_key}' found in cache.")
            return self.cache[model_key]
        else:
            # Cache miss; need to load a new model.
            if self.policy == "LRU":
                if len(self.cache) >= self.capacity:
                    evicted_key, evicted_model = self.cache.popitem(last=False)
                    evicted_model.unload()
                    print(f"Evicted model '{evicted_key}' from cache (LRU policy).")
            else:
                # Demand-based policy: evict the model with the least number of requests in the past demand_window seconds.
                if len(self.cache) >= self.capacity:
                    min_count = float('inf')
                    model_to_evict = None
                    # Evaluate each model in the cache.
                    for key in list(self.cache.keys()):
                        timestamps = self.request_history.get(key, [])
                        # Remove timestamps older than demand_window.
                        timestamps = [t for t in timestamps if t >= now - self.demand_window]
                        self.request_history[key] = timestamps  # Update cleaned timestamps.
                        count = len(timestamps)
                        if count < min_count:
                            min_count = count
                            model_to_evict = key
                    if model_to_evict is not None:
                        evicted_model = self.cache.pop(model_to_evict)
                        evicted_model.unload()
                        print(f"Evicted model '{model_to_evict}' from cache (demand policy: {min_count} requests in last {self.demand_window} sec).")
            # Load the new model and add it to the cache.
            new_model = model_lookup[model_key]
            new_model.load()
            self.cache[model_key] = new_model
            return new_model

    def current_cache(self):
        """Return the list of model keys currently in cache."""
        return list(self.cache.keys())

eviction_policy = "demand"
cache = ModelCache(capacity=4, policy=eviction_policy, demand_window=60)
model_lookup = {
        "gemma-fitness": HFModel("./finetune/models/gemma_overtrained_fitness", "192.168.1.1", 5000),
        "gemma-medical": HFModel("./finetune/models/gemma_overtrained_medical_instruct", "192.168.1.2", 5001),
        "gemma-mental": HFModel("./finetune/models/gemma_overtrained_mental_health", "192.168.1.3", 5002),
        "llama-fitness": HFModel("./finetune/models/llama_overtrained_fitness", "192.168.1.4", 5003),
        "llama-medical": HFModel("./finetune/models/llama_overtrained_medical_instruct", "192.168.1.5", 5004),
        "llama-mental": HFModel("./finetune/models/llama_overtrained_mental_health", "127.0.0.1", 5005)
 }


def get_ip():
    """Get the local IP address of this machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connect to an external IP. This doesn't actually send data,
        # but forces the OS to assign a proper local IP.
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception as e:
        print(f"Error obtaining local IP: {e}")
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

local_ip = get_ip()

def register_model(model_name, listen_port, router_host, router_reg_port):
    """
    Connect to the router's registration service and register the model.
    Registration message format: "ip,port,model_name,dummy_description"
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((router_host, router_reg_port))

            dummy_description = f"Dummy description for {model_name}"
            registration_msg = f"{local_ip},{listen_port},{model_name},{dummy_description}"
            sock.sendall(registration_msg.encode('utf-8'))
            response = sock.recv(1024).decode('utf-8')
            print(f"Registration response for '{model_name}': {response}")
    except Exception as e:
        print(f"Error registering model '{model_name}': {e}")

def register_all_models(router_host, router_reg_port, listen_port):
    """
    Register each available model with the router.
    Even though the models are not loaded yet, they are advertised as available.
    """
    for model_name in model_lookup.keys():
        register_model(model_name, listen_port, router_host, router_reg_port)


# --- Server to Handle Requests ---
def handle_client(client_socket):
    """
    Handle an incoming connection.
    Expect the request to be in the format: "model_name|request_text"
    """
    with client_socket:
        try:
            message = client_socket.recv(4096).decode('utf-8')
            if not message:
                return
            # Parse the incoming message.
            if '|' in message:
                model_name, request_text = message.split('|', 1)
            else:
                # If no delimiter is provided, assume a default model.
                model_name = "bert-base-uncased"
                request_text = message
            print(f"Received request for model '{model_name}': {request_text}")
            # Retrieve (and load if needed) the model using the cache.
            model = cache.get_model(model_name, model_lookup)
            response = model.predict(request_text)
            client_socket.sendall(response.encode('utf-8'))
        except Exception as e:
            print(f"Error handling client request: {e}")
            
def start_server(listen_port):
    """
    Start the server that listens for incoming requests forwarded by the router.
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', listen_port))
    server_socket.listen(5)
    print(f"Server listening on port {listen_port} for requests...")
    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")
        threading.Thread(target=handle_client, args=(client_socket,), daemon=True).start()

def main():
    # Define a lookup for Hugging Face models with their corresponding GPU identifiers.
    
    router_reg_port = 9091
    # Port on which this node will listen for incoming requests.
    listen_port = 5000

    router_host = sys.argv[1]
    # First, register all available models with the router.
    register_all_models(router_host, router_reg_port, listen_port)

    # Start the server to process incoming requests.
    start_server(listen_port)


if __name__ == "__main__":
    main()
