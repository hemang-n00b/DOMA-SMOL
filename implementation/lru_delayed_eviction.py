import torch
import gc
from collections import OrderedDict, deque
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GemmaTokenizer
import socket
import threading
import json
import sys
import os 
import matplotlib.pyplot as plt
import time
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from accelerate import Accelerator

# Initialize accelerate
accelerator = Accelerator()

torch.set_default_dtype(torch.float32)
torch._dynamo.config.suppress_errors = True
torch._inductor.config.triton.cudagraphs = False  
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

MAX_LEN = 100
os.environ['TRANSFORMERS_CACHE'] = ''
os.environ['HF_HOME'] = ''
os.environ['TORCH_HOME'] = ''

gpu_usage_history = deque(maxlen=MAX_LEN)
request_count_history = deque(maxlen=MAX_LEN)
time_intervals = deque(maxlen=MAX_LEN)

throughput = 0

class ModelWorker:
    def __init__(self, model_name, model_cache):
        self.model_name = model_name
        self.model_cache = model_cache
        self.queue = Queue()
        self.thread = threading.Thread(target=self.process_queue, daemon=True)
        self.thread.start()

    def process_queue(self):
        while True:
            print(f"Processing queue for model '{self.model_name}'...")
            client_socket, request_text = self.queue.get()
            model = self.model_cache.get_model(self.model_name, model_lookup)
            response = model.predict(request_text)
            client_socket.sendall(response.encode('utf-8'))
            client_socket.close()

class HFModel:
    def __init__(self, model_name, gpu_ip, gpu_port, model_description):
        """
        model_name: Hugging Face model identifier (e.g., "bert-base-uncased")
        gpu_ip, gpu_port: identifiers for the GPU hosting this model (for logging/management)
        """
        self.model_name = model_name
        self.gpu_ip = gpu_ip
        self.gpu_port = gpu_port
        self.model_description = model_description   # Description of the model's function AKA system message
        self.model = None  # Will hold the actual transformer model
        self.tokenizer = None
        self.inference_times = deque(maxlen=MAX_LEN)
        # Lock to ensure that inference and unload do not occur concurrently.
        self.inference_lock = threading.Lock()

    def load(self):
        """Load the model and tokenizer, move the model to GPU."""
        print(f"Loading model '{self.model_name}' onto GPU at {self.gpu_ip}:{self.gpu_port}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, cache_dir="/scratch/rg/hfCache")
        if "gemma" in self.model_name:
            self.tokenizer = GemmaTokenizer.from_pretrained(self.model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.model = self.model.to('cuda')
        self.model = accelerator.prepare(self.model)
        self.model.eval()

    def unload(self):
        """
        Unload the model from GPU by moving it back to CPU, deleting references,
        and clearing the CUDA cache. This method waits until any ongoing inference completes.
        """
        # Acquire the lock to ensure no inference is running.
        with self.inference_lock:
            if self.model is not None:
                print(f"Unloading model '{self.model_name}' from GPU at {self.gpu_ip}:{self.gpu_port}...")
                self.model.cpu()
                del self.model
                self.model = None
                if self.tokenizer is not None:
                    del self.tokenizer
                    self.tokenizer = None
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
    def predict(self, text):
        # Lock is held during the inference to prevent unload concurrently.
        with self.inference_lock:
            inf_time_start = time.time()
            print(f"Generating response for model '{self.model_name}'...")
            prompt = f"""### Instruction:
            {self.model_description}
            
            ### Input:
            {text}
            
            ### Response:"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to("cuda")
            inputs = accelerator.prepare(inputs)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.9)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            try: 
                generated_text = generated_text.split("### Response:")[1].strip()
            except IndexError:
                generated_text = generated_text.strip()
            
            inf_time_end = time.time()
            self.inference_times.append(inf_time_end - inf_time_start)
            
            return generated_text

class ModelCache:
    def __init__(self, capacity=4):
        self.capacity = capacity
        self.cache = OrderedDict()
        # Lock to protect concurrent access to the cache.
        self.cache_lock = threading.Lock()

    def get_model(self, model_key, model_lookup):
        """
        Retrieve a model by key (e.g., model identifier).
        model_lookup: dictionary mapping model keys to HFModel instances.
        """
        with self.cache_lock:
            if model_key in self.cache:
                self.cache.move_to_end(model_key)
                print(f"Model '{model_key}' found in cache.")
                return self.cache[model_key]
            else:
                if len(self.cache) >= self.capacity:
                    evicted_key, evicted_model = self.cache.popitem(last=False)
                    evicted_model.unload()
                    print(f"Evicted model '{evicted_key}' from cache.")
                new_model = model_lookup[model_key]
                new_model.load()
                self.cache[model_key] = new_model
                return new_model

    def current_cache(self):
        """Return the list of model keys currently in cache."""
        with self.cache_lock:
            return list(self.cache.keys())

model_cache = ModelCache(capacity=1)

with open("descriptions.json", "r", encoding="utf-8") as file:
    model_description = json.load(file)

model_lookup = {
    "gemma-fitness": HFModel("../finetune2/models/gemma-2-2b-it-fitness", "192.168.1.1", 5000, model_description=model_description["gemma-fitness"]),
    "gemma-medical": HFModel("../finetune2/models/gemma-2-2b-it-medical-instruct", "0.0.0.0", 5000, model_description=model_description["gemma-medical"]),
    "gemma-mental": HFModel("../finetune2/models/gemma-2-2b-it-mental", "0.0.0.0", 5000, model_description=model_description["gemma-mental"]),
    "llama-fitness": HFModel("../finetune2/models/Llama-3.2-3B-Instruct-fitness", "0.0.0.0", 5000, model_description=model_description["llama-fitness"]),
    "llama-medical": HFModel("../finetune2/models/Llama-3.2-3B-Instruct-medical-instruct", "0.0.0.0", 5000, model_description=model_description["llama-medical"]),
    "llama-mental": HFModel("../finetune2/models/Llama-3.2-3B-Instruct-mental", "0.0.0.0", 5000, model_description=model_description)
}

# Create a worker for each model.
workers = {model_name: ModelWorker(model_name, model_cache) for model_name in model_lookup.keys()}

def get_ip():
    """Get the local IP address of this machine."""
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
    
def register_model(model_name, listen_port, router_host, router_reg_port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((router_host, router_reg_port))
            dummy_description = model_description.get(model_name, "No description available.")
            registration_msg = f"{local_ip};{listen_port};{model_name};{dummy_description}"
            sock.sendall(registration_msg.encode('utf-8'))
            response = sock.recv(1024).decode('utf-8')
            print(f"Registration response for '{model_name}': {response}")
    except Exception as e:
        print(f"Error registering model '{model_name}': {e}")

def register_all_models(router_host, router_reg_port, listen_port):
    for model_name in model_lookup.keys():
        register_model(model_name, listen_port, router_host, router_reg_port)

def handle_client(client_socket):
    print("Handling client request...")
    message = client_socket.recv(4096).decode('utf-8')
    if not message:
        return
    if '|' in message:
        model_name, request_text = message.split('|', 1)
    else:
        model_name = "bert-base-uncased"
        request_text = message
    print(f"Received request for model '{model_name}': {request_text}")
    if model_name in workers:
        workers[model_name].queue.put((client_socket, request_text))
    else:
        print(f"Model '{model_name}' not found in cache.")
        client_socket.sendall(b"Model not found in cache.")

def monitor_system():
    plot_counter = 0
    while True:
        gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        gpu_usage_history.append(gpu_memory)
        request_count_history.append(len(request_count_history) + 1)
        time_intervals.append(time.time())
        
        plot_counter += 1
        if plot_counter >= 60:
            plt.figure(figsize=(10, 5))
            plt.plot(time_intervals, gpu_usage_history, label='GPU Memory (MB)')
            plt.xlabel('Time')
            plt.ylabel('GPU Usage (MB)')
            plt.title('GPU Utilization Over Time')
            plt.legend()
            plt.savefig(f'gpu_usage_{int(time.time())}.png')
            plt.close()
            
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.set_xlabel('Time')
            ax1.set_ylabel('GPU Usage (MB)', color='tab:red')
            ax1.plot(time_intervals, gpu_usage_history, color='tab:red', label='GPU Usage')
            ax1.tick_params(axis='y', labelcolor='tab:red')
            ax2 = ax1.twinx()
            ax2.set_ylabel('Requests Served', color='tab:blue')
            ax2.plot(time_intervals, request_count_history, color='tab:blue', label='Requests')
            ax2.tick_params(axis='y', labelcolor='tab:blue')
            plt.title('System Metrics Over Time')
            fig.savefig(f'system_metrics_{int(time.time())}.png')
            plt.close()
            plot_counter = 0
            
        time.sleep(5)

def plot_gpu_usage():
    plt.figure(figsize=(10, 5))
    plt.plot(time_intervals, gpu_usage_history, label='GPU Memory (MB)')
    plt.xlabel('Time')
    plt.ylabel('GPU Usage (MB)')
    plt.title('GPU Utilization Over Time')
    plt.legend()
    plt.show()

def plot_request_throughput():
    plt.figure(figsize=(10, 5))
    plt.plot(time_intervals, request_count_history, label='Requests Served')
    plt.xlabel('Time')
    plt.ylabel('Requests')
    plt.title('Request Throughput Over Time')
    plt.legend()
    plt.show()

def plot_system_metrics():
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Time')
    ax1.set_ylabel('GPU Usage (MB)', color='tab:red')
    ax1.plot(time_intervals, gpu_usage_history, color='tab:red', label='GPU Usage')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Requests Served', color='tab:blue')
    ax2.plot(time_intervals, request_count_history, color='tab:blue', label='Requests')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    plt.title('System Metrics Over Time')
    fig.tight_layout()
    plt.show()

monitor_thread = threading.Thread(target=monitor_system, daemon=True)
monitor_thread.start()

def start_server(listen_port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', listen_port))
    server_socket.listen(10)
    print(f"Server listening on port {listen_port}...")
    request_timestamps = deque(maxlen=100)
    try:
        while True:
            client_socket, addr = server_socket.accept()
            print(f"New connection from {addr}")
            current_time = time.time()
            request_timestamps.append(current_time)
            if len(request_timestamps) > 1:
                window_duration = current_time - request_timestamps[0]
                if window_duration > 0:
                    throughput = len(request_timestamps) / window_duration
                    print(f"Current throughput: {throughput:.2f} requests/second")
            handle_client(client_socket)
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server_socket.close()

def send_metrics():
    print("Sending metrics to router...")
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((sys.argv[1], 9092))
                gpu_usage = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                current_models = ",".join(model_cache.current_cache())
                avg_inf_times = []
                for model_name in model_cache.current_cache():
                    model = model_cache.cache[model_name]
                    if model.inference_times:
                        avg_inf = sum(model.inference_times) / len(model.inference_times)
                        avg_inf_times.append(f"{model_name}:{avg_inf:.2f}")
                    else:
                        avg_inf_times.append(f"{model_name}:0.0")
                inf_times_str = "|".join(avg_inf_times)
                metrics_msg = f"{local_ip},{gpu_usage:.2f},{throughput:.2f},{inf_times_str}"
                print(f"Sending metrics to router: {local_ip} ie {metrics_msg}")
                sock.sendall(metrics_msg.encode('utf-8'))
        except Exception as e:
            print(f"Error sending metrics: {e}")
        time.sleep(60)

def main():
    router_host = sys.argv[1]
    router_reg_port = 9091
    listen_port = 5000
    register_all_models(router_host, router_reg_port, listen_port)
    print("Model registration complete.")
    metric_sender_thread = threading.Thread(target=send_metrics, daemon=True)
    metric_sender_thread.start()
    start_server(listen_port)

if __name__ == "__main__":
    main()
