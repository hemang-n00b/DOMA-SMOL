import torch
import gc
from collections import OrderedDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GemmaTokenizer
import socket
import threading
import json
import sys
import os 
import matplotlib.pyplot as plt
from collections import deque
import time
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import pynvml

from hf_model import HFModel

# torch.set_default_dtype(torch.float32)
# torch._dynamo.config.suppress_errors = True
# torch._inductor.config.triton.cudagraphs = False  
# torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

router_host = sys.argv[1]
doma_host = sys.argv[2]
ROOT_MODEL_PATH = sys.argv[3]
MAX_NEW_TOKENS = int(sys.argv[4])
router_port = 8080 
listen_port = 5000

with open("descriptions.json", "r", encoding="utf-8") as file:
    model_description = json.load(file)
    
model_energy_ratio ={
    "gemma-fitness": 0.17,
    "gemma-medical": 0.1483,
    "gemma-mental": 0.1319,
    "llama-fitness": 0.1785,
    "llama-medical": 0.1835,
    "llama-mental": 0.1884
}


    
model_lookup = {
        "gemma-fitness": HFModel(f"{ROOT_MODEL_PATH}/gemma-2-2b-it-fitness", model_description = model_description["gemma-fitness"] , max_tokens=MAX_NEW_TOKENS , gpu_ip = router_host , gpu_port = listen_port),
        "gemma-medical": HFModel(f"{ROOT_MODEL_PATH}/gemma-2-2b-it-medical-instruct", model_description = model_description["gemma-medical"] , max_tokens=MAX_NEW_TOKENS , gpu_ip = router_host , gpu_port = listen_port),
        "gemma-mental": HFModel(f"{ROOT_MODEL_PATH}/gemma-2-2b-it-mental", model_description = model_description["gemma-mental"] , max_tokens=MAX_NEW_TOKENS , gpu_ip = router_host , gpu_port = listen_port),
        # "llama-fitness": HFModel(f"{ROOT_MODEL_PATH}/Llama-3.2-3B-Instruct-fitness", model_description = model_description["llama-fitness"] , max_tokens=MAX_NEW_TOKENS , gpu_ip = router_host , gpu_port = listen_port),
        "llama-medical": HFModel(f"{ROOT_MODEL_PATH}/Llama-3.2-3B-Instruct-medical-instruct", model_description = model_description["llama-medical"] , max_tokens=MAX_NEW_TOKENS , gpu_ip = router_host , gpu_port = listen_port),
        "llama-mental": HFModel(f"{ROOT_MODEL_PATH}/Llama-3.2-3B-Instruct-mental", model_description = model_description["llama-mental"] , max_tokens=MAX_NEW_TOKENS , gpu_ip = router_host , gpu_port = listen_port)
   }

for model in model_lookup.values():
    model.load()


def get_per_query_energy(start, end, model_name):
    relevant_energy_data = {
        t: data for t, data in energy_monitor.energy_data.items() if start <= t <= end
    }
    energies = []
    for t, data in relevant_energy_data.items():
            e_sum = sum(model_energy_ratio[val] for val in data["active_models"])
            k = data["energy"] * model_energy_ratio[model_name] / e_sum
            energies.append(k)
    return (sum(energies)/len(energies))*(end-start)
    

class ModelWorker:
    def __init__(self, model_name, model):
        self.model_name = model_name
        self.model = model
        self.queue = Queue()
        self.thread = threading.Thread(target=self.process_queue, daemon=True)
        self.thread.start()
        self.is_running = False
    
    def process_queue(self):
        while True:
            print(f"Processing queue for model '{self.model_name}'...")
            client_socket, request_text = self.queue.get()
            self.is_running = True
            start = time.time() - start_time
            response = self.model.predict(request_text)
            end = time.time() - start_time
            energy_consumption = get_per_query_energy(start, end, self.model_name)
            self.is_running = False
            response_msg= f"{energy_consumption}|{response}"
            client_socket.sendall(response_msg.encode('utf-8'))
            client_socket.close()
            
model_workers = {name: ModelWorker(name, model) for name, model in model_lookup.items()}

class EnergyMonitor:
    def __init__(self, start_time ,interval=1 ):
        self.interval = interval
        self.running = True
        self.energy_data = {} 
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.thread = threading.Thread(target=self.monitor_energy, args=(start_time,), daemon=True)
        self.thread.start()
        
    def get_gpu_energy(self):
        energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.gpu_handle) 
        return energy
    
    def monitor_energy(self , start_time):
        start_energy = self.get_gpu_energy()
        prev_energy = start_energy
        while self.running:
            active_models = [name for name, worker in model_workers.items() if worker.is_running]
            energy_now= self.get_gpu_energy()
            timestamp = time.time()
            self.energy_data[timestamp - start_time] = {
                "active_models": active_models,
                "energy": energy_now - prev_energy,
            }
            prev_energy = energy_now
            time.sleep(self.interval)
    
    def stop(self):
        self.running = False
        self.thread.join()
        pynvml.nvmlShutdown()
        
start_time = time.time()
energy_monitor = EnergyMonitor(start_time)

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
    model_workers[model_name].queue.put((client_socket, request_text))

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', listen_port))
    server_socket.listen(20)
    while True:
        client_socket, addr = server_socket.accept()
        handle_client(client_socket)


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
    start_server()
    energy_monitor.stop()
    
