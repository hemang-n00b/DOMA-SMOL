import socket
import threading
import queue
import time
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pynvml

HOST = '127.0.0.1'
PORT = 9099

os.environ["CUDA_VISIBLE_DEVICE"] = "2"

model_name = "deepseek-ai/deepseek-moe-16b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name , torch_dtype=torch.float16 , cache_dir = "/scratch/rahul.garg/hfCache" , device_map="auto" , low_cpu_mem_usage=True).to("cuda")

request_queue = queue.Queue()

pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def get_energy_consumption():
    energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle)
    return energy

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    start_energy = get_energy_consumption()
    output = model.generate(**inputs, max_new_tokens=256)
    end_energy = get_energy_consumption()
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    try: 
        generated_text = generated_text.split("### Response:")[1].strip()
    except:
        generated_text = generated_text.strip()
    energy_used = (end_energy - start_energy) / 1000  #To convert into Joules
    return generated_text, energy_used

def get_prompt(request):
    prompt = f"""### Instruction:
        You are a knowledgeable AI assistant, providing helpful, accurate, and balanced responses to queries related to legal, finace, and medical advice.
        
        ### Input:
        {request}
        
        ### Response:"""
    return prompt 

def process_requests():
    
    while True:
        client_socket, request = request_queue.get()
        if request is None:
            break
        prompt = get_prompt(request)
        response, energy_used = generate_response(prompt)
        # print(energy_used)
        response_data = f"{response}|{energy_used}"
        print(response_data)
        client_socket.sendall(response_data.encode('utf-8'))
        client_socket.close()
        
def server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen(20)
        print("[Server] Listening for connections...")

        while True:
            client_socket, addr = server_socket.accept()
            print(f"[Server] Connection from {addr}")
            request = client_socket.recv(4096).decode('utf-8')
            request_queue.put((client_socket, request))
            
if __name__ == "__main__":
    threading.Thread(target=process_requests, daemon=True).start()
    server()