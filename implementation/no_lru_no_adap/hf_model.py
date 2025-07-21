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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class HFModel:
    def __init__(self, model_name, model_description,  max_tokens=256, gpu_ip = "0.0.0.0" , gpu_port=5000):
        self.model_name = model_name
        self.model_name = model_name
        self.gpu_ip = gpu_ip
        self.gpu_port = gpu_port
        self.model_description = model_description  
        self.model = None  
        self.tokenizer = None
        self.max_tokens = max_tokens
        
        
    def load(self):
        print(f"Loading model '{self.model_name}' on GPU {self.gpu_ip}:{self.gpu_port}...")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,torch_dtype=torch.float16, trust_remote_code=True , low_cpu_mem_usage=True ,cache_dir = "/scratch/rahul.garg/hfCache").to("cuda")
        if "gemma" in self.model_name:
            self.tokenizer = GemmaTokenizer.from_pretrained(self.model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.eval() 
        
    def predict(self, text):
        print(f"Generating response for model '{self.model_name}'...")
        prompt = f"""### Instruction:
        {self.model_description}
        
        ### Input:
        {text}
        
        ### Response:"""
        # inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to("cuda")
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=True , temperature = 0.9)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        try: 
            generated_text = generated_text.split("### Response:")[1].strip()
        except:
            generated_text = generated_text.strip()
            
        return generated_text
    
    def unload(self):
        print(f"Unloading model '{self.model_name}'...")
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        self.model = None
        self.tokenizer = None
        print(f"Model '{self.model_name}' unloaded successfully.")
    
            