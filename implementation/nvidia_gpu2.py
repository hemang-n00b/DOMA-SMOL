import torch
import time
import threading
import queue
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Query queue for parallel processing
query_queue = queue.Queue()

# Shared lock for thread-safe energy logging
energy_lock = threading.Lock()

# Global energy log per query
query_energy_log = {}

# Initialize NVML (NVIDIA Power Monitoring)
nvmlInit()
gpu_handle = nvmlDeviceGetHandleByIndex(0)  # Assume using GPU 0

def get_gpu_power():
    """Returns the current GPU power usage in watts."""
    return nvmlDeviceGetPowerUsage(gpu_handle) / 1000.0  # Convert mW to W

def process_query(model_name, query_id, query_text):
    """Processes a single query and logs its energy consumption."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    
    inputs = tokenizer(query_text, return_tensors="pt").to(device)

    # Measure power before inference
    power_start = get_gpu_power()
    start_time = time.perf_counter()

    with torch.no_grad():
        _ = model(**inputs)

    end_time = time.perf_counter()
    power_end = get_gpu_power()

    # Estimate energy consumption
    avg_power = (power_start + power_end) / 2  
    duration = end_time - start_time  
    energy_consumed = avg_power * (duration / 3600)  

    with energy_lock:
        query_energy_log[query_id] = {
            "query": query_text,
            "energy_consumed_joules": energy_consumed * 3600,  
            "execution_time_sec": duration
        }

def query_worker(model_name):
    """Continuously processes queries from the queue."""
    while True:
        query_id, query_text = query_queue.get()
        if query_text is None:  # Stop signal
            break
        process_query(model_name, query_id, query_text)
        query_queue.task_done()

# Example: Models running in parallel
models = ["bert-base-uncased", "distilbert-base-uncased"]
num_threads = len(models)

# Start threads
threads = []
for model in models:
    t = threading.Thread(target=query_worker, args=(model,))
    t.start()
    threads.append(t)

# Simulate continuous incoming queries
queries = [
    "What is the capital of France?",
    "Who discovered gravity?",
    "What is AI?",
    "Explain quantum computing",
    "Define machine learning",
]

for i, q in enumerate(queries):
    query_queue.put((i, q))

# Wait for all queries to be processed
query_queue.join()

# Stop worker threads
for _ in range(num_threads):
    query_queue.put((None, None))  # Stop signal

for t in threads:
    t.join()

# Print results
print("\n=== Energy Consumption Report ===")
for query_id, data in query_energy_log.items():
    print(f"Query {query_id}: {data['query']}")
    print(f"  🔋 Energy: {data['energy_consumed_joules']:.4f} J")
    print(f"  ⏱️ Execution Time: {data['execution_time_sec']:.4f} sec")
    print("-" * 40)
