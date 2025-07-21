import torch
import time
import threading
import queue
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- Global shared structures ---
# Queue for incoming queries
query_queue = queue.Queue()

# Lock for safely updating the per-query energy log
energy_lock = threading.Lock()
query_energy_log = {}

# Global list to store periodic GPU measurements
energy_log_entries = []
energy_log_lock = threading.Lock()

# Global counter for models active on the GPU
active_models = 0
active_models_lock = threading.Lock()

# Flag to stop the energy logger thread
stop_logging = threading.Event()

# Initialize NVML (NVIDIA Power Monitoring)
nvmlInit()
gpu_handle = nvmlDeviceGetHandleByIndex(0)  # Assume GPU 0 is used

def get_gpu_power():
    """Returns the current GPU power usage in watts."""
    # nvmlDeviceGetPowerUsage returns power in milliwatts.
    return nvmlDeviceGetPowerUsage(gpu_handle) / 1000.0  # Convert mW to W

def energy_logger():
    """
    Logs the GPU power and the number of models active every 5 seconds.
    Each log entry is a tuple: (timestamp, gpu_power_in_watts, num_active_models)
    """
    while not stop_logging.is_set():
        ts = time.time()
        power = get_gpu_power()
        with active_models_lock:
            num_models = active_models
        with energy_log_lock:
            energy_log_entries.append((ts, power, num_models))
        time.sleep(5)  # Log every 5 seconds

def process_query(model_name, query_id, query_text):
    """
    Processes a single query:
      - Increments active model count.
      - Loads the model and tokenizer.
      - Records query start and end times.
      - Runs inference.
      - Computes query-specific energy consumption by selecting logger entries within the query's runtime,
        attributing only a fraction of the GPU power (dividing by number of models active).
    """
    global active_models

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Record query start time (using time.time() so that it matches logger timestamps)
    query_start = time.time()
    
    # Indicate a model is being used
    with active_models_lock:
        active_models += 1

    # Load model and tokenizer (each query loads its own model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    
    inputs = tokenizer(query_text, return_tensors="pt").to(device)
    
    # Run inference without gradient calculation
    with torch.no_grad():
        _ = model(**inputs)
    
    # Record query end time
    query_end = time.time()
    
    # Model is no longer active for this query
    with active_models_lock:
        active_models -= 1

    # Now, extract the energy log entries that fall within the query's execution period
    with energy_log_lock:
        relevant_entries = [entry for entry in energy_log_entries if query_start <= entry[0] <= query_end]
    
    # If no log entries fall inside the query period (e.g. very fast query),
    # then take an average of the GPU power at start and end times.
    if not relevant_entries:
        # Take a simple average of power at start and end.
        power_start = get_gpu_power()  # using current reading (approximation)
        time.sleep(0.1)  # brief pause
        power_end = get_gpu_power()
        avg_allocated_power = ((power_start + power_end) / 2.0)
    else:
        # For each entry, attribute only the share corresponding to one model:
        allocated_powers = []
        for ts, power, num_models in relevant_entries:
            if num_models > 0:
                allocated_powers.append(power / num_models)
            else:
                allocated_powers.append(0)
        avg_allocated_power = sum(allocated_powers) / len(allocated_powers)
    
    # Compute the energy consumed (in joules) for this query:
    # energy (J) = average allocated power (W) * execution duration (s)
    execution_duration = query_end - query_start
    energy_consumed = avg_allocated_power * execution_duration

    # Log the results for this query
    with energy_lock:
        query_energy_log[query_id] = {
            "query": query_text,
            "energy_consumed_joules": energy_consumed,
            "execution_time_sec": execution_duration
        }

def query_worker(model_name):
    """Worker thread that continuously processes queries from the queue."""
    while True:
        query_id, query_text = query_queue.get()
        # Use a stop signal: when query_text is None
        if query_text is None:
            query_queue.task_done()
            break
        process_query(model_name, query_id, query_text)
        query_queue.task_done()

# --- Start the energy logging thread ---
logger_thread = threading.Thread(target=energy_logger)
logger_thread.start()

# --- Launch model-specific worker threads ---
models = ["bert-base-uncased", "distilbert-base-uncased"]
num_threads = len(models)
threads = []
for model in models:
    t = threading.Thread(target=query_worker, args=(model,))
    t.start()
    threads.append(t)

# --- Simulate incoming queries ---
queries = [
    "What is the capital of France?",
    "Who discovered gravity?",
    "What is AI?",
    "Explain quantum computing",
    "Define machine learning",
]

for i, q in enumerate(queries):
    query_queue.put((i, q))

# Wait until all queries are processed
query_queue.join()

# Signal the worker threads to stop
for _ in range(num_threads):
    query_queue.put((None, None))

for t in threads:
    t.join()

# Stop the energy logger thread
stop_logging.set()
logger_thread.join()

# --- Report energy consumption per query ---
print("\n=== Energy Consumption Report ===")
with energy_lock:
    for query_id, data in query_energy_log.items():
        print(f"Query {query_id}: {data['query']}")
        print(f"  🔋 Energy: {data['energy_consumed_joules']:.4f} J")
        print(f"  ⏱️ Execution Time: {data['execution_time_sec']:.4f} sec")
        print("-" * 40)
