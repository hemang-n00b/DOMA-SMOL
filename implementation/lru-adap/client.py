import socket
import json
import sys
import threading
import time
from datetime import datetime
from evaluate_response import evaluate_with_gpt4

SERVER_IP = '127.0.0.1' 
SERVER_PORT = 8081

FILE_PATH = sys.argv[1]

def load_requests():
    requests = []
    responses = []
    timestamps = []
    with open(FILE_PATH, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            timestamps.append(data['timestamp'])
            if 'fitness' in data['source']:
                requests.append(data['instruction'])
                responses.append(data['completion'])
            elif 'mental' in data['source']:
                requests.append(data['input'])
                responses.append(data['output'])
            elif 'medical' in data['source']:
                requests.append(data['merged_input'])
                responses.append(data['output'])
            elif 'legal' in data['source']:
                requests.append(data['question'])
                responses.append(data['answer'])
            elif 'finance' in data['source']:
                requests.append(data['question'])
                responses.append(data['answer'])
    return requests , responses , timestamps

def send_request(request, ideal_response):
    
    print(f"[Thread] Sending request at {datetime.now()}")
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        
        client_socket.connect((SERVER_IP, SERVER_PORT))
        client_socket.sendall(request.encode('utf-8'))
        
        response = client_socket.recv(4096).decode('utf-8')
        print(f"[Thread] Received response at {datetime.now()}")
        
        evaluation_score = evaluate_with_gpt4(request, response, ideal_response)
        msg = f"{ideal_response[:4095]}|{evaluation_score}"
        client_socket.sendall(msg.encode('utf-8'))
        

if __name__ == "__main__":
    
    requests,responses,timestamps = load_requests()
    
    grouped_requests = {}
    for i,request in enumerate(requests):
        if timestamps[i] not in grouped_requests:
            grouped_requests[timestamps[i]] = []
        grouped_requests[timestamps[i]].append((request, responses[i]))
        
    print(f"Loaded {len(requests)} requests. Spawning threads dynamically based on timestamps...")
    
    main_start = time.time()
    threads = []
    
    for timestamp in sorted(grouped_requests.keys()):
        current_time = time.time()
        sleep_time = max(0, timestamp - current_time + main_start)  # Ensure non-negative sleep time
        if sleep_time !=0 :
            time.sleep(sleep_time)
        
        for request, ideal_response in grouped_requests[timestamp]:
        #     print(1)
            thread = threading.Thread(target=send_request, args=(request, ideal_response))
            threads.append(thread)
            thread.start()
        
    for thread in threads:
        thread.join()
    
    print("All requests have been processed.")
    