import socket
import json
import sys 
import threading

from evaluate_response import evaluate_with_gpt4

# The router's IP and port (update as needed)
ROUTER_IP = '127.0.0.1'  # Change this to the actual router's IP
ROUTER_PORT = 9090
test_file = sys.argv[1]
def load_requests():
    """Loads requests from a file."""
    requests = []
    responses = []
    with open(test_file, 'r' , encoding='utf-8') as file:
        for line in file: 
            data = json.loads(line.strip())
            if 'fitness' in test_file:
                requests.append(data['instruction'])
                responses.append(data['completion'])
            elif 'mental' in test_file:
                requests.append(data['input'])
                responses.append(data['output'])
            elif 'medical' in test_file:
                requests.append(data['merged_input'])
                responses.append(data['output'])
    return requests[:100], responses[:100]

def send_request(message, ideal_response):
    """Sends a request to the router and receives a response."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((ROUTER_IP, ROUTER_PORT))
            client_socket.sendall(message.encode('utf-8'))
            
            # Receive response from the router
            response = client_socket.recv(1024).decode('utf-8')
            print(f"Received response: {response}")
            
            evaluation_score = evaluate_with_gpt4(message, response, ideal_response)
            print(f"Evaluated response with GPT-4: {evaluation_score}")
            
            # Send evaluation score back to router
            client_socket.sendall(str(evaluation_score).encode('utf-8'))
            
            # Store data in JSONL format
            log_entry = {
                "request": message,
                "response": response,
                "ideal_response": ideal_response,
                "evaluation_score": evaluation_score
            }
            if 'fitness' in test_file:
                with open('results_fitness.jsonl', 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry) + '\n')
            elif 'mental' in test_file:
                with open('results_mental.jsonl', 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry) + '\n')
            elif 'medical' in test_file:
                with open('results_medical.jsonl', 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry) + '\n')
            
            return response, evaluation_score
    except Exception as e:
        print(f"Error communicating with the router: {e}")
        return "Error", "Error"
    
    
def client_worker(client_id, requests):
    while True: 
        try : 
            request = requests.pop(0)
            ideal_response = responses.pop(0)
        except IndexError:
            break
        print(f"[Client {client_id}] Sending message")
        generated_response , score = send_request(request, ideal_response)
        print(generated_response)
        print (100 - len(requests) )
        print(f"[Client {client_id}] Received received")

if __name__ == "__main__":
    

    num_threads = int(sys.argv[2])
    
    requests , responses = load_requests()
    print(f"Loaded {len(requests)} requests. Distributing across {num_threads} clients...")
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=client_worker, args=(i, requests))
        threads.append(thread)
        thread.start()
        
    for thread in threads:
        thread.join()
        
    print("All threads have completed.")
    
    # request_message = input("Enter your request: ")
    # print(f"Sending request: {request_message}")
    # response = send_request(request_message)
    # print(f"Received response: {response}")
