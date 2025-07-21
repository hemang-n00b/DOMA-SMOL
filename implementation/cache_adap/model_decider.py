from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import sys
import json
import os 

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def calculate_avg_inference_times(inference_times):
    avg_inf_time = {}
    
    for model, times in inference_times.items():
        if len(times) > 0:
            avg_inf_time[model] = sum(times) / len(times)
        else:
            avg_inf_time[model] = 0.0  # or float('inf') if you prefer
    
    return avg_inf_time


def load_decider():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embedding_model

# Decides which model to use based on the user request
def model_decider(request, registered_nodes , embedding_model):

    # model_wise_avg_metrics = {
    #         model: {
    #             "avg_latency": np.mean(model_wise_latencies[model]),
    #             "avg_confidence": np.mean(model_wise_confidences[model]),
    #             "avg_energy": np.mean(model_wise_energy[model])
    #         } for model in model_wise_latencies
    # }

    if not registered_nodes:
        raise Exception("An error occurred! Exiting...")
        sys.exit(1)
        
    model_descriptions = {
        v["model_name"]: v["description"] for v in registered_nodes.values()
    }
        
    # embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    model_keys = list(model_descriptions.keys())
    model_embeddings = embedding_model.encode(list(model_descriptions.values()), convert_to_tensor=True , device="cuda")
    user_embedding = embedding_model.encode(request, convert_to_tensor=True , device="cuda")
    similarities = util.pytorch_cos_sim(user_embedding, model_embeddings)[0]
    similarities = similarities.cpu().numpy()
    sorted_similarities = sorted(
        zip(model_keys, similarities),
        key=lambda x: x[1],
        reverse=True
    )
    model_similarity_dict = {
        model: float(score) for model, score in sorted_similarities
    }
    with open("similarities.jsonl", "a") as f:
        json.dump({
            "request": request,
            "similarities": model_similarity_dict
        }, f)
        f.write("\n")
        
    
    ##get best available model    
    
    best_model_idx = np.argmax(similarities)
    best_model = model_keys[best_model_idx]
    
    return model_keys,similarities


def model_decider_cache_sa(request_text: str,
                           available_models: dict,
                           similarities: dict = None,
                           loaded_models:dict =None) -> str:

    if not available_models:
        raise ValueError("No models available for selection")
    
    if similarities is None:
        similarities = {model: 0 for model in available_models}
     
    # If model is in loaded memory add 0.05 else add 0    
    scores = {
        model: similarities.get(model, 0) + (0.00 if model in loaded_models else 0)
        for model in available_models
    }
    
    best_model = max(scores.items(), key=lambda x: x[1])[0]
    
    return best_model

def model_decider_sa_latency(request_text: str,
                           available_models: dict,
                           similarities: dict = None,
                           inference_times: dict = None) -> str:
    if not available_models:
        raise ValueError("No models available for selection")

    # 1. Calculate average inference times
    avg_latencies = {}
    if inference_times:
        avg_latencies = {
            model: sum(times)/len(times) if times else float('inf')
            for model, times in inference_times.items()
        }
    else:
        avg_latencies = {model: float('inf') for model in available_models}

    # 2. Handle missing similarities
    if similarities is None:
        similarities = {model: 0 for model in available_models}

    # 3. Normalize latencies (0-1 range)
    latency_values = [lat for lat in avg_latencies.values() if lat != float('inf')]
    max_latency = max(latency_values) if latency_values else 1
    normalized_latencies = {
        model: (lat / max_latency) if lat != float('inf') else 1.0
        for model, lat in avg_latencies.items()
    }

    # 4. Compute scores using the formula: (1 - similarity) + (0.5 * normalized_latency)
    scores = {
        model: (1 - similarities.get(model, 0)) + (0.2 * normalized_latencies.get(model, 1))
        for model in available_models
    }

    # 5. Select model with minimum score
    best_model = min(scores.items(), key=lambda x: x[1])[0]

    return best_model

def model_decider_sa_confidence(request_text: str,
                           available_models: dict,
                           similarities: dict = None,
                           confidence_values: dict = None) -> str:

    if not available_models:
        raise ValueError("No models available for selection")

    # 1. Calculate average inference times
    avg_conf = {}
    if confidence_values:
        avg_conf = {
            model: conf if conf is not None else 5
            for model, conf in confidence_values.items()
        }
    else:
        avg_conf = {model: float('inf') for model in available_models}

    # 2. Handle missing similarities
    if similarities is None:
        similarities = {model: 0 for model in available_models}

    # 3. Normalize latencies (0-1 range)
    conf_values = [lat for lat in avg_conf.values() if lat != float('inf')]
    max_conf = 5
    normalized_conf = {
        model: (lat / max_conf) if lat != float('inf') else 1.0
        for model, lat in avg_conf.items()
    }

    # 4. Compute scores using the formula: (1 - similarity) + (0.5 * normalized_latency)
    scores = {
        model: similarities.get(model, 0) + (0.3 * normalized_conf.get(model, 1))
        for model in available_models
    }

    # 5. Select model with minimum score
    best_model = max(scores.items(), key=lambda x: x[1])[0]

    return best_model