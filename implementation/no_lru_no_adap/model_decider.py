from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import sys
import json

import torch
from sentence_transformers.util import cos_sim
from sentence_transformers import CrossEncoder

def load_decider():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embedding_model

def load_decider_cross_encoder():
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")
    return cross_encoder
    
def model_decider_cross_encoder(request, registered_nodes, cross_encoder: CrossEncoder):
    if not registered_nodes:
        raise Exception("An error occurred! Exiting...")

    model_descriptions = {v["model_name"]: v["description"] for v in registered_nodes.values()}
    model_keys = list(model_descriptions.keys())

    pairs = [(request, desc) for desc in model_descriptions.values()]
    scores = cross_encoder.predict(pairs)
    best_model_idx = int(np.argmax(scores))
    return model_keys[best_model_idx]


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
    return best_model

def model_decider_maxsim(request, registered_nodes, embedding_model):
    if not registered_nodes:
        raise Exception("An error occurred! Exiting...")

    model_descriptions = {v["model_name"]: v["description"] for v in registered_nodes.values()}
    model_keys = list(model_descriptions.keys())

    request_tokens = embedding_model.tokenize([request])
    request_embeddings = embedding_model.encode([request], convert_to_tensor=True, device="cuda", output_value="token_embeddings")[0]

    scores = []
    for desc in model_descriptions.values():
        desc_embeddings = embedding_model.encode([desc], convert_to_tensor=True, device="cuda", output_value="token_embeddings")[0]
        similarity_matrix = cos_sim(request_embeddings, desc_embeddings)
        max_sim = torch.max(similarity_matrix).item()
        scores.append(max_sim)

    best_model_idx = int(np.argmax(scores))
    return model_keys[best_model_idx]



def model_decider_sa_latency(request, registered_nodes, embedding_model, model_wise_avg_metrics):
    if not registered_nodes:
        raise Exception("An error occurred! Exiting...")

    model_descriptions = {
        v["model_name"]: v["description"] for v in registered_nodes.values()
    }
    
    

    model_keys = list(model_descriptions.keys())
    model_embeddings = embedding_model.encode(list(model_descriptions.values()), convert_to_tensor=True, device="cuda")
    user_embedding = embedding_model.encode(request, convert_to_tensor=True, device="cuda")
    
    similarities = util.pytorch_cos_sim(user_embedding, model_embeddings)[0]
    similarities = similarities.cpu().numpy()
    
    # Zip models with similarity
    model_similarity_dict = {
        model: float(similarity) for model, similarity in zip(model_keys, similarities)
    }
    
    

    # Top 3 models with similarity 
    valid_models = sorted(
        [model
         for model, similarity in model_similarity_dict.items()],
        key=lambda x: x[1],
        reverse=True
    )[:3]

    
    print("valid models",valid_models)
    if not valid_models:
        raise Exception(f"No models found")

    # Among valid models, choose the one with lowest average latency
    # best_model = min(valid_models, key=lambda x: model_wise_avg_metrics[x[0]]["avg_latency"])[0]
    min_latency = float("inf")
    best_model = None
    for model in list(model_wise_avg_metrics.keys()):

        latency = model_wise_avg_metrics[model]["avg_latency"]

        if latency < min_latency and model in valid_models:
            min_latency = latency
            best_model = model
                
    if best_model is None:
        ##Get highest similarity
        best_model_idx = np.argmax(similarities)
        best_model = model_keys[best_model_idx]
    # Log similarity scores
    # with open("similarities.jsonl", "a") as f:
    #     json.dump({
    #         "request": request,
    #         "similarities": model_similarity_dict
    #     }, f)
    #     f.write("\n")
    # print(best_model)
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
            model: sum(times)/len(times) if times else float('inf')
            for model, times in confidence_values.items()
        }
    else:
        avg_conf = {model: 0 for model in available_models}

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
        model: (1 - similarities.get(model, 0)) + (0.0 * normalized_conf.get(model, 1))
        for model in available_models
    }

    # 5. Select model with minimum score
    best_model = min(scores.items(), key=lambda x: x[1])[0]

    return best_model