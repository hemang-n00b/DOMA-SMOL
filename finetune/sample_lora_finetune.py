import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import sys
from accelerate import infer_auto_device_map
from transformers import BitsAndBytesConfig
import os

import huggingface_hub
huggingface_hub.login(token="")
os.environ['TRANSFORMERS_CACHE'] = ''
os.environ['HF_HOME'] = ''
os.environ['TORCH_HOME'] = ''

# device_id = sys.argv[2]
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

base_model_name = sys.argv[1]

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name, 
    device_map="auto",
    torch_dtype=torch.bfloat16, 
    cache_dir=""
)

base_model.gradient_checkpointing_enable()
base_model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.model_max_length = 512

subset_dataset = load_dataset("json", data_files=f"./datasets/fitness/train_split.json")["train"]

def format_data(sample):
 return f"""### Instruction:
        You're a supportive fitness guide who offers general advice on exercise, nutrition, and wellness. You don’t diagnose or prescribe—always encourage users to consult certified professionals for personalized help.
                                       
        ### Input:
        {sample['instruction']}
        
        ### Response:
        {sample['completion']}{tokenizer.eos_token}
        """
        
def oov_proportion(domain_words, tokenizer):
    vocab = tokenizer.vocab.keys()
    oov = [i not in vocab for i in domain_words]
    return sum(oov) / len(oov)

    
def add_new_tokens(model, tokenizer, new_tokens):
    new_tokens = list(set(new_tokens) - set(tokenizer.vocab.keys()))
    n_new_tokens = tokenizer.add_tokens(new_tokens)
    print(f"{n_new_tokens} tokens added to tokenizer")
    model.resize_token_embeddings(len(tokenizer))

    with torch.no_grad():
        for i, token in enumerate(reversed(new_tokens), start=1):
            tokenized = tokenizer.tokenize(token) 
            tokenized_ids = tokenizer.convert_tokens_to_ids(tokenized) 
            model.model.embed_tokens.weight[-i, :] = model.model.embed_tokens.weight[tokenized_ids].mean(axis=0)
            
lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj" ,"k_proj", "o_proj" , "dense" , "mlp.dense_h_to_4h"],
    r=128,
    bias="none",
    task_type="CAUSAL_LM",
)

base_model = prepare_model_for_kbit_training(base_model)
base_model = get_peft_model(base_model, lora_config)

training_args = TrainingArguments(
    output_dir=base_model_name + "-fitness",
    # auto_find_batch_size=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-3,
    num_train_epochs=2,
    logging_dir="logs",
    logging_strategy="steps",
    logging_steps=500,
    warmup_ratio=0.03,
    lr_scheduler_type="linear",
)

# Setting up the actual Trainer
trainer = SFTTrainer(
    model=base_model,
    tokenizer=tokenizer,
    train_dataset=subset_dataset,
    peft_config=lora_config,
    formatting_func=format_data,
    args=training_args
)

trainer.train() 

trainer.save_model()