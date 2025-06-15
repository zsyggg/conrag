import argparse
import os
import re
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import get_scheduler
import json

def clean_and_split_document(doc_str):
    """
    直接处理并分割 documents 字符串，支持混合的单引号和双引号
    """
    doc_str = doc_str.replace("'", '"')
    documents = re.findall(r'"(.*?)"', doc_str)
    return documents

def get_data(file):
    queries = []
    documents = []
    consensuses = []
    with open(file, "r", encoding="utf-8") as f:
        query = None
        document_list_str = None # Changed variable name for clarity
        consensus_lines = []
        for line in f:
            line = line.strip()
            if line.startswith("query:"):
                if query and document_list_str and consensus_lines: # Check document_list_str
                    consensuses.append(" ".join(consensus_lines))
                query = line.split("query:")[1].strip()
                queries.append(query)
                consensus_lines = []
                document_list_str = None # Reset for new query
            elif line.startswith("documents:"):
                document_list_str = line.split("documents:")[1].strip()
                # Parse documents string into a list of strings
                # Assuming documents are in a JSON-like string list: "[\"doc1\", \"doc2\"]"
                try:
                    parsed_documents = json.loads(document_list_str)
                    if isinstance(parsed_documents, list):
                        documents.append([str(doc) for doc in parsed_documents]) # Ensure all are strings
                    else:
                        # Fallback or error for unexpected format
                        print(f"Warning: Documents format not a list for query '{query}': {document_list_str}")
                        documents.append([]) # Append empty list or handle error
                except json.JSONDecodeError:
                    # Fallback to original regex method if JSON parsing fails (e.g. for "doc1" "doc2" format)
                    print(f"Warning: JSONDecodeError for documents: {document_list_str}. Trying regex.")
                    parsed_documents_regex = clean_and_split_document(document_list_str)
                    documents.append(parsed_documents_regex)

            elif line.startswith("consensus:"):
                consensus_lines.append(line.split("consensus:")[1].strip())
            else:
                if line and consensus_lines:  # 捕获继续在同一个共识段落的内容
                    consensus_lines.append(line)
        # 最后一条记录的共识处理
        if query and document_list_str and consensus_lines:
             consensuses.append(" ".join(consensus_lines))
        
        # Ensure documents and consensuses have the same length as queries
        # This is a simplified alignment; more robust handling might be needed if data is misaligned
        if len(documents) < len(queries):
            documents.extend([[] for _ in range(len(queries) - len(documents))])
        if len(consensuses) < len(queries):
            consensuses.extend(["" for _ in range(len(queries) - len(consensuses))])

    return queries, documents, consensuses


def data_preprocess(file, tokenizer):
    queries, documents, consensuses = get_data(file)
    
    # Check for length mismatches after get_data
    if not (len(queries) == len(documents) == len(consensuses)):
        raise ValueError(f"Data length mismatch after get_data: "
                         f"Queries: {len(queries)}, Documents: {len(documents)}, Consensuses: {len(consensuses)}")

    # For input_texts, join the list of document strings into a single string for T5 input
    input_texts = [f"query: {q} documents: {' '.join(d_list)}" for q, d_list in zip(queries, documents)]
    
    data = pd.DataFrame({"input_text": input_texts, "target_text": consensuses})
    
    print(f"Number of records for training: {len(data)}")
    if len(data) == 0:
        raise ValueError("No data loaded for training. Please check the input file and its parsing in get_data.")

    train_data = tokenizer(data.input_text.to_list(), padding="max_length", max_length=512, truncation=True, return_tensors="pt")
    target_data = tokenizer(data.target_text.to_list(), padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    return train_data, target_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True) # Made required
    parser.add_argument('--save_path', type=str, required=True) # Made required
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train_file = args.train_file
    SEED = args.seed

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED) 

    tokenizer_path = "/workspace/conRAG/google-t5/t5-large"
    model_path = "/workspace/conRAG/google-t5/t5-large"
    
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    print(f"Loading model from: {model_path}")
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    print(f"Preprocessing data from: {train_file}")
    train_data, target_data = data_preprocess(train_file, tokenizer)

    # config
    batch_size = args.batch_size
    train_dataset = TensorDataset(train_data["input_ids"], train_data["attention_mask"], target_data["input_ids"])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Removed RandomSampler for simplicity with shuffle=True
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_epochs = args.num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- Modified DataParallel handling ---
    num_gpus = 0
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        if num_gpus > 1:
            print(f"Using DataParallel for {num_gpus} GPUs.")
            # Explicitly define device_ids for DataParallel
            device_ids = list(range(num_gpus))
            model = nn.DataParallel(model, device_ids=device_ids)
        elif num_gpus == 1:
            print("Running on a single GPU. DataParallel not used.")
            # model is already on 'device' which would be 'cuda:0'
        else: # num_gpus == 0
             print("No GPUs available. Running on CPU. DataParallel not used.")
    else:
        print("CUDA not available. Running on CPU. DataParallel not used.")
    # --- End of modification ---


    best_loss = float('inf')
    best_model_path = None 

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for step, batch in enumerate(progress_bar):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            optimizer.zero_grad() # Zero gradients before forward pass

            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            
            if num_gpus > 1: # If using DataParallel, loss is a tensor with losses from each GPU
                loss = loss.mean()  # Take the mean of the losses
            
            loss.backward()
            total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            
            progress_bar.set_postfix({'loss': loss.item(), 'avg_epoch_loss': total_loss / (step + 1)})


        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch: {epoch + 1}, Average Training Loss: {avg_train_loss:.4f}")

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            # Ensure save_path directory exists
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            
            current_best_model_dir = os.path.join(args.save_path, f"best_model_epoch_{epoch+1}_loss_{avg_train_loss:.4f}")
            if not os.path.exists(current_best_model_dir):
                 os.makedirs(current_best_model_dir)

            print(f"New best model found. Saving to: {current_best_model_dir}")
            
            # If model is wrapped with DataParallel, save the underlying module
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(current_best_model_dir)
            tokenizer.save_pretrained(current_best_model_dir)
            best_model_path = current_best_model_dir # Update the path of the best model

    if best_model_path:
        print(f"Best model saved at: {best_model_path} with loss: {best_loss:.4f}")
    else:
        print("Training completed, but no model was saved as 'best' (e.g., if loss did not improve or only one epoch).")
        # Fallback: save the last state if no best model was explicitly saved due to no improvement
        # This part can be adjusted based on whether you always want to save the last model
        if num_epochs > 0:
            last_model_dir = os.path.join(args.save_path, f"last_model_epoch_{num_epochs}_loss_{avg_train_loss:.4f}")
            if not os.path.exists(last_model_dir):
                 os.makedirs(last_model_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(last_model_dir)
            tokenizer.save_pretrained(last_model_dir)
            print(f"Last model state saved to: {last_model_dir}")


if __name__ == "__main__":
    main()
