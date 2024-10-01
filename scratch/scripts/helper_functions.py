import json
import urllib.request
from datasets import load_dataset
import os

def download_squad_deprecated(version='v1.1'):
    """
    Download SQuAD dataset JSON files.
    """
    base_url = f"https://rajpurkar.github.io/SQuAD-explorer/dataset/train-{version}.json"
    dev_url = f"https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-{version}.json"

    print(f"Downloading SQuAD {version} training set...")
    urllib.request.urlretrieve(base_url, f"train-{version}.json")
    print(f"Downloading SQuAD {version} dev set...")
    urllib.request.urlretrieve(dev_url, f"dev-{version}.json")

def download_squad(version='v1.1'):
    """
    Download SQuAD dataset JSON files if they don't exist locally.
    """
    base_url = f"https://rajpurkar.github.io/SQuAD-explorer/dataset/train-{version}.json"
    dev_url = f"https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-{version}.json"
    
    train_file = f"train-{version}.json"
    dev_file = f"dev-{version}.json"
    
    if not os.path.exists(train_file):
        print(f"Downloading SQuAD {version} training set...")
        urllib.request.urlretrieve(base_url, train_file)
    else:
        print(f"SQuAD {version} training set already exists. Skipping download.")
    
    if not os.path.exists(dev_file):
        print(f"Downloading SQuAD {version} dev set...")
        urllib.request.urlretrieve(dev_url, dev_file)
    else:
        print(f"SQuAD {version} dev set already exists. Skipping download.")

def load_squad_json(file_path):
    """
    Load SQuAD JSON file and return the data.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def load_squad_datasets():
    """
    Load SQuAD v1.1 and v2.0 datasets using the datasets library.
    """
    squad_v1 = load_dataset("squad")
    squad_v2 = load_dataset("squad_v2")
    return squad_v1, squad_v2

def get_squad():
    # Download SQuAD v1.1 and v2.0 JSON files

    download_squad('v1.1')
    download_squad('v2.0')

    # Load downloaded JSON files
    squad_v1_train = load_squad_json('train-v1.1.json')
    squad_v1_dev = load_squad_json('dev-v1.1.json')
    squad_v2_train = load_squad_json('train-v2.0.json')
    squad_v2_dev = load_squad_json('dev-v2.0.json')

    print("SQuAD v1.1 and v2.0 JSON files loaded successfully.")

    # Load datasets using the datasets library
    squad_v1, squad_v2 = load_squad_datasets()

    print("SQuAD v1.1 and v2.0 datasets loaded successfully using the datasets library.")
    print(f"SQuAD v1.1 train size: {len(squad_v1['train'])}")
    print(f"SQuAD v2.0 train size: {len(squad_v2['train'])}")

    return squad_v1, squad_v2

# helper_functions.py

import os
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import math

# Candidate models:
# timpal0l/mdeberta-v3-base-squad2
# deepset/roberta-base-squad2
# deepset/tinyroberta-squad2

def setup_qa_model(model_name="deepset/roberta-base-squad2", save_directory="./model_cache"):
    """
    Download or load the model and tokenizer, and set up a question-answering pipeline.
    
    Args:
    model_name (str): The name of the model to use from HuggingFace.
    save_directory (str): The directory to save/load the model and tokenizer.
    
    Returns:
    tuple: A question-answering pipeline and the raw model and tokenizer for detailed analysis.
    """
    def download_or_load_model(model_name, save_directory):
        """
        Download the model and tokenizer if not present locally, or load them from local directory.
        """
        model_path = os.path.join(save_directory, "model")
        tokenizer_path = os.path.join(save_directory, "tokenizer")

        if not os.path.exists(model_path):
            print(f"Downloading model {model_name}...")
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            model.save_pretrained(model_path)
        else:
            print("Loading model from local directory...")
            model = AutoModelForQuestionAnswering.from_pretrained(model_path)

        if not os.path.exists(tokenizer_path):
            print(f"Downloading tokenizer for {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(tokenizer_path)
        else:
            print("Loading tokenizer from local directory...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        return model, tokenizer

    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Download or load the model and tokenizer
    model, tokenizer = download_or_load_model(model_name, save_directory)

    # Create a question-answering pipeline
    qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)

    return qa_pipeline, model, tokenizer

def analyze_qa_output(model, tokenizer, question, context, max_length=512, debug=False):
    """
    Analyze the output of the question-answering model.
    
    Args:
    model: The QA model.
    tokenizer: The tokenizer for the model.
    question: The question string.
    context: The context string.
    max_length: Maximum sequence length for the model (default 612).
    debug: Extended debug information (default False).
    
    Returns:
    dict: A dictionary containing the analysis results.
    """

    if debug:
        # Debug prints for word counts
        print(f"Number of words in question: {len(question.split())}")
        print(f"Number of words in context: {len(context.split())}")

        # Debug prints for token counts
        print(f"Number of tokens in question: {len(tokenizer.tokenize(question))}")
        print(f"Number of tokens in context: {len(tokenizer.tokenize(context))}")

    inputs = tokenizer(question, context, return_tensors="pt", max_length=max_length, truncation=True)

    if debug:
        print("Keys in the BatchEncoding object:", inputs.keys())
        print("Shape of input_ids:", inputs['input_ids'].shape)
        print("Shape of attention_mask:", inputs['attention_mask'].shape)

    if debug:
        # Decode the entire sequence
        decoded_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        # Print each token ID with its corresponding text
        for id, token in zip(inputs['input_ids'][0], decoded_tokens):
            print(f"ID: {id:5d} -> Token: {token}")

        # ID:     0 -> Token: <s> # start of sequence token
        # ID:  1121 -> Token: In
        # ID:    99 -> Token: Ġwhat # "Ġ" character (Unicode U+0120) is used to represent the beginning of a word that follows a space in the original text
        # ID:   247 -> Token: Ġcountry            

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # Get the most likely beginning and end of answer
    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits)

    # Convert to probabilities
    start_probs = torch.softmax(start_logits, dim=1)[0]
    end_probs = torch.softmax(end_logits, dim=1)[0]

    start_prob = start_probs[start_index].item()
    end_prob = end_probs[end_index].item()

    # Compute score (product of probabilities)
    score = start_prob * end_prob

    # Calculate entropy: -Σ(p * log(p))
    start_entropy = -(start_probs * torch.log(start_probs)).sum().item()
    end_entropy = -(end_probs * torch.log(end_probs)).sum().item()

    # Calculate combined entropy using Root Mean Square
    combined_rms_entropy = math.sqrt((start_entropy**2 + end_entropy**2) / 2)

    # Calculate normalized entropy
    max_entropy = math.log(len(inputs['input_ids'][0]))
    normalized_rms_entropy = combined_rms_entropy / (max_entropy / math.sqrt(2))

    if debug:
        print(f"Start entropy: {start_entropy}")
        print(f"End entropy: {end_entropy}")
        print(f"Combined rms entropy: {combined_rms_entropy}")
        print(f"Normalized rms entropy: {normalized_rms_entropy}")

    # Get the answer span
    all_tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    answer = tokenizer.convert_tokens_to_string(all_tokens[start_index:end_index+1])

    if debug:
        print(f"Number of tokens in input: {len(all_tokens)}")

    return {
        "answer": answer,
        "score": score,
        "start_probability": start_prob,
        "end_probability": end_prob,
        "start_index": start_index.item(),
        "end_index": end_index.item(),
        "start_probs": start_probs.tolist(),
        "end_probs": end_probs.tolist(),
        "tokens": all_tokens,
        "start_entropy": start_entropy,
        "end_entropy": end_entropy,
        "combined_entropy": combined_rms_entropy,
        "normalized_entropy": normalized_rms_entropy
    }

import numpy as np
from collections import Counter

def analyze_squad(qa_pipeline, model, tokenizer, dataset, num_samples=None):
    """
    Analyze a subset of the SQuAD dataset using the QA model and compute metrics.
    
    Args:
    qa_pipeline: The question-answering pipeline.
    model: The raw model for detailed analysis.
    tokenizer: The tokenizer for the model.
    dataset: The SQuAD dataset to analyze.
    num_samples: Number of samples to analyze (default None, which uses all samples).
    
    Returns:
    tuple: A tuple containing:
        - list: A list of dictionaries containing the analysis results.
        - dict: A dictionary containing the computed metrics.
    """
    results = []
    metrics = Counter()
    
    # If num_samples is None, use all samples in the dataset
    if num_samples is None:
        num_samples = dataset.num_rows
    
    # Iterate through the dataset
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        
        context = example['context']
        question = example['question']
        answer = example['answers']['text'][0] if example['answers']['text'] else ""
        
        # Get predictions using the pipeline
        pipeline_result = qa_pipeline(question=question, context=context)
        
        # Get detailed analysis
        detailed_result = analyze_qa_output(model, tokenizer, question, context)
        
        # Combine results
        combined_result = {
            'id': example['id'],
            'title': example['title'],
            'context': context,
            'question': question,
            'answers': example['answers'],
            'pipeline_result': pipeline_result,
            'detailed_result': detailed_result
        }
        
        results.append(combined_result)
        
        # Compute metrics
        predicted_answer = pipeline_result['answer']
        exact_match = int(predicted_answer == answer)
        f1_score = compute_f1(predicted_answer, answer)
        
        metrics['total'] += 1
        metrics['exact'] += exact_match
        metrics['f1'] += f1_score
        
        if answer:  # HasAns case
            metrics['HasAns_total'] += 1
            metrics['HasAns_exact'] += exact_match
            metrics['HasAns_f1'] += f1_score
        else:  # NoAns case
            metrics['NoAns_total'] += 1
            metrics['NoAns_exact'] += exact_match
            metrics['NoAns_f1'] += f1_score
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{num_samples} samples")
    
    # Compute final metrics
    for key in ['exact', 'f1', 'HasAns_exact', 'HasAns_f1', 'NoAns_exact', 'NoAns_f1']:
        if key.startswith('HasAns'):
            total = metrics['HasAns_total']
        elif key.startswith('NoAns'):
            total = metrics['NoAns_total']
        else:
            total = metrics['total']
        metrics[key] = (metrics[key] / total) * 100 if total > 0 else 0
    
    return results, dict(metrics)

def compute_f1(prediction, ground_truth):
    """Compute F1 score between prediction and ground truth."""
    prediction_tokens = prediction.lower().split()
    ground_truth_tokens = ground_truth.lower().split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1