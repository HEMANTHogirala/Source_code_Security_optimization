#!/usr/bin/env python
# coding: utf-8

import re
import json
import os
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel
from sklearn.ensemble import RandomForestClassifier
import pickle

print("Initializing...")

# Function to preprocess Python code
def preprocess_python_code(code):
    code = re.sub(r'\/\*[\s\S]*?\*\/', '', code)  # Remove multiline comments
    code = re.sub(r'\/\/.*', '', code)  # Remove single line comments
    code = re.sub(r'#.*', '', code)  # Remove Python comments
    code = re.sub(r'\s*\n\s*', '\n', code)  # Remove extra whitespace around newlines
    return code.strip()

print("Loading model and tokenizer...")

# Load pre-trained model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
model = RobertaModel.from_pretrained('microsoft/codebert-base')

# Path to the JSON file containing the dataset
json_file_path = r'D:\source_code_opt\datasets\python.json'

print("Loading dataset...")

# Load and preprocess the dataset
with open(json_file_path, 'r') as file:
    dataset = json.load(file)

print("Tokenizing and embedding code snippets...")

# Tokenize and get embeddings for the code snippets
tokenized_inputs = tokenizer([preprocess_python_code(item['code']) for item in dataset], padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    model_outputs = model(**tokenized_inputs)
embeddings = model_outputs.last_hidden_state.mean(dim=1).numpy()

print("Creating labels...")

# Create labels from the dataset
labels = np.array([item['label'] for item in dataset])

print("Training classifier...")

# Train a RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(embeddings, labels)

print("Saving trained model...")

# Save the trained model
model_dir = r'D:\source_code_opt\models'
model_filename = 'python_model.pkl'
model_path = os.path.join(model_dir, model_filename)

with open(model_path, 'wb') as model_file:
    pickle.dump(classifier, model_file)

print("Training complete. Model saved to:", model_path)
