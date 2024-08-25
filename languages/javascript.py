import re
import os
import torch
from transformers import RobertaTokenizer, RobertaModel
import pickle

# Function to sanitize JavaScript code
def sanitize_code(code):
    code = re.sub(r'\/\*[\s\S]*?\*\/', '', code)  # Remove multiline comments
    code = re.sub(r'\/\/.*', '', code)  # Remove single line comments
    code = re.sub(r'#.*', '', code)  # Remove any mistakenly included Python comments
    code = re.sub(r'\s*\n\s*', '\n', code)  # Eliminate extra whitespace around newlines
    return code.strip()

# Function to detect and label unsafe JavaScript code (0 for safe, 1 for unsafe)
def identify_unsafe_code(snippet):
    unsafe_patterns = {
        'eval': ('Code Injection', 'Avoid using `eval` for code execution. Use safer alternatives like the `Function` constructor or other execution methods.'),
        'innerHTML': ('Cross-Site Scripting (XSS)', 'Sanitize and escape user input before inserting it into the DOM.'),
        'outerHTML': ('Cross-Site Scripting (XSS)', 'Sanitize and escape user input before setting HTML content.'),
        'document.write': ('Cross-Site Scripting (XSS)', 'Avoid `document.write` as it can overwrite the entire document. Use DOM manipulation methods instead.'),
        'setTimeout': ('Potential Code Injection', 'Be cautious with user-provided input. Prefer direct function calls over evaluating strings.'),
        'setInterval': ('Potential Code Injection', 'Validate user input before using it in functions.'),
        'Function': ('Code Injection', 'Avoid the `Function` constructor for dynamic code execution. Use predefined functions or safer methods.'),
        'location.href': ('Open Redirect', 'Validate and sanitize URLs before redirecting to prevent open redirects.'),
        'document.location': ('Open Redirect', 'Ensure URLs are validated and not influenced by user input.'),
        'XMLHttpRequest': ('Sensitive Data Exposure', 'Ensure secure data handling and use HTTPS for requests.'),
        '<script>': ('Cross-Site Scripting (XSS)', 'Sanitize and escape dynamically inserted content to prevent XSS attacks.'),
        'document.body.innerHTML': ('Cross-Site Scripting (XSS)', 'Avoid setting `innerHTML` directly from user input without proper sanitization.')
    }

    unsafe_lines = []
    for line in snippet.split('\n'):
        for pattern, (vuln_type, advice) in unsafe_patterns.items():
            if pattern in line:
                unsafe_lines.append((line.strip(), vuln_type, advice))
    return unsafe_lines

# Initialize pre-trained model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
model = RobertaModel.from_pretrained('microsoft/codebert-base')

# Determine the path for the trained model
script_directory = os.path.dirname(__file__)
model_file_path = os.path.join(script_directory, '../models/javascript_model.pkl')

# Load the pre-trained model
with open(model_file_path, 'rb') as model_file:
    classifier = pickle.load(model_file)

def evaluate_code(test_code):
    # Sanitize the code for embedding
    sanitized_code = sanitize_code(test_code)

    # Detect unsafe code patterns
    unsafe_code_lines = identify_unsafe_code(sanitized_code)

    # Tokenize and generate embeddings for the sanitized code
    tokenized_inputs = tokenizer([sanitized_code], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_outputs = model(**tokenized_inputs)
    code_embeddings = model_outputs.last_hidden_state.mean(dim=1).numpy()

    # Make predictions using the trained classifier
    prediction = classifier.predict(code_embeddings)
    
    return unsafe_code_lines, prediction
