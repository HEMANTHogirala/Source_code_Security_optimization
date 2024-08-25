import re
import os
import torch
from transformers import RobertaTokenizer, RobertaModel
import pickle

# Function to sanitize PHP code
def sanitize_code(code):
    code = re.sub(r'\/\*[\s\S]*?\*\/', '', code)  # Remove multiline comments
    code = re.sub(r'\/\/.*', '', code)  # Remove single line comments
    code = re.sub(r'#.*', '', code)  # Remove shell comments if included
    code = re.sub(r'\s*\n\s*', '\n', code)  # Remove extra whitespace around newlines
    return code.strip()

# Function to detect and label unsafe PHP code (0 for safe, 1 for unsafe)
def detect_unsafe_code(snippet):
    unsafe_patterns = {
        'eval(': ('Dynamic Code Execution', 'Avoid using `eval` for code execution. Use safer alternatives like direct function calls.'),
        'exec(': ('Command Injection', 'Avoid using `exec` for running system commands. Use safer PHP functions or validate and sanitize inputs.'),
        'shell_exec(': ('Command Injection', 'Do not use `shell_exec` for executing shell commands. Use safe alternatives and validate inputs.'),
        'system(': ('Command Injection', 'Avoid `system` for running system commands. Prefer safe functions and validate inputs.'),
        'popen(': ('Command Injection', 'Avoid using `popen` to run commands. Use safer alternatives and validate input.'),
        'fopen(': ('File Handling', 'Validate file paths properly to prevent directory traversal vulnerabilities.'),
        'file_get_contents(': ('File Handling', 'Sanitize file paths to prevent file inclusion vulnerabilities.'),
        'include(': ('File Inclusion', 'Avoid including files based on user input. Use predefined paths or sanitize inputs.'),
        'require(': ('File Inclusion', 'Do not include files based on user inputs. Use secure practices and predefined paths.'),
        'include_once(': ('File Inclusion', 'Prevent file inclusion based on user inputs. Use secure methods and predefined paths.'),
        'require_once(': ('File Inclusion', 'Avoid including files based on user inputs. Use secure paths and practices.'),
        'mysqli_query(': ('SQL Injection', 'Use prepared statements or parameterized queries instead of `mysqli_query` to avoid SQL injection.'),
        'PDO->query(': ('SQL Injection', 'Utilize PDO prepared statements to prevent SQL injection vulnerabilities.'),
        'unserialize(': ('Deserialization Vulnerability', 'Avoid using `unserialize` with untrusted data. Use safer serialization methods or ensure proper validation.'),
        'base64_decode(': ('Potential Data Exposure', 'Ensure that base64-decoded data is sanitized and validated properly.'),
        'preg_replace(': ('Regular Expression Injection', 'Avoid using user-controlled input in `preg_replace` without proper validation and escaping.'),
        'assert(': ('Dynamic Code Execution', 'Do not use `assert` for code execution. Use safer alternatives for debugging and validation.'),
        'create_function(': ('Dynamic Code Execution', 'Avoid `create_function` for dynamic code execution. Use modern and safer alternatives.')
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
model_file_path = os.path.join(script_directory, '../models/php_model.pkl')

# Load the trained model from the file
with open(model_file_path, 'rb') as model_file:
    classifier = pickle.load(model_file)

def evaluate_code(test_code):
    # Sanitize the code for embedding
    sanitized_code = sanitize_code(test_code)

    # Detect unsafe code patterns
    unsafe_code_lines = detect_unsafe_code(sanitized_code)

    # Tokenize and generate embeddings for the sanitized code
    tokenized_inputs = tokenizer([sanitized_code], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_outputs = model(**tokenized_inputs)
    code_embeddings = model_outputs.last_hidden_state.mean(dim=1).numpy()

    # Make predictions using the trained classifier
    prediction = classifier.predict(code_embeddings)
    
    return unsafe_code_lines, prediction
