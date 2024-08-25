import re
import os
import torch
from transformers import RobertaTokenizer, RobertaModel
import pickle

# Function to sanitize Python code
def sanitize_code(code):
    code = re.sub(r'\/\*[\s\S]*?\*\/', '', code)  # Remove multiline comments
    code = re.sub(r'\/\/.*', '', code)  # Remove single line comments
    code = re.sub(r'#.*', '', code)  # Remove Python comments
    code = re.sub(r'\s*\n\s*', '\n', code)  # Remove extra whitespace around newlines
    return code.strip()

# Function to detect and label unsafe Python code (0 for safe, 1 for unsafe)
def detect_unsafe_code(snippet):
    unsafe_patterns = {
        'eval(': ('Dynamic Code Execution', 'Use safer alternatives like `literal_eval` from `ast` for evaluation.'),
        'exec(': ('Dynamic Code Execution', 'Avoid `exec` for dynamic execution; use function calls or classes instead.'),
        'subprocess.call(': ('Command Injection', 'Use `subprocess.run` with `shell=False` to avoid command injection.'),
        'subprocess.Popen(': ('Command Injection', 'Avoid using `shell=True` in subprocesses.'),
        'input(': ('Unvalidated Input', 'Validate and sanitize user inputs to prevent vulnerabilities.'),
        'open(': ('File Handling', 'Use `with open(...) as file:` to ensure proper file closure.'),
        'os.system(': ('Command Injection', 'Use `subprocess.run` instead of `os.system` for better security.'),
        'pickle.load(': ('Deserialization Vulnerability', 'Avoid untrusted data sources or use `safe_load` with YAML.'),
        'pickle.dumps(': ('Serialization Vulnerability', 'Ensure data integrity when serializing with `pickle`.'),
        'import(': ('Dynamic Import', 'Avoid using dynamic imports; use static imports at the top of the file.'),
        'os.getenv(': ('Environment Variable Exposure', 'Use safe defaults or restrict sensitive environment variables.'),
        'glob.glob(': ('File Exposure', 'Restrict file pattern matching to trusted paths.'),
        'shutil.copy(': ('File Handling', 'Ensure proper permissions and file existence checks when copying files.'),
        'shutil.move(': ('File Handling', 'Check file paths and existence before moving files.'),
        'sqlite3.connect(': ('SQL Injection', 'Use parameterized queries to prevent SQL injection.'),
        'pymysql.connect(': ('SQL Injection', 'Always use parameterized queries for MySQL.'),
        'psycopg2.connect(': ('SQL Injection', 'Utilize parameterized queries for PostgreSQL.'),
        'requests.get(': ('Potential Data Exposure', 'Use SSL/TLS for secure data transmission.'),
        'requests.post(': ('Potential Data Exposure', 'Ensure data is encrypted and validate server certificates.')
    }
    unsafe_lines = []
    for line in snippet.split('\n'):
        for pattern, (vuln_type, recommendation) in unsafe_patterns.items():
            if pattern in line:
                unsafe_lines.append((line.strip(), vuln_type, recommendation))
    return unsafe_lines

# Initialize pre-trained model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
model = RobertaModel.from_pretrained('microsoft/codebert-base')

# Determine the path for the trained model
script_directory = os.path.dirname(__file__)
model_file_path = os.path.join(script_directory, '../models/python_model.pkl')

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
