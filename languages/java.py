import re
import os
import torch
from transformers import RobertaTokenizer, RobertaModel
import pickle

# Function to sanitize Java code
def sanitize_code(code):
    code = re.sub(r'\/\*[\s\S]*?\*\/', '', code)  # Remove multiline comments
    code = re.sub(r'\/\/.*', '', code)  # Remove single line comments
    code = re.sub(r'\s*\n\s*', '\n', code)  # Remove extra whitespace around newlines
    return code.strip()

# Function to detect and label unsafe Java code (0 for safe, 1 for unsafe)
def detect_unsafe_code(snippet):
    unsafe_patterns = {
        'Runtime.getRuntime().exec': ('Command Execution', 'Avoid using `exec` for executing system commands. Use safer alternatives and validate inputs.'),
        'ProcessBuilder': ('Command Execution', 'Avoid using `ProcessBuilder` for executing system commands. Prefer safer methods and validate inputs.'),
        'FileOutputStream': ('File I/O', 'Ensure proper validation of file paths to avoid directory traversal vulnerabilities.'),
        'FileInputStream': ('File I/O', 'Sanitize input when handling file paths to prevent unauthorized file access.'),
        'BufferedWriter': ('File I/O', 'Validate file paths and handle I/O operations securely.'),
        'BufferedReader': ('File I/O', 'Ensure file paths are validated before use to prevent unauthorized access.'),
        'ObjectInputStream': ('Deserialization', 'Avoid using `ObjectInputStream` with untrusted data. Use safer deserialization methods or validate inputs.'),
        'ObjectOutputStream': ('Serialization', 'Ensure serialized data is properly handled and validated.'),
        'setAccessible': ('Reflection', 'Avoid using `setAccessible` to bypass access control. Use secure and proper access mechanisms.'),
        'getDeclaredField': ('Reflection', 'Avoid using reflection to access private fields. Use secure methods for accessing fields.'),
        'getDeclaredMethod': ('Reflection', 'Avoid using reflection to access private methods. Prefer direct method calls.'),
        'URLClassLoader': ('URL Class Loading', 'Avoid loading classes from untrusted sources. Use secure class loading practices.'),
        'URLConnection': ('Network Communication', 'Validate and sanitize network inputs to prevent vulnerabilities.'),
        'Socket': ('Network Communication', 'Ensure network sockets are used securely and inputs are validated.'),
        'ServerSocket': ('Network Communication', 'Use secure practices for network communication and validate inputs.'),
        'printStackTrace': ('Information Disclosure', 'Avoid using `printStackTrace` for error handling in production. Use proper logging mechanisms.'),
        'getSystemProperty': ('Information Disclosure', 'Avoid disclosing system properties that may reveal sensitive information.'),
        'System.loadLibrary': ('Library Loading', 'Ensure libraries are loaded from trusted sources to avoid security issues.'),
        'Class.forName': ('Dynamic Class Loading', 'Avoid dynamic class loading from untrusted sources. Use secure class loading practices.'),
        'Method.invoke': ('Reflection', 'Avoid using reflection for invoking methods. Use secure alternatives for method invocations.'),
        'ObjectInputStream.readObject': ('Deserialization', 'Avoid reading objects from untrusted sources. Use secure deserialization practices.'),
        'DriverManager.getConnection': ('SQL Injection', 'Use prepared statements instead of `DriverManager.getConnection` to prevent SQL injection.'),
        'Statement.execute': ('SQL Injection', 'Use parameterized queries to prevent SQL injection vulnerabilities.'),
        'ResultSet.getString': ('SQL Injection', 'Validate and sanitize data retrieved from SQL queries to prevent injection attacks.'),
        'MessageDigest.getInstance("MD5")': ('Insecure Hashing', 'Avoid using MD5 for hashing. Use stronger algorithms like SHA-256.'),
        'MessageDigest.getInstance("SHA1")': ('Insecure Hashing', 'Use more secure hashing algorithms such as SHA-256.'),
        'Random': ('Insecure Randomness', 'Avoid using `Random` for cryptographic purposes. Use `SecureRandom` for secure random number generation.'),
        'Math.random': ('Insecure Randomness', 'Do not use `Math.random` for security-sensitive applications. Use `SecureRandom` instead.'),
        'Cipher.getInstance("DES")': ('Insecure Encryption', 'Avoid using DES for encryption. Use more secure algorithms like AES.'),
        'Cipher.getInstance("Blowfish")': ('Insecure Encryption', 'Consider using AES or other secure encryption algorithms instead of Blowfish.'),
        'SSLContext': ('SSL Context', 'Ensure proper configuration of SSL contexts to prevent security issues.'),
        'TrustManager': ('SSL Trust Management', 'Validate and secure trust managers to prevent vulnerabilities.'),
        'HostnameVerifier': ('SSL Hostname Verification', 'Ensure proper hostname verification in SSL/TLS contexts to prevent spoofing.'),
        'new File': ('File Creation', 'Validate and sanitize file paths before creating files to avoid security risks.'),
        'XStream': ('XML Serialization', 'Use secure XML serialization methods and validate inputs to prevent vulnerabilities.'),
        'JavaSerializer': ('Java Serialization', 'Ensure safe handling of serialized data and avoid insecure deserialization practices.'),
        'System.out.println': ('Information Disclosure', 'Avoid using `System.out.println` for sensitive information. Use secure logging practices.')
    }

    unsafe_lines = []
    for line in snippet.split('\n'):
        for pattern, (vuln_type, recommendation) in unsafe_patterns.items():
            if pattern in line:
                unsafe_lines.append((line.strip(), vuln_type, recommendation))
    return unsafe_lines

# Load pre-trained model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
model = RobertaModel.from_pretrained('microsoft/codebert-base')

# Determine the path for the trained model
script_directory = os.path.dirname(__file__)
model_file_path = os.path.join(script_directory, '../models/java_model.pkl')

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
