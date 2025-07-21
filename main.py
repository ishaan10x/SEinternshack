import os
os.environ['LLAMA_SET_ROWS'] = '1'

from pathlib import Path
from llama_cpp import Llama
import multiprocessing

from flask import Flask, request, jsonify
from flask_cors import CORS

# model setup
model_path = Path("./models/Meta-Llama-3-8B-Instruct.Q6_K.gguf")

print("Loading model from", model_path)
print("Threads", multiprocessing.cpu_count(), "context 8192 batch 512")

llm = Llama(
    model_path = str(model_path),
    n_ctx = 8192,
    n_threads = multiprocessing.cpu_count(),
    n_batch = 512,
    use_mlock = True,
    use_mmap = True,
    verbose = False
)

# system prompt for redaction
SYSTEM_PROMPT = """Remove ALL sensitive information while keeping the text useful for AI assistance. This is for corporate use - strip anything that could identify people, companies, or leak business intelligence.
ALWAYS REPLACE WITH RANGES/GENERICS:
- ALL personal names → Person A, Person B, Person C be consistent
- ALL company names → Company X Company Y Client Corp etc
- ALL email addresses → contact@example.com client@example.com
- ALL phone numbers → 555-0000
- Exact dollar amounts → proper ranges
- Project names → Project Alpha Project Beta
- System names → System A Database X
- Account numbers IDs → [ACCOUNT] [ID-123]
- Dates → [DATE] next week Q4
- Titles with names → Person C Role etc
REDACT:
- Client names and identifiers
- Internal team department names
- Proprietary platform names
- IPs URLs physical addresses
- Contract ticket deal numbers
- Vendor names and internal process names
PRESERVE:
- Technical metrics
- Business requirements
- Job roles titles
- General cities or timing
- Industry-standard concepts
Consistency readability and privacy"""

app = Flask(__name__)
CORS(app)

@app.route('/prompt', methods=['POST'])
def handle_prompt():
    data = request.json
    prompt = data.get('prompt', '').strip()
    print("Received prompt", prompt)

    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': prompt}
    ]

    try:
        resp = llm.create_chat_completion(
            messages = messages,
            max_tokens = 1000,
            stop = ["<|im_end|>", "</s>", "<|user|>", "You:"],
            temperature = 0.7
        )
        reply = resp['choices'][0]['message']['content'].strip()
    except Exception as e:
        print("Inference error", e)
        reply = f"Error: {e}"

    print("Model response", reply)
    return jsonify({'response': reply})

if __name__ == '__main__':
    app.run(port = 5000)