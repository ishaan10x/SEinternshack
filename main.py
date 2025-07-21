# Use new KV cache scheme
import os
os.environ['LLAMA_SET_ROWS'] = '1'

from pathlib import Path
from llama_cpp import Llama
import multiprocessing

model_path = Path("./models/Meta-Llama-3-8B-Instruct.Q6_K.gguf")

optimal_n_threads = multiprocessing.cpu_count()
context_window_size = 8192
batch_size = 512

try:
    print(f"Loading model from: {model_path}")
    print(f"Using {optimal_n_threads} CPU threads for inference.")
    print(f"Context window size (n_ctx): {context_window_size} tokens.")
    print(f"Batch size (n_batch): {batch_size} tokens.")

    llm = Llama(
        model_path=str(model_path),
        n_ctx= context_window_size,
        n_threads= optimal_n_threads,
        n_batch= batch_size,
        use_mlock= True,
        use_mmap= True,
        verbose= False
    )
except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")
    print("Ensure llama-cpp-python is correctly installed and compatible with your system.")
    exit()

SYSTEM_PROMPT = "You are a helpful assistant that redacts personal information from user messages. Your task is to identify and replace any names, email addresses, and phone numbers with '[REDACTED]'. The rest of the message should remain exactly the same. If no personal information is found, return the original message unchanged."

while True:
    user_prompt = input("\nYou: ")
    
    messages_for_completion = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    try:
        stream = llm.create_chat_completion(
            messages=messages_for_completion,
            max_tokens=1000,
            stop=["<|im_end|>", "</s>", "<|user|>", "You:"],
            temperature=0.7,
            stream=True
        )

        assistant_reply = ""
        print("\nTrust Layer: ", end="", flush=True)
        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                print(delta["content"], end="", flush=True)
                assistant_reply += delta["content"]
        print()

    except Exception as e:
        print(f"An error occurred during inference: {e}")