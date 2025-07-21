from pathlib import Path
from llama_cpp import Llama
import multiprocessing

model_path = Path("./models/Meta-Llama-3-8B-Instruct.Q6_K.gguf")

optimal_n_threads = multiprocessing.cpu_count()
context_window_size = 8192
batch_size = 512
use_mlock = True
use_mmap = True

try:
    print(f"Loading model from: {model_path}")
    print(f"Using {optimal_n_threads} CPU threads for inference.")
    print(f"Context window size (n_ctx): {context_window_size} tokens.")
    print(f"Batch size (n_batch): {batch_size} tokens.")

    llm = Llama(
        model_path=str(model_path),
        n_ctx=context_window_size,
        n_threads=optimal_n_threads,
        n_batch=batch_size,
        use_mlock=use_mlock,
        use_mmap=use_mmap,
        verbose=False
    )
    print("Model loaded successfully! Type 'exit' to quit.")

except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")
    print("Ensure llama-cpp-python is correctly installed and compatible with your system.")
    exit()


#Chatting

messages = []

while True:
    user_prompt = input("You: ")
    if user_prompt.lower() == 'exit':
        print("Exiting chat. Goodbye!")
        break
    
    messages.append({"role": "user", "content": user_prompt})

    try:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=500,
            stop=["<|im_end|>", "</s>", "<|user|>", "You:"], 
            temperature=0.7 
        )
        
        assistant_reply = response["choices"][0]["message"]["content"].strip()
        print("Llama:", assistant_reply)
        
        messages.append({"role": "assistant", "content": assistant_reply})

    except Exception as e:
        print(f"An error occurred during inference: {e}")
        print("Please check your model and prompt. Resetting conversation history due to error.")
        messages = []