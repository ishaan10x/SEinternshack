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
    print("Enter your message to be masked or type 'exit' to quit.\n")
except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")
    print("Ensure llama-cpp-python is correctly installed and compatible with your system.")
    exit()

#Chatting

# Define the system prompt for redaction
SYSTEM_PROMPT = f"""Remove ALL sensitive information while keeping the text useful for AI assistance. This is for corporate use - strip anything that could identify people, companies, or leak business intelligence.
ALWAYS REPLACE WITH RANGES/GENERICS:
- ALL personal names → Person A, Person B, Person C (be consistent - don't miss any!)
- ALL company names → Company X, Company Y, Client Corp, etc.
- ALL email addresses → contact@example.com, client@example.com
- ALL phone numbers → 555-0000
- Exact dollar amounts → proper ranges
- Project names → Project Alpha, Project Beta
- System names → System A, Database X
- Account numbers, IDs → [ACCOUNT], [ID-123]
- Dates → [DATE], next week, Q4
- Titles with names → Person C (Role), etc.
REDACT:
- Client names and identifiers
- Internal team/department names
- Proprietary platform names
- IPs, URLs, physical addresses
- Contract/ticket/deal numbers
- Vendor names and internal process names
PRESERVE:
- Technical metrics
- Business requirements
- Job roles/titles
- General cities or timing
- Industry-standard concepts
Ensure consistency, readability, and complete privacy."""

messages = []

while True:
    user_prompt = input("You: ")
    if user_prompt.lower() == 'exit':
        print("Exiting chat. Goodbye!")
        break
    
    # Always start with the system prompt, then add user message
    # This ensures the system prompt is always at the beginning of the conversation for the model.
    messages_for_completion = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages_for_completion.extend(messages) # Add previous conversation history
    messages_for_completion.append({"role": "user", "content": user_prompt})


    try:
        stream = llm.create_chat_completion(
            messages=messages_for_completion, # Use the combined messages including system prompt
            max_tokens=500,
            stop=["<|im_end|>", "</s>", "<|user|>", "You:"],
            temperature=0.7,
            stream=True
        )

        assistant_reply = ""
        print("Llama: ", end="", flush=True)
        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                print(delta["content"], end="", flush=True)
                assistant_reply += delta["content"]
        print()

        # Append only the user and assistant messages to the main conversation history
        # The system prompt is prepended for each completion call but not stored in `messages`
        messages.append({"role": "user", "content": user_prompt}) # Add user prompt to history
        messages.append({"role": "assistant", "content": assistant_reply.strip()}) # Add assistant reply to history

    except Exception as e:
        print(f"An error occurred during inference: {e}")
        print("Please check your model and prompt. Resetting conversation history due to error.")
        messages = [] # Reset messages on error