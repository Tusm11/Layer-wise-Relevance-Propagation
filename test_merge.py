import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./merged-model-directory"

# Load your custom LRP-merged model
tokenizer = AutoTokenizer.from_pretrained(model_path)
# Load in float16 and automatically map to GPU/CPU to prevent memory crashes
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="auto", 
    torch_dtype=torch.float16
    
)

# Chat generation
messages = [
    {"role": "system", "content": "You are a helpful and intelligent AI assistant."},
    {"role": "user", "content": "What is photosynthesis?"}
]

# Apply the chat template
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate response
outputs = model.generate(
    **inputs, 
    max_new_tokens=50,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

# Decode only the newly generated assistant tokens
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(f"User: {messages[1]['content']}")
print(f"Assistant: {response.strip()}")
