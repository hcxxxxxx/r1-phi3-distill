from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
# Load fine-tuned model
model = AutoModelForCausalLM.from_pretrained(
    "./phi-3-deepseek-finetuned",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained("./phi-3-deepseek-finetuned")
model.resize_token_embeddings(len(tokenizer))

# Create chat pipeline
chat_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

# Generate response
prompt = """<|user|>
which one is bigger, 9.11 or 9.9
<|end|>
<|assistant|>
"""

output = chat_pipeline(
    prompt,
    max_new_tokens=5000,
    temperature=0.7,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id
)

print(output[0]['generated_text'])
