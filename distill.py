import os

# Disable P2P and IB
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

from datasets import load_dataset
# Load the dataset
dataset = load_dataset("Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B", split="train", token="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", cache_dir="./Hcx")

#dataset = dataset["train"]
# Split dataset into 90% train, 10% test
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

# Access the new splits
train_ds = split_dataset["train"]
test_ds = split_dataset["test"]

###
train_ds = train_ds.select(range(int(len(train_ds) * 0.01)))
test_ds = test_ds.select(range(int(len(test_ds) * 0.01)))
###

# Print sizes
print(f"Train size: {len(train_ds)}")
print(f"Test size: {len(test_ds)}")
# Format the dataset
#def format_instruction(example):   
#    return { "text": (            "<|user|>\n"            f"{example['instruction']}\n"            "<|end|>\n"            "<|assistant|>\n"            f"{example['response']}\n"            "<|end|>"        )    }
#formatted_dataset = dataset.map(format_instruction, batched=False, remove_columns=dataset.column_names)
#formatted_dataset = formatted_dataset.train_test_split(test_size=0.1)  # 90-10 train-test split

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "microsoft/phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, cache_dir="./Hcx")

# Add custom tokens
CUSTOM_TOKENS = ["<think>", "</think>"]
tokenizer.add_special_tokens({"additional_special_tokens": CUSTOM_TOKENS})
tokenizer.pad_token = tokenizer.eos_token

###
def format(ex):
    system_message=f"""
    You are a master at {ex['task_category']}. Based on the question asked, answer the question to the best of your abilities
    """
    prompt = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Answer this question:\n{ex['instruction']}"}
            ]
    prompt=tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, add_special_tokens=False)
    full= [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Answer this question:\n{ex['instruction']}"},
                {"role":"assistant", "content": ex['response']}
            ]
    
    resp=ex['response']
    return {'full': full, 'query': prompt, 'response': resp}
    
    
from trl import DataCollatorForCompletionOnlyLM    

train_ds=train_ds.map(format)


train_dataset=train_ds.remove_columns(['instruction', 'conversations', 'gen_input_configs', 'gen_response_configs', 'intent', 'knowledge', 'difficulty', 'difficulty_generator', 'input_quality', 'quality_explanation', 'quality_generator', 'task_category', 'other_task_category', 'task_category_generator', 'language', 'full'])

split_dataset = train_dataset.train_test_split(test_size=0.1, seed=50)

instruction_template = "<｜User｜>"
response_template = "<｜Assistant｜>"
collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template,response_template=response_template, tokenizer=tokenizer)

###

# Load model with flash attention
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=False,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    cache_dir="./Hcx"
)
model.resize_token_embeddings(len(tokenizer))  # Resize for custom tokens

from peft import LoraConfig

peft_config = LoraConfig(
    r=8,  # Rank of the low-rank matrices
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.2,  # Dropout rate
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target attention layers
    bias="none",  # No bias terms
    task_type="CAUSAL_LM"  # Task type
)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./phi-3-deepseek-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    learning_rate=2e-5,
#    fp16=True,
    fp16=False,
    optim="paged_adamw_32bit",
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
)

num_training_steps = (len(train_dataset) // 32) * 3

from trl import SFTTrainer
from transformers import DataCollatorForLanguageModeling

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


from transformers import AdamW, get_scheduler

# 创建 AdamW 优化器
optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

# 创建学习率调度器（例如线性调度器）
lr_scheduler = get_scheduler(
    name="cosine",  # 可以选择其他类型的调度器
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=training_args.max_steps
)

# Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset['train'],
    eval_dataset=split_dataset['test'],
    data_collator=collator,  # Your custom data collator
    tokenizer=tokenizer,
    dataset_text_field="query",
    optimizers=(optimizer, lr_scheduler)
#    train_dataset=formatted_dataset["train"],
#    eval_dataset=formatted_dataset["test"],
#    data_collator=data_collator,
#    packing=True
    )

# Start training
trainer.train()
trainer.save_model("./phi-3-deepseek-finetuned")
tokenizer.save_pretrained("./phi-3-deepseek-finetuned")


#final_model = trainer.model.merge_and_unload()
#final_model.save_pretrained("./phi-3-deepseek-finetuned-final")
#tokenizer.save_pretrained("./phi-3-deepseek-finetuned-final")

