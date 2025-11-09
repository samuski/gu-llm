from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import default_data_collator

base = "meta-llama/Llama-3.2-1B-Instruct"
tok = AutoTokenizer.from_pretrained(base, use_fast=True)

# PAD = EOS for Llama
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id

def format_examples(example):
    text = tok.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
    enc = tok(
        text,
        max_length=384,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
    )
    # labels = input_ids, but ignore loss on PAD by setting -100 where mask==0
    labels = enc["input_ids"][:]
    amask = enc["attention_mask"]
    for i in range(len(labels)):
        if amask[i] == 0:
            labels[i] = -100
    enc["labels"] = labels
    return enc

ds = load_dataset("json", data_files={"train":"data/train.jsonl","val":"data/val.jsonl"})
ds = ds.map(format_examples, remove_columns=ds["train"].column_names, desc="Tokenizing")

# Plain fp16/bf16 load (1B fits easily); add quantization if you want QLoRA
model = AutoModelForCausalLM.from_pretrained(base, device_map={"": 0})
model.config.pad_token_id = tok.pad_token_id

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    model.config.attn_implementation = "flash_attention_2"
except Exception:
    model.config.attn_implementation = "sdpa"

try:
    model.generation_config.pad_token_id = tok.pad_token_id
except Exception:
    pass

# LoRA
lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"]
)
model = get_peft_model(model, lora_cfg)

args = TrainingArguments(
    output_dir="out",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=1.5e-4,
    num_train_epochs=2,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    optim="paged_adamw_8bit",
    eval_strategy="no",
    save_strategy="no",
    logging_steps=200,
    report_to="none",
    seed=42,
)

data_collator = default_data_collator

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    tokenizer=tok,
    data_collator=data_collator
)

trainer.train()
trainer.save_model("adapters")     # writes adapters (PEFT)
tok.save_pretrained("adapters")    # saves tokenizer alongside
