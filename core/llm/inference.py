# core/llm/inference.py
import os, torch
import re
from django.conf import settings
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

_MODEL = _TOK = None

def _local_ready(path):
    if not path or not os.path.isdir(path):
        return False
    for f in ("tokenizer_config.json", "config.json", "generation_config.json",
              "model.safetensors", "pytorch_model.bin"):
        if os.path.exists(os.path.join(path, f)):
            return True
    return False

def _load():
    global _MODEL, _TOK
    if _MODEL is not None:
        return

    local = getattr(settings, "LLAMA_PATH", None)
    src = local if _local_ready(local) else os.getenv("HF_MODEL_REPO")
    if not src:
        raise RuntimeError("No model available")

    _TOK = AutoTokenizer.from_pretrained(src, use_fast=True)
    if _TOK.pad_token is None:
        _TOK.pad_token = _TOK.eos_token
        _TOK.pad_token_id = _TOK.eos_token_id

    _MODEL = AutoModelForCausalLM.from_pretrained(
        src,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    _MODEL.config.pad_token_id = _TOK.pad_token_id

    # -------- LoRA adapters (comment out this block to use base model only) --------
    adapters_path = getattr(settings, "ADAPTERS_PATH", None) or os.getenv("ADAPTERS_PATH")
    if adapters_path and os.path.isdir(adapters_path):
        print(f"[llm] Loading LoRA adapters from: {adapters_path}")
        _MODEL = PeftModel.from_pretrained(_MODEL, adapters_path)
    else:
        print("[llm] No adapters found; using base model only")
    # ------------------------------------------------------------------------------    

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    _MODEL.eval()

def generate(messages, max_new_tokens=1024, temperature=0.7, top_p=0.9):
    _load()
    prompt = _TOK.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _TOK(prompt, return_tensors="pt").to(_MODEL.device)

    with torch.no_grad():
        gen = _MODEL.generate(
            **inputs,
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            max_new_tokens=int(max_new_tokens),
            eos_token_id=_TOK.eos_token_id,
            pad_token_id=_TOK.eos_token_id,
        )

    out_tokens = gen[0, inputs["input_ids"].shape[1]:]
    text = _TOK.decode(out_tokens, skip_special_tokens=True)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
