from __future__ import annotations

from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


PROMPT_QUERY_EXPAND = """You rewrite search queries to improve retrieval.
Given a query, expand it with key context, synonyms, and specific entities.
Keep it as a concise search query (not an answer). 10-25 words.
Return ONLY the rewritten query, no quotes, no extra lines."""
# This string should be logged once at the top of generate-retrieve.txt.


@dataclass
class LlamaQueryExpander:
    model_id: str
    max_new_tokens: int = 64

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model.eval()

    @torch.inference_mode()
    def expand(self, query: str) -> str:
        messages = [
            {"role": "system", "content": PROMPT_QUERY_EXPAND},
            {"role": "user", "content": f"Query: {query}\nRewritten:"},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        if torch.cuda.is_available():
            input_ids = input_ids.to(self.model.device)

        out = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,          # deterministic
            temperature=0.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        gen = self.tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()

        # Keep only first non-empty line (models sometimes add extra)
        gen = next((ln.strip() for ln in gen.splitlines() if ln.strip()), "")
        return gen or query  # fallback: if weird empty output, use original
