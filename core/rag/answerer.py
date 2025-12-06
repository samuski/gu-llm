from __future__ import annotations

from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPT_RETRIEVE_GENERATE = """You answer questions using ONLY the provided retrieved passages.
If the passages do not contain enough information to answer, say: "I don't know based on the provided passages."
"""


@dataclass
class LlamaAnswerer:
    model_id: str
    max_new_tokens: int = 256

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
    def answer(self, question: str, passages: list[tuple[str, str]]) -> str:
        """
        passages: list of (doc_id, text)
        """
        ctx_lines = []
        for i, (doc_id, text) in enumerate(passages, start=1):
            ctx_lines.append(f"[{i}] docID: {doc_id}\n{text}")

        user_content = (
            f"Question: {question}\n\n"
            f"Retrieved passages:\n\n" + "\n\n".join(ctx_lines) + "\n\n"
            "Answer:"
        )

        messages = [
            {"role": "system", "content": PROMPT_RETRIEVE_GENERATE},
            {"role": "user", "content": user_content},
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
            do_sample=False,
            temperature=0.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        gen = self.tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
        gen = gen.replace("\r\n", "\n").strip()
        # collapse excessive blank lines
        while "\n\n\n" in gen:
            gen = gen.replace("\n\n\n", "\n\n")
        return gen
