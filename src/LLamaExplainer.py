import os
import torch
from transformers import pipeline, AutoTokenizer

from helpers import build_user_prompt, extract_code_from_string
from Explainer import Explainer


class LLamaExplainer(Explainer):
    def __init__(self):
        token = os.environ.get("LLAMA_KEY")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=token)
        self.llm = pipeline(
            "text-generation",
            model="meta-llama/Llama-2-7b-chat-hf",
            torch_dtype=torch.bfloat16,
            device="cuda:0",
            token=token,
            tokenizer=self.tokenizer
        )

    def explain(self, sample: str, prediction: bool) -> str:
        prompt, prompt_len = build_user_prompt(sample, prediction)
        prompt += " <code>"

        sequences = self.llm(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=3000,
        )

        response = sequences[0]['generated_text']
        print(response[prompt_len:])
        explanation = extract_code_from_string(response[prompt_len:])

        return explanation
