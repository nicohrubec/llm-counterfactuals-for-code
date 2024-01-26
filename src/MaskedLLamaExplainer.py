import os
import torch
from transformers import pipeline, AutoTokenizer

from helpers import extract_code_from_string
from src.prompt import build_masked_prompt, build_llama_prompt
from Explainer import Explainer


class MaskedLLamaExplainer(Explainer):
    def __init__(self, model_str):
        token = os.environ.get("LLAMA_KEY")
        self.tokenizer = AutoTokenizer.from_pretrained(model_str, token=token)
        self.llm = pipeline(
            "text-generation",
            model=model_str,
            torch_dtype=torch.bfloat16,
            device="cuda:0",
            token=token,
            tokenizer=self.tokenizer
        )

    def explain(self, sample: str, prediction: bool) -> str:
        prompt = build_masked_prompt(sample, prediction)
        prompt, prompt_len = build_llama_prompt(self.system_prompt, prompt)

        sequences = self.llm(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=3000,
        )

        response = sequences[0]['generated_text']
        explanation = extract_code_from_string(response[prompt_len:])

        print(explanation)

        return explanation