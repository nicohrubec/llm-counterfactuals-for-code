import os
import torch
from transformers import pipeline, AutoTokenizer

from Explainer import Explainer


class LLamaExplainer(Explainer):
    def __init__(self, model_str):
        token = os.environ.get("LLAMA_KEY")
        self.tokenizer = AutoTokenizer.from_pretrained(model_str, token=token)
        self.llm = self.get_llama(model_str, token)

    def get_llama(self, model_str, token):
        return pipeline(
            "text-generation",
            model=model_str,
            torch_dtype=torch.bfloat16,
            device="cuda:0",
            token=token,
            tokenizer=self.tokenizer
        )

    def ask_llama(self, prompt):
        sequences = self.llm(
            prompt,
            do_sample=True,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            temperature=self.temperature,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=3000,
        )
        response = sequences[0]['generated_text']
        return response
