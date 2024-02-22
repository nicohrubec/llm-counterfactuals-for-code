import torch
from transformers import pipeline, AutoTokenizer

from Explainer import Explainer


class GemmaExplainer(Explainer):
    def __init__(self):
        model_str = "google/gemma-7b-it"
        self.tokenizer = AutoTokenizer.from_pretrained(model_str)
        self.llm = self.get_gemma(model_str)

    def get_gemma(self, model_str):
        return pipeline(
            "text-generation",
            model=model_str,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda"
        )

    def ask_gemma(self, prompt):
        messages = [
            {"role": "user", "content": prompt},
        ]

        prompt = self.llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipeline(
            prompt,
            max_new_tokens=3000,
            add_special_tokens=True,
            do_sample=True,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty
        )

        return outputs[0]["generated_text"][len(prompt):]
