import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from Explainer import Explainer


class DeepSeekExplainer(Explainer):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-7b-instruct-v1.5",
                                                       trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-7b-instruct-v1.5",
                                                          trust_remote_code=True).cuda()

    def ask_deepseek(self, prompt):
        messages = [
            {'role': 'user', 'content': prompt}
        ]

        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")\
            .to(self.model.device)

        outputs = self.model.generate(inputs,
                                      max_new_tokens=3000,
                                      do_sample=False,
                                      top_k=50,
                                      top_p=0.95,
                                      num_return_sequences=1,
                                      eos_token_id=self.tokenizer.eos_token_id)

        return self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)