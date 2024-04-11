import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from params import ModelParams as mp


class DeepSeekExplainer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct",
                                                       trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct",
                                                          trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

    def ask_deepseek(self, prompt):
        messages = [
            {'role': 'user', 'content': prompt}
        ]

        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")\
            .to(self.model.device)

        outputs = self.model.generate(inputs,
                                      max_new_tokens=3000,
                                      do_sample=True,
                                      top_k=mp.top_k,
                                      temperature=mp.temperature,
                                      top_p=mp.top_p,
                                      repetition_penalty=mp.repetition_penalty,
                                      num_return_sequences=1,
                                      eos_token_id=self.tokenizer.eos_token_id,
                                      pad_token_id=self.tokenizer.pad_token_id)

        return self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
