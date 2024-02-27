from openai import OpenAI
import json

from params import ModelParams as mp


class GPTExplainer:
    def __init__(self, model_str):
        self.model = model_str
        self.client = OpenAI()

    def ask_gpt(self, prompt):
        messages = [
            {"role": "system", "content": mp.system_prompt}
        ]

        if isinstance(prompt, str):  # simple input prompt string
            messages += [
                {"role": "user", "content": prompt}
            ]
        else:
            messages += prompt

        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=mp.temperature,
            frequency_penalty=mp.repetition_penalty,
            top_p=mp.top_p,
            messages=messages
        )

        response = json.loads(completion.model_dump_json())['choices'][0]['message']['content']

        return response
