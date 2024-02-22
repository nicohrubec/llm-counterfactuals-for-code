from openai import OpenAI
import json

from Explainer import Explainer


class GPTExplainer(Explainer):
    def __init__(self, model_str):
        self.model = model_str
        self.client = OpenAI()

    def ask_gpt(self, prompt):
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        if isinstance(prompt, str):  # simple input prompt string
            messages += [
                {"role": "user", "content": prompt}
            ]
        else:
            messages += prompt

        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            frequency_penalty=self.repetition_penalty,
            top_p=self.top_p,
            messages=messages
        )

        response = json.loads(completion.model_dump_json())['choices'][0]['message']['content']

        return response
