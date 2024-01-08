from openai import OpenAI
import json

from src.helpers import build_user_prompt, extract_code_from_string
from src.explainers.Explainer import Explainer


class GPTExplainer(Explainer):
    def __init__(self, model_str):
        self.model = model_str
        self.client = OpenAI()

    def explain(self, sample: str, target: bool) -> str:
        prompt, prompt_len = build_user_prompt(sample, target)

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        response = json.loads(completion.model_dump_json())['choices'][0]['message']['content']
        explanation = extract_code_from_string(response)

        return explanation
