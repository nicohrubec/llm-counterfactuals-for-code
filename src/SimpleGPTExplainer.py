from openai import OpenAI
import json

from helpers import build_explainer_prompt, extract_code_from_string
from Explainer import Explainer


class SimpleGPTExplainer(Explainer):
    def __init__(self, model_str):
        self.model = model_str
        self.client = OpenAI()

    def explain(self, sample: str, prediction: bool) -> str:
        prompt = build_explainer_prompt(sample, prediction)

        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0.4,
            frequency_penalty=1.1,
            top_p=1.0,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        response = json.loads(completion.model_dump_json())['choices'][0]['message']['content']
        explanation = extract_code_from_string(response)

        print(explanation)

        return explanation
