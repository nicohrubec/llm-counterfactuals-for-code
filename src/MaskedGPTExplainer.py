from openai import OpenAI
import json

from helpers import extract_code_from_string
from src.prompt import build_masked_prompt
from Explainer import Explainer


class MaskedGPTExplainer(Explainer):
    def __init__(self, model_str):
        self.model = model_str
        self.client = OpenAI()

    def explain(self, sample: str, prediction: bool) -> str:
        prompt = build_masked_prompt(sample, prediction)

        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0.4,
            frequency_penalty=1.1,
            top_p=1.0,
            max_tokens=100,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        response = json.loads(completion.model_dump_json())['choices'][0]['message']['content']

        try:
            explanation = extract_code_from_string(response)
        except:
            explanation = ""

        print(explanation)

        return explanation
