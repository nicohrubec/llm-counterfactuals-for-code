from openai import OpenAI
import json

from helpers import build_identify_words_prompt, build_explainer_with_identified_words_prompt, extract_code_from_string
from Explainer import Explainer


class GPTCoTExplainer(Explainer):
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
            temperature=0.4,
            frequency_penalty=1.1,
            top_p=1.0,
            messages=messages
        )

        response = json.loads(completion.model_dump_json())['choices'][0]['message']['content']

        return response

    def explain(self, sample: str, prediction: bool) -> str:
        identify_words_prompt = build_identify_words_prompt(sample, prediction)

        messages = [
            {"role": "user", "content": identify_words_prompt}
        ]

        identified_words_response = self.ask_gpt(messages)
        prompt = build_explainer_with_identified_words_prompt(prediction)

        messages += [
            {"role": "assistant", "content": identified_words_response},
            {"role": "user", "content": prompt}
        ]

        answer = self.ask_gpt(messages)
        explanation = extract_code_from_string(answer)

        print(explanation)

        return explanation
