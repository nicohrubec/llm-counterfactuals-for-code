from helpers import extract_code_from_string
from src.prompt import build_explainer_prompt
from GPTExplainer import GPTExplainer


class SimpleGPTDefectExplainer(GPTExplainer):
    def explain(self, sample, prediction: bool) -> str:
        prompt = build_explainer_prompt(sample, prediction)
        response = self.ask_gpt(prompt)
        explanation = extract_code_from_string(response)

        print(explanation)

        return explanation
