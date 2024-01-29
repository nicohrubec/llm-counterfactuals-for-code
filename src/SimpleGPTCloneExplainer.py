from helpers import extract_code_from_string
from prompt import build_clone_explainer_prompt
from GPTExplainer import GPTExplainer


class SimpleGPTCloneExplainer(GPTExplainer):
    def explain(self, sample: str, prediction: bool) -> str:
        prompt = build_clone_explainer_prompt(sample, prediction)
        response = self.ask_gpt(prompt)
        explanation = extract_code_from_string(response)

        print(explanation)

        return explanation
