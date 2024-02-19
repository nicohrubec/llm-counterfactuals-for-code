from helpers import extract_code_from_string
from prompt import build_clone_explainer_prompt
from DeepSeekExplainer import DeepSeekExplainer


class SimpleDeepSeekCloneExplainer(DeepSeekExplainer):
    def explain(self, sample: str, prediction: bool) -> str:
        prompt = build_clone_explainer_prompt(sample, prediction)
        response = self.ask_deepseek(prompt)
        explanation = extract_code_from_string(response)

        return explanation
