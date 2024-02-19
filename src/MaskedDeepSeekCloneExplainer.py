from helpers import extract_code_from_string
from prompt import build_clone_masked_prompt
from DeepSeekExplainer import DeepSeekExplainer


class MaskedDeepSeekCloneExplainer(DeepSeekExplainer):
    def explain(self, sample: str, prediction: bool) -> str:
        prompt = build_clone_masked_prompt(sample, prediction)
        response = self.ask_deepseek(prompt)

        try:
            explanation = extract_code_from_string(response)
        except:
            explanation = ""

        return explanation
