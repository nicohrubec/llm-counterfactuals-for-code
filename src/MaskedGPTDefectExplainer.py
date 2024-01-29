from helpers import extract_code_from_string
from prompt import build_masked_prompt
from GPTExplainer import GPTExplainer


class MaskedGPTDefectExplainer(GPTExplainer):
    def explain(self, sample, prediction: bool) -> str:
        prompt = build_masked_prompt(sample, str(prediction))
        response = self.ask_gpt(prompt)

        try:
            explanation = extract_code_from_string(response)
        except:
            explanation = ""

        print(explanation)

        return explanation
