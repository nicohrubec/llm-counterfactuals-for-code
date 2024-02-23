from typing import List

from helpers import extract_code_from_string
from prompt import build_clone_masked_prompt
from GPTExplainer import GPTExplainer


class MaskedGPTCloneExplainer(GPTExplainer):
    def explain(self, sample: str, prediction: bool) -> List[str]:
        prompt = build_clone_masked_prompt(sample, prediction)
        response = self.ask_gpt(prompt)

        try:
            explanation = extract_code_from_string(response)
        except:
            explanation = ""

        return [explanation]
