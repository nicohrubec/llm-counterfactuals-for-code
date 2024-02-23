from typing import List

from helpers import extract_code_from_string
from prompt import build_clone_masked_prompt
from GemmaExplainer import GemmaExplainer


class MaskedGemmaCloneExplainer(GemmaExplainer):
    def explain(self, sample: str, prediction: bool) -> List[str]:
        prompt = build_clone_masked_prompt(sample, prediction)
        response = self.ask_gemma(prompt)

        try:
            explanation = extract_code_from_string(response)
        except:
            explanation = ""

        return [explanation]
