from helpers import extract_code_from_string
from prompt import build_defect_masked_prompt
from GemmaExplainer import GemmaExplainer


class MaskedGemmaDefectExplainer(GemmaExplainer):
    def explain(self, sample: str, prediction: bool) -> str:
        prompt = build_defect_masked_prompt(sample, prediction)
        response = self.ask_gemma(prompt)

        try:
            explanation = extract_code_from_string(response)
        except:
            explanation = ""

        return explanation
