from typing import List, Tuple

from helpers import extract_code_from_string
from prompt import build_defect_masked_prompt
from DeepSeekExplainer import DeepSeekExplainer


class MaskedDeepSeekDefectExplainer(DeepSeekExplainer):
    def explain(self, sample: str, prediction: bool, previous_solutions: List[Tuple[str, int]] = None) -> List[str]:
        prompt = build_defect_masked_prompt(sample, prediction)
        response = self.ask_deepseek(prompt)

        try:
            explanation = extract_code_from_string(response)
        except:
            explanation = ""

        return [explanation]
