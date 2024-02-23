from typing import List

from helpers import extract_code_from_string
from prompt import build_defect_explainer_prompt
from DeepSeekExplainer import DeepSeekExplainer


class SimpleDeepSeekDefectExplainer(DeepSeekExplainer):
    def explain(self, sample: str, prediction: bool) -> List[str]:
        prompt = build_defect_explainer_prompt(sample, prediction)
        response = self.ask_deepseek(prompt)
        explanation = extract_code_from_string(response)

        return [explanation]
