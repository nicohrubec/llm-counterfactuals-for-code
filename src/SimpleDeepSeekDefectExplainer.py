from typing import List

from helpers import extract_all_code_from_string
from prompt import build_defect_explainer_prompt
from DeepSeekExplainer import DeepSeekExplainer


class SimpleDeepSeekDefectExplainer(DeepSeekExplainer):
    def __init__(self, num_counterfactuals=1):
        super().__init__()
        self.num_counterfactuals = num_counterfactuals

    def explain(self, sample: str, prediction: bool) -> List[str]:
        prompt = build_defect_explainer_prompt(sample, prediction, self.num_counterfactuals)
        response = self.ask_deepseek(prompt)
        explanation = extract_all_code_from_string(response)

        return explanation
