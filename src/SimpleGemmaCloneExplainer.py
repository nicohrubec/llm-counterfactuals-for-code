from typing import List, Tuple

from helpers import extract_all_code_from_string
from prompt import build_clone_explainer_prompt
from GemmaExplainer import GemmaExplainer


class SimpleGemmaCloneExplainer(GemmaExplainer):
    def __init__(self, num_counterfactuals=1):
        super().__init__()
        self.num_counterfactuals = num_counterfactuals

    def explain(self, sample: str, prediction: bool, previous_solutions: List[Tuple[str, float]] = None) -> List[str]:
        prompt = build_clone_explainer_prompt(sample, prediction, self.num_counterfactuals, previous_solutions)
        response = self.ask_gemma(prompt)
        explanation = extract_all_code_from_string(response)

        return explanation
