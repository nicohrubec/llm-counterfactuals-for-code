from typing import List, Tuple

from helpers import extract_code_from_string, remove_comments
from prompt import build_clone_masked_prompt
from DeepSeekExplainer import DeepSeekExplainer


class MaskedDeepSeekCloneExplainer(DeepSeekExplainer):
    def explain(self, sample: str, prediction: bool, previous_solutions: List[Tuple[str, int]] = None) -> List[str]:
        prompt = build_clone_masked_prompt(sample, prediction)
        response = self.ask_deepseek(prompt)

        try:
            explanation = extract_code_from_string(response)
            explanation = remove_comments(explanation)
        except:
            explanation = ""

        return [explanation]
