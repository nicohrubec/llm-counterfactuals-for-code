from typing import List

from helpers import extract_code_from_string, remove_comments
from prompt import build_clone_masked_line_prompt
from DeepSeekExplainer import DeepSeekExplainer
from MaskedExplainer import MaskedExplainer


class MaskedDeepSeekCloneExplainer(MaskedExplainer, DeepSeekExplainer):
    def explain(self, sample: str, prediction: bool, original_line: str) -> List[str]:
        prompt = build_clone_masked_line_prompt(sample, prediction, original_line)
        response = self.ask_deepseek(prompt)

        try:
            explanation = extract_code_from_string(response)
            explanation = remove_comments(explanation)
        except:
            explanation = ""

        return [explanation]
