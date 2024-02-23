from typing import List

from helpers import extract_code_from_string
from prompt import build_defect_explainer_with_identified_words_prompt, build_defect_identify_words_prompt
from GPTExplainer import GPTExplainer


class CoTGPTDefectExplainer(GPTExplainer):
    def explain(self, sample: str, prediction: bool) -> List[str]:
        identify_words_prompt = build_defect_identify_words_prompt(sample, prediction)

        messages = [
            {"role": "user", "content": identify_words_prompt}
        ]

        identified_words_response = self.ask_gpt(messages)
        prompt = build_defect_explainer_with_identified_words_prompt(prediction)

        messages += [
            {"role": "assistant", "content": identified_words_response},
            {"role": "user", "content": prompt}
        ]

        answer = self.ask_gpt(messages)
        explanation = extract_code_from_string(answer)

        return [explanation]
