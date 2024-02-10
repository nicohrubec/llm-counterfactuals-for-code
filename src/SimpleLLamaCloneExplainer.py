from helpers import extract_code_from_string
from prompt import build_clone_explainer_prompt, build_llama_prompt
from LLamaExplainer import LLamaExplainer


class SimpleLLamaCloneExplainer(LLamaExplainer):
    def explain(self, sample: str, prediction: bool) -> str:
        prompt = build_clone_explainer_prompt(sample, prediction)
        prompt, prompt_len = build_llama_prompt(self.system_prompt, prompt)
        response = self.ask_llama(prompt)
        explanation = extract_code_from_string(response[prompt_len:])

        print(explanation)

        return explanation