import re
from typing import Tuple

from datasets import load_dataset


def get_dataset(n_samples=1000):
    return load_dataset("code_x_glue_cc_defect_detection", split="train").to_pandas()[:n_samples]

def extract_code_from_string(output: str) -> str:
    pattern = re.compile(r'<code>(.*?)<\/code>|\`\`\`c(.*?)\`\`\`', re.DOTALL)
    matched = pattern.search(output)

    return matched.group(1) if matched.group(1) else matched.group(2) if matched.group(2) else None



def build_user_prompt(sample, target: bool) -> Tuple[str, int]:
    prompt = f"""
    In the task of Code Defect Detection on the Devign dataset, a trained black-box classifier predicted the label {target} for the following code.
    Generate a counterfactual explanation by making minimal changes to the code, so that the label changes from {target} to {not target}.
    Use the following definition of 'counterfactual explanation':
    “A counterfactual explanation reveals what should have been different in an instance to observe a diverse outcome."
    In your answer create an altered version of the following code with a proposed counterfactual.
    Output the altered code in the following format:
    <code> altered_code <code>
    
    In particular do not only return your proposed change, but rather the full original code with your suggestion already embedded.
    Definitely do not abbreviate any code by using comments. Always return the full code.
    Explain your reasoning.
    Follow my instructions as precisely as possible.\n—\nCode:\n{sample}\n
    """
    prompt_len = len(prompt)

    return prompt, prompt_len


# TODO: does it make sense to generate counterfactuals for samples that the model mispredicted in the first place?
