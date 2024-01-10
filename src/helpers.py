import re

from datasets import load_dataset


def count_lines(text):
    return text.count('\n') + 1


def get_dataset(n_samples=100, max_num_lines=50):
    dataset = load_dataset("code_x_glue_cc_defect_detection", split="train").to_pandas()
    dataset['num_lines'] = dataset['func'].apply(count_lines)
    dataset = dataset[dataset.num_lines <= max_num_lines]

    return dataset[:n_samples]


def extract_questions(questions_answer):
    pattern = re.compile(r'Q\d+: (.+?)(?:\n|$)')
    matches = pattern.findall(questions_answer)

    return matches


def extract_code_from_string(output: str) -> str:
    pattern = re.compile(r'<code>(.*?)<\/?code>|\`\`\`c(.*?)\`\`\`', re.DOTALL)
    matched = pattern.search(output)

    return matched.group(1) if matched.group(1) else matched.group(2) if matched.group(2) else None


def build_user_prompt(sample, prediction: bool) -> str:
    prompt = f"""
    In the task of Code Defect Detection on the Devign dataset, a trained black-box classifier predicted the label {prediction} for the following code.
    Generate a counterfactual explanation by making minimal changes to the code, so that the label changes from {prediction} to {not prediction}, 
    where 'True' means that a Defect was detected, while 'False' means no Defect was found in the code.
    
    Use the following definition of 'Code Defect Detection':
    "The objective is to identify whether a body of source code contains defects that may be used to attack software systems, such as resource leaks, use-after-free vulnerabilities, and DoS attack."
    
    Use the following definition of 'counterfactual explanation':
    “A counterfactual explanation reveals what should have been different in an instance to observe a diverse outcome."
    
    In your answer update the provided code to generate a counterfactual explanation.
    Always return the full original code with your changes embedded. 
    Never abbreviate using comments, for example by using:
    /* rest of the code remains unchanged */
    
    Enclose the code with the counterfactual in <code> tags. 
    \n—\nCode:\n{sample}\n
    """

    return prompt


# TODO: does it make sense to generate counterfactuals for samples that the model mispredicted in the first place?
