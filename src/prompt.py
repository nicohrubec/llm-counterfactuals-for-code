def build_defect_explainer_prompt(sample, prediction: bool) -> str:
    prompt = f"""
    In the task of Code Defect Detection on the Devign dataset, a trained black-box classifier predicted the label {prediction} for the following code.
    Generate a counterfactual explanation by making minimal changes to the code, so that the label changes from {prediction} to {not prediction}, 
    where 'True' means that a Defect was detected, while 'False' means no Defect was found in the code.
    
    Use the following definition of 'Code Defect Detection':
    "The objective is to identify whether a body of source code contains defects that may be used to attack software systems, such as resource leaks, use-after-free vulnerabilities, and DoS attack."
    
    Use the following definition of 'counterfactual explanation':
    “A counterfactual explanation reveals what should have been different in an instance to observe a diverse outcome."
    
    Enclose the code with the counterfactual in <code> tags. 
    
    In your answer update the provided code to generate a counterfactual explanation.
    Always return the full original code with your changes embedded. 
    Never abbreviate using comments, for example by using:
    /* rest of the code remains unchanged */
    
    \n—\nCode:\n{sample}\n
    """

    return prompt


def build_clone_explainer_prompt(sample, prediction: bool) -> str:
    prompt = f"""
    In the task of Code Clone Detection on the Big Clone Bench dataset, a trained black-box classifier predicted the label {prediction} for the following code containing two functions.
    Generate a counterfactual explanation by making minimal changes to the code, so that the label changes from {prediction} to {not prediction}, 
    where 'True' means that the functions are clones, while 'False' means the functions are not clones.

    Use the following definition of 'Code Clone Detection':
    "The objective is to identify whether two functions are semantically the same. If two functions are semantically equal, they are called Code Clones."

    Use the following definition of 'counterfactual explanation':
    “A counterfactual explanation reveals what should have been different in an instance to observe a diverse outcome."

    Enclose the code with the counterfactual in <code> tags. 

    In your answer update the provided code to generate a counterfactual explanation.
    Always return the full original code with your changes embedded. 
    Never abbreviate using comments, for example by using:
    /* rest of the code remains unchanged */

    \n—\nCode:\n{sample}\n
    """

    return prompt


def build_defect_masked_prompt(sample, prediction: str) -> str:
    prompt = f"""
    In the task of Code Defect Detection on the Devign dataset, a trained black-box classifier predicted the label {prediction} for the following code.
    One line of the original program was masked using the <MASK> token. 
    Generate a single line counterfactual explanation by suggesting a replacement for the masked line that makes sense based on the rest of the code, such that the label changes from {prediction} to {not prediction},
    where 'True' means that a Defect was detected, while 'False' means no Defect was found in the code.

    Use the following definition of 'Code Defect Detection':
    "The objective is to identify whether a body of source code contains defects that may be used to attack software systems, such as resource leaks, use-after-free vulnerabilities, and DoS attack."

    Use the following definition of 'counterfactual explanation':
    “A counterfactual explanation reveals what should have been different in an instance to observe a diverse outcome."

    Enclose the suggested code line with the counterfactual in <code> tags. Do not explainy your reasoning. Only return the code.

    \n—\nCode:\n{sample}\n
    """

    return prompt


def build_defect_explainer_with_identified_words_prompt(prediction: bool) -> str:
    prompt = f"""
    Generate a counterfactual explanation for the original text by ONLY changing a minimal set of the lines you identified,
    so that the label changes from {prediction} to {not prediction}, 
    where 'True' means that a Defect was detected, while 'False' means no Defect was found in the code.
    
    Use the following definition of 'Code Defect Detection':
    "The objective is to identify whether a body of source code contains defects that may be used to attack software systems, such as resource leaks, use-after-free vulnerabilities, and DoS attack."

    Use the following definition of 'counterfactual explanation':
    “A counterfactual explanation reveals what should have been different in an instance to observe a diverse outcome."
    
    Enclose the code with the counterfactual in <code> tags. 
    
    In your answer update the provided code to generate a counterfactual explanation.
    Always return the full original code with your changes embedded. 
    Never abbreviate using comments, for example by using:
    /* rest of the code remains unchanged */
    
    """

    return prompt


def build_defect_identify_words_prompt(sample, prediction: bool) -> str:
    prompt = f"""
    In the task of Code Defect Detection on the Devign dataset, a trained black-box classifier predicted the label {prediction} for the following code. 
    Explain why the model predicted the label {prediction} by identifying lines in the input that caused the label.
    List ONLY the lines as a comma separated list.\n—\nCode:\n{sample}\n"""

    return prompt


def build_llama_prompt(system_prompt, prompt):
    prompt = f"""
<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{prompt} [/INST] 
"""
    prompt_len = len(prompt)
    prompt += " <code>"

    return prompt, prompt_len
