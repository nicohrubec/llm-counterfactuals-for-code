from typing import List, Tuple

clone_task = "Code Defect Detection on the Devign dataset"
defect_task = "Code Clone Detection on the Big Clone Bench dataset"
clone_label_explanation = "The label 'True' means that the functions are clones, so you need to change the semantics in such a way that the functions are not clones anymore." \
                          "The label 'False' means that the functions are not clones, so you need to change the semantics of one of the functions such that it is a clone of the other."
defect_label_explanation = "The label 'True' means that one or multiple Defects were detected, so you need to remove all Defects to flip the label to 'False'." \
                           "The label 'False' means that no Defect was detected, so you need to introduce a Defect to flip the label to 'False'."
clone_definition = """Use the following definition of 'Code Clone Detection':
"The objective is to identify whether two functions are semantically the same. If two functions are semantically equal, they are called Code Clones."
"""
defect_definition = """Use the following definition of 'Code Defect Detection':
"The objective is to identify whether a body of source code contains defects that may be used to attack software systems, such as resource leaks, use-after-free vulnerabilities, and DoS attack."
"""
counterfactual_definition = """Use the following definition of 'counterfactual explanation':
A counterfactual explanation reveals what should have been different in an instance to observe a diverse outcome."""
detailed_instructions_single_shot = """
Enclose the code with the counterfactual in <code> tags.
For each counterfactual you propose, return the full original code with your proposed alteration embedded in it.
As mentioned the change should be minimal and therefore not affect more than 5 lines of the original code.
For your proposed counterfactuals, favor removing or modifying existing lines over inserting new code into the program.
Do not insert code comments.
"""
detailed_instructions_multi_shot = """
Enclose the suggested code line with the counterfactual in <code> tags. 
Do not explain your reasoning. Only return the code.
Do not insert code comments.
"""


def add_previous_solutions(previous_solutions: List[Tuple[str, float]]):
    previous_solutions.sort(key=lambda x: x[1])
    previous_solutions_prompt = """
    Previously you proposed the following list of altered programs when asked to generate a counterfactual explanation.
    The prediction change describes by how much the confidence of the black-box classifier shifted as a result of the counterfactual, so a higher prediction change means we are closer to flipping the label.
    You can choose to build on top of these solutions by adding additional changes or create a new one.
    """

    for prediction_change, previous_solution in previous_solutions:
        previous_solutions_prompt += f"""
        The following counterfactual resulted in a prediction change of {prediction_change}:\n
        {previous_solution}\n\n
        """

    return previous_solutions_prompt


def build_defect_explainer_prompt(sample, prediction: bool, n: int, previous_solutions: List[Tuple[str, float]]) -> str:
    prompt = f"""
    In the task of {defect_task}, a trained black-box classifier predicted the label {prediction} for the following code.
    Generate {n} counterfactual explanations by making minimal changes to the code, so that the label changes from {prediction} to {not prediction}.
    {defect_label_explanation}
    You should suggest exactly {n} counterfactual explanations.

    {defect_definition}\n\n{counterfactual_definition}\n\n{detailed_instructions_single_shot}

    \n—\nCode:\n{sample}\n\n
    """

    if previous_solutions:
        prompt += add_previous_solutions(previous_solutions)

    return prompt


def build_clone_explainer_prompt(sample, prediction: bool, n: int, previous_solutions: List[Tuple[str, float]]) -> str:
    prompt = f"""
    In the task of {clone_task}, a trained black-box classifier predicted the label {prediction} for the following code containing two functions.
    Generate {n} counterfactual explanations by making minimal changes to the code, so that the label changes from {prediction} to {not prediction}.
    {clone_label_explanation}
    You should suggest exactly {n} counterfactual explanations.
    
    {clone_definition}\n\n{counterfactual_definition}\n\n{detailed_instructions_single_shot}

    \n—\nCode:\n{sample}\n\n
    """

    if previous_solutions:
        prompt += add_previous_solutions(previous_solutions)

    return prompt


def build_defect_masked_prompt(sample, prediction: bool, original_line: str) -> str:
    prompt = f"""
    In the task of {defect_task}, a trained black-box classifier predicted the label {prediction} for the following code.
    One line of the original program was masked using the <MASK> token. 
    Generate a single line counterfactual explanation by suggesting a replacement for the masked line that is syntactically and semantically coherent with the rest of the program, such that the label changes from {prediction} to {not prediction}.
    {defect_label_explanation}

    {defect_definition}\n\n{counterfactual_definition}\n\n{detailed_instructions_multi_shot}
    
    \n—\nCode:\n{sample}\n\nOriginal line: {original_line}\n
    """

    return prompt


def build_clone_masked_prompt(sample, prediction: bool, original_line: str) -> str:
    prompt = f"""
    In the task of {clone_task}, a trained black-box classifier predicted the label {prediction} for the following code containing two functions.
    One line of the original program was masked using the <MASK> token. 
    Generate a single line counterfactual explanation by suggesting a replacement for the masked line that is syntactically and semantically coherent with the rest of the program, such that the label changes from {prediction} to {not prediction}.
    {clone_label_explanation}

    {clone_definition}\n\n{counterfactual_definition}\n\n{detailed_instructions_multi_shot}
    
    \n—\nCode:\n{sample}\nOriginal line: {original_line}\n
    """

    return prompt


def build_defect_explainer_with_identified_words_prompt(prediction: bool) -> str:
    prompt = f"""
    Generate a counterfactual explanation for the original text by ONLY changing a minimal set of the lines you identified,
    so that the label changes from {prediction} to {not prediction}.
    {defect_label_explanation}
    
    {defect_definition}\n\n{counterfactual_definition}\n\n{detailed_instructions_single_shot}
    
    """

    return prompt


def build_clone_explainer_with_identified_words_prompt(prediction: bool) -> str:
    prompt = f"""
    Generate a counterfactual explanation for the original text by ONLY changing a minimal set of the lines you identified,
    so that the label changes from {prediction} to {not prediction}.
    {clone_label_explanation}

    {clone_definition}\n\n{counterfactual_definition}\n\n{detailed_instructions_single_shot}

    """

    return prompt


def build_defect_identify_words_prompt(sample, prediction: bool) -> str:
    prompt = f"""
    In the task of {defect_task}, a trained black-box classifier predicted the label {prediction} for the following code. 
    Explain why the model predicted the label {prediction} by identifying lines in the input that caused the label.
    List ONLY the lines as a comma separated list.\n—\nCode:\n{sample}\n"""

    return prompt


def build_clone_identify_words_prompt(sample, prediction: bool) -> str:
    prompt = f"""
    In the task of {clone_task}, a trained black-box classifier predicted the label {prediction} for the following code. 
    Explain why the model predicted the label {prediction} by identifying lines in the input that caused the label.
    List ONLY the lines as a comma separated list.\n—\nCode:\n{sample}\n"""

    return prompt
