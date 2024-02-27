import re
from typing import List
import string

from datasets import load_dataset


def count_lines(text):
    return text.count('\n') + 1


def get_dataset(dataset_name, n_samples=100, max_num_lines=25, filter_col='func'):
    dataset = load_dataset(dataset_name, split="train").to_pandas()
    dataset['num_lines'] = dataset[filter_col].apply(count_lines)
    dataset = dataset[dataset.num_lines <= max_num_lines]

    return dataset[:n_samples]


def extract_code_from_string(output: str) -> str:
    pattern = re.compile(r'<code>(.*?)<\/?code>|\`\`\`(?:c|java)(.*?)\`\`\`', re.DOTALL)
    matched = pattern.search(output)

    return matched.group(1) if matched.group(1) else matched.group(2) if matched.group(2) else None


def extract_all_code_from_string(output: str) -> List[str]:
    pattern = re.compile(r'<code>(.*?)<\/?code>|\`\`\`<code>(.*?)\`\`\`|\`\`\`(?:c|java)(.*?)\`\`\`', re.DOTALL)
    matches = pattern.findall(output)
    snippets = []
    for match in matches:
        if match[0]:
            snippet = match[0]
        elif match[1]:
            snippet = match[1]
        else:
            snippet = match[2]
        if snippet:
            snippet = "\n".join([line for line in snippet.split("\n") if "<code>" not in line and "</code>" not in line])
            snippets.append(snippet)

    return snippets


def remove_comments(program: str) -> str:
    lines = []
    for line in program.split('\n'):
        # Split the line on the first occurrence of "//"
        line_without_comment, _, _ = line.partition('//')
        lines.append(line_without_comment)

    lines = '\n'.join(lines)
    return remove_empty_lines(lines)


# Compares if two programs are the same while ignoring whitespace
def is_equal_for_programs(p1: str, p2: str) -> bool:
    remove = string.punctuation + string.whitespace
    mapping = {ord(c): None for c in remove}
    return p1.translate(mapping) == p2.translate(mapping)


def remove_empty_lines(input_string):
    lines = input_string.split('\n')
    non_empty_lines = filter(lambda x: x.strip(), lines)
    result_string = '\n'.join(non_empty_lines)
    return result_string
