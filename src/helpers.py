import re
from typing import List

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
    pattern = re.compile(r'<code>(.*?)<\/?code>|\`\`\`(?:c|java)(.*?)\`\`\`', re.DOTALL)
    matches = pattern.findall(output)
    snippets = []
    for match in matches:
        snippet = match[0] if match[0] else match[1]
        if snippet:
            snippets.append(snippet)
    return snippets
