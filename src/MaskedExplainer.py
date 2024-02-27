from typing import List


class MaskedExplainer:
    def explain(self, sample: str, prediction: bool, original_line: str) -> List[str]:
        raise NotImplementedError
