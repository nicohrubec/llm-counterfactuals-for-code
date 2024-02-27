from typing import List, Tuple


class SimpleExplainer:
    def explain(self, sample: str, prediction: bool, previous_solutions: List[Tuple[str, float]] = None) -> List[str]:
        raise NotImplementedError
