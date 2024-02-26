from typing import List, Tuple


class Explainer:
    system_prompt = "You are an oracle explanation module in a machine learning pipeline."

    top_k = 50
    temperature = 0.4
    top_p = 1.0
    repetition_penalty = 1.1

    def explain(self, sample: str, prediction: bool, previous_solutions: List[Tuple[str, float]] = None) -> List[str]:
        raise NotImplementedError
