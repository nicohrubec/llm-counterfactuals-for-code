from typing import Tuple, Union

from BlackBox import BlackBox
from SimpleExplainer import SimpleExplainer
from MaskedExplainer import MaskedExplainer
from SimilarityMetric import SimilarityMetric
from helpers import format_code


class CounterfactualGenerator:
    def __init__(self, explainer: Union[SimpleExplainer, MaskedExplainer], blackbox_name: str):
        self.blackbox = BlackBox(blackbox_name)
        self.explainer = explainer
        self.similarity_score = SimilarityMetric()

    def get_counterfactual(self, sample, target) -> Tuple[str, bool, float, int]:
        raise NotImplementedError

    def print_results(self, candidate_counterfactual, counterfactual_label, counterfactual_score, original_label,
                      original_score, sample, similarity_score, target, token_distance):
        print(f"The correct label is: {target}")
        print(f"Originally the model predicted {original_label} with a confidence of {original_score}.")
        print(
            f"After applying the counterfactual the model predicted {counterfactual_label} with a confidence of "
            f"{counterfactual_score}.")
        print(f"Similarity score: {similarity_score:.{4}f}")
        print(f"Token distance: {token_distance}")
        print(f"Original sample:\n{sample}\n\nProposed counterfactual:\n{format_code(candidate_counterfactual)}")
        print()
