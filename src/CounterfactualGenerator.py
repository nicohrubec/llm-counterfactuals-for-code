from typing import Tuple

from BlackBox import BlackBox
from Explainer import Explainer
from SimilarityMetric import SimilarityMetric


class CounterfactualGenerator:
    def __init__(self, explainer: Explainer, blackbox_name: str):
        self.blackbox = BlackBox(blackbox_name)
        self.explainer = explainer
        self.similarity_score = SimilarityMetric()

    def get_counterfactual(self, sample, target) -> Tuple[str, bool, float]:
        raise NotImplementedError
