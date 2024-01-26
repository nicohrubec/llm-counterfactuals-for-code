from typing import Tuple

from Explainer import Explainer
from CounterfactualGenerator import CounterfactualGenerator

label2target = {'LABEL_0': False, 'LABEL_1': True}


class OneShotCounterfactual(CounterfactualGenerator):
    def __init__(self, explainer: Explainer):
        super().__init__(explainer)

    def get_counterfactual(self, sample, target) -> Tuple[str, bool, float]:
        original_label, original_score = self.blackbox(sample)
        candidate_counterfactual = self.explainer.explain(sample, label2target[original_label])
        counterfactual_label, counterfactual_score = self.blackbox(candidate_counterfactual)
        similarity_score = float(self.similarity_score(sample, candidate_counterfactual)[0][0])

        print(f"The correct label is: {target}")
        print(f"Originally the model predicted {label2target[original_label]} with a confidence of {original_score}.")
        print(
            f"After applying the counterfactual the model predicted {label2target[counterfactual_label]} with a confidence of {counterfactual_score}.")
        print(f"Similarity score: {similarity_score:.{4}f}")
        print()

        return candidate_counterfactual, original_label != counterfactual_label, similarity_score
