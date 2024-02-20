from typing import Tuple
from Levenshtein import distance

from CounterfactualGenerator import CounterfactualGenerator
from WrongPredictionError import WrongPredictionError


class OneShotCounterfactual(CounterfactualGenerator):
    def get_counterfactual(self, sample, target) -> Tuple[str, bool, float, int]:
        original_label, original_score = self.blackbox(sample)

        if original_label != target:
            raise WrongPredictionError

        candidate_counterfactual = self.explainer.explain(sample, original_label)
        counterfactual_label, counterfactual_score = self.blackbox(candidate_counterfactual)
        similarity_score = float(self.similarity_score(sample, candidate_counterfactual)[0][0])
        token_distance = distance(sample, candidate_counterfactual)

        print(f"The correct label is: {target}")
        print(f"Originally the model predicted {original_label} with a confidence of {original_score}.")
        print(
            f"After applying the counterfactual the model predicted {counterfactual_label} with a confidence of "
            f"{counterfactual_score}.")
        print(f"Similarity score: {similarity_score:.{4}f}")
        print(f"Token distance: {token_distance}")
        print(f"Original sample:\n{sample}\n\nProposed counterfactual:\n{candidate_counterfactual}")
        print()

        return candidate_counterfactual, original_label != counterfactual_label, similarity_score, token_distance
