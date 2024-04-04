from typing import Tuple
from Levenshtein import distance

from CounterfactualGenerator import CounterfactualGenerator
from WrongPredictionError import WrongPredictionError


class BaselineCounterfactual(CounterfactualGenerator):
    def get_counterfactual(self, sample, target) -> Tuple[str, bool, float, int]:
        original_label, original_score = self.blackbox(sample)

        if original_label != target:
            raise WrongPredictionError

        counterfactual = self.explainer.explain(sample, original_label)

        if counterfactual:
            counterfactual = counterfactual[0]
            similarity_score = float(self.similarity_score(sample, counterfactual)[0][0])
            token_distance = distance(sample, counterfactual)
            counterfactual_label, counterfactual_score = self.blackbox(counterfactual)

            self.print_results(counterfactual, counterfactual_label, counterfactual_score, original_label,
                               original_score, sample, similarity_score, target, token_distance)

            return counterfactual, True, similarity_score, token_distance

        print(f"Original sample:\n{sample}\n")
        print("No counterfactual was found!")
        return "", False, 0.0, 0
