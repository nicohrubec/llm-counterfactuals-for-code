from typing import Tuple
from Levenshtein import distance
import heapq

from CounterfactualGenerator import CounterfactualGenerator
from WrongPredictionError import WrongPredictionError


class OneShotCounterfactual(CounterfactualGenerator):
    def get_counterfactual(self, sample, target) -> Tuple[str, bool, float, int]:
        original_label, original_score = self.blackbox(sample)

        if original_label != target:
            raise WrongPredictionError

        candidate_counterfactuals = self.explainer.explain(sample, original_label)
        counterfactual_scores = []
        heapq.heapify(counterfactual_scores)
        counterfactual_found = False

        for candidate_counterfactual in candidate_counterfactuals:
            counterfactual_label, counterfactual_score = self.blackbox(candidate_counterfactual)
            similarity_score = float(self.similarity_score(sample, candidate_counterfactual)[0][0])
            token_distance = distance(sample, candidate_counterfactual)

            if counterfactual_label != original_label:
                counterfactual_found = True

            heapq.heappush(counterfactual_scores, (token_distance, counterfactual_label, counterfactual_score,
                                                   candidate_counterfactual, similarity_score))

        counterfactual_label = original_label
        if counterfactual_found:  # find counterfactual with minimal token distance to original sample
            while counterfactual_label == original_label:
                token_distance, counterfactual_label, counterfactual_score, candidate_counterfactual, \
                    similarity_score = heapq.heappop(counterfactual_scores)
        else:  # no counterfactual found, return minimal token distance sample
            token_distance, counterfactual_label, counterfactual_score, candidate_counterfactual, similarity_score, = \
                heapq.heappop(counterfactual_scores)

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
