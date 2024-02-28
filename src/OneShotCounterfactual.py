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

        # evaluate perturbations
        for candidate_counterfactual in candidate_counterfactuals:
            counterfactual_label, counterfactual_score = self.blackbox(candidate_counterfactual)
            similarity_score = float(self.similarity_score(sample, candidate_counterfactual)[0][0])
            token_distance = distance(sample, candidate_counterfactual)

            if counterfactual_label != original_label:
                counterfactual_found = True

            heapq.heappush(counterfactual_scores, (token_distance, counterfactual_label, counterfactual_score,
                                                   candidate_counterfactual, similarity_score))

        counterfactual_label = original_label
        token_distance = 0
        similarity_score = 1.0
        candidate_counterfactual = sample
        counterfactual_score = original_score

        if candidate_counterfactuals:
            if counterfactual_found:  # find counterfactual with minimal token distance to original sample
                while counterfactual_label == original_label:
                    token_distance, counterfactual_label, counterfactual_score, candidate_counterfactual, \
                        similarity_score = heapq.heappop(counterfactual_scores)
            else:  # no counterfactual found, return minimal token distance sample that is not the original sample
                while candidate_counterfactuals and token_distance == 0:
                    token_distance, counterfactual_label, counterfactual_score, candidate_counterfactual, \
                        similarity_score, = heapq.heappop(counterfactual_scores)

            self.print_results(candidate_counterfactual, counterfactual_label, counterfactual_score, original_label,
                               original_score, sample, similarity_score, target, token_distance)
        else:  # use original sample as default
            self.print_results(sample, original_label, original_score, original_label, original_score, sample,
                               similarity_score, target, token_distance)

        return candidate_counterfactual, original_label != counterfactual_label, similarity_score, token_distance
