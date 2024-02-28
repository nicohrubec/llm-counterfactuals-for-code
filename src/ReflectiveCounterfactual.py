from typing import Tuple
from Levenshtein import distance
import heapq
from collections import defaultdict

from CounterfactualGenerator import CounterfactualGenerator
from WrongPredictionError import WrongPredictionError
from SimpleExplainer import SimpleExplainer


class ReflectiveCounterfactual(CounterfactualGenerator):
    def __init__(self, explainer: SimpleExplainer, blackbox_name: str, max_iterations=5):
        super().__init__(explainer, blackbox_name)
        self.max_iterations = max_iterations
        self.counterfactual_iteration = defaultdict(int)

    def get_counterfactual_iteration(self):
        return self.counterfactual_iteration

    def get_counterfactual(self, sample, target) -> Tuple[str, bool, float, int]:
        previous_solutions = list()

        original_label, original_score = self.blackbox(sample)

        if original_label != target:
            raise WrongPredictionError

        for i in range(1, self.max_iterations+1):
            candidate_counterfactuals = self.explainer.explain(sample, original_label, previous_solutions)
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
            token_distance = 0
            similarity_score = 1.0
            counterfactual_score = original_score
            candidate_counterfactual = sample

            if candidate_counterfactuals:
                if counterfactual_found:  # find counterfactual with minimal token distance to original sample
                    while counterfactual_label == original_label:
                        token_distance, counterfactual_label, counterfactual_score, candidate_counterfactual, \
                            similarity_score = heapq.heappop(counterfactual_scores)
                    print(f"Counterfactual found in iteration {i}!")
                    self.counterfactual_iteration[str(i)] += 1

                    break
                else:  # no counterfactual found, return minimal token distance sample that is not the original sample
                    while candidate_counterfactuals and token_distance == 0:
                        token_distance, counterfactual_label, counterfactual_score, candidate_counterfactual, \
                            similarity_score, = heapq.heappop(counterfactual_scores)

                    previous_solutions.append((original_score - counterfactual_score, candidate_counterfactual))

        self.print_results(candidate_counterfactual, counterfactual_label, counterfactual_score, original_label,
                           original_score, sample, similarity_score, target, token_distance)

        return candidate_counterfactual, original_label != counterfactual_label, similarity_score, token_distance
