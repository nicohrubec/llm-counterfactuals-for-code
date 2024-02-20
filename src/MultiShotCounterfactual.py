import heapq
from typing import Tuple
from Levenshtein import distance

from Explainer import Explainer
from Parser import Parser
from CounterfactualGenerator import CounterfactualGenerator
from WrongPredictionError import WrongPredictionError


class MultiShotCounterfactual(CounterfactualGenerator):
    line_mask = "<MASK>"
    one_shot_flipped = 0
    num_candidates_produced = 0

    def __init__(self, explainer: Explainer, blackbox_name: str, parser: Parser):
        super().__init__(explainer, blackbox_name)
        self.parser = parser

    def get_masked_program(self, split_program, idx):
        split_program[idx] = self.line_mask
        return self.parser.unparse(split_program)

    def unmask_program(self, program, replacement):
        return program.replace(self.line_mask, replacement)

    def replace_line(self, program, replacement, idx):
        program[idx] = replacement
        return "\n".join(program)

    def get_one_shot_flip_ratio(self):
        return self.one_shot_flipped / self.num_candidates_produced

    def update_one_shot_stats(self, flipped):
        if flipped: self.one_shot_flipped += 1
        self.num_candidates_produced += 1

    def get_counterfactual(self, sample, target) -> Tuple[str, bool, float, int]:
        original_label, original_score = self.blackbox(sample)

        if original_label != target:
            raise WrongPredictionError

        parsed_sample = self.parser.parse(sample)
        potential_counterfactuals = []
        heapq.heapify(potential_counterfactuals)

        for idx, line in enumerate(parsed_sample):
            if idx == 0 or len(line.strip()) == 1:
                continue

            masked_program = self.get_masked_program(parsed_sample.copy(), idx)
            potential_counterfactual = self.explainer.explain(masked_program, original_label)
            unmasked_program = self.unmask_program(masked_program, potential_counterfactual)
            counterfactual_label, counterfactual_score = self.blackbox(unmasked_program)

            # high confident samples with different labels should be considered first
            flipped = counterfactual_label != original_label
            if flipped:
                counterfactual_score = 1 - counterfactual_score

            self.update_one_shot_stats(flipped)
            heapq.heappush(potential_counterfactuals, (counterfactual_score, potential_counterfactual, idx))

        parsed_counterfactual_program = parsed_sample.copy()
        counterfactual_score = original_score
        while potential_counterfactuals:
            _, potential_counterfactual, idx = heapq.heappop(potential_counterfactuals)

            counterfactual_program = self.replace_line(parsed_counterfactual_program.copy(), potential_counterfactual, idx)
            prev_counterfactual_score = counterfactual_score
            counterfactual_label, counterfactual_score = self.blackbox(counterfactual_program)

            flipped = counterfactual_label != original_label
            if flipped:
                counterfactual_score = 1 - counterfactual_score

            # check if last step has brought us closer to a solution
            if counterfactual_score < prev_counterfactual_score:
                parsed_counterfactual_program[idx] = potential_counterfactual
            else:
                counterfactual_score = prev_counterfactual_score

            if flipped:  # counterfactual found
                proposed_counterfactual, similarity_score, token_distance = \
                    self.report_results(counterfactual_label, 1 - counterfactual_score, original_label, original_score,
                                        parsed_counterfactual_program, sample, target)

                return proposed_counterfactual, True, similarity_score, token_distance

        proposed_counterfactual, similarity_score, token_distance = \
            self.report_results(original_label, counterfactual_score, original_label, original_score,
                                parsed_counterfactual_program, sample, target)

        return proposed_counterfactual, False, similarity_score, token_distance

    def report_results(self, counterfactual_label, counterfactual_score, original_label, original_score,
                       parsed_counterfactual_program, sample, target):
        proposed_counterfactual = self.parser.unparse(parsed_counterfactual_program)
        similarity_score = float(self.similarity_score(sample, proposed_counterfactual)[0][0])
        token_distance = distance(sample, proposed_counterfactual)

        print(f"The correct label is: {target}")
        print(
            f"Originally the model predicted {original_label} with a confidence of {original_score}.")
        print(
            f"After applying the counterfactual the model predicted {counterfactual_label} "
            f"with a confidence of {counterfactual_score}.")
        print(f"Similarity score: {similarity_score:.{4}f}")
        print(f"Token distance: {token_distance}")
        print(f"Original sample:\n{sample}\n\nProposed counterfactual:\n{proposed_counterfactual}")
        print()
        return proposed_counterfactual, similarity_score, token_distance
