import heapq
from typing import Tuple

from Explainer import Explainer
from Parser import Parser
from CounterfactualGenerator import CounterfactualGenerator

label2target = {'LABEL_0': False, 'LABEL_1': True}


class MultiShotCounterfactual(CounterfactualGenerator):
    line_mask = "<MASK>"
    one_shot_flipped = 0
    num_candidates_produced = 0

    def __init__(self, explainer: Explainer, blackbox_name: str, parser: Parser):
        super().__init__(explainer, blackbox_name)
        self.parser = parser

    def get_masked_program(self, split_program, idx):
        original_line = split_program[idx]
        split_program[idx] = self.line_mask
        return self.parser.unparse(split_program), original_line

    def unmask_program(self, program, replacement, idx):
        program[idx] = replacement
        return "\n".join(program)

    def get_one_shot_flip_ratio(self):
        return self.one_shot_flipped / self.num_candidates_produced

    def update_one_shot_stats(self, flipped):
        if flipped: self.one_shot_flipped += 1
        self.num_candidates_produced += 1

    def get_counterfactual(self, sample, target) -> Tuple[str, bool, float]:
        original_label, original_score = self.blackbox(sample)
        parsed_sample = self.parser.parse(sample)
        potential_counterfactuals = []
        heapq.heapify(potential_counterfactuals)

        for idx, line in enumerate(parsed_sample):
            if not len(line.strip()) > 1:
                continue

            masked_program, original_line = self.get_masked_program(parsed_sample.copy(), idx)
            potential_counterfactual = self.explainer.explain(masked_program, label2target[original_label])
            unmasked_program = self.unmask_program(self.parser.parse(masked_program), potential_counterfactual, idx)
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

            counterfactual_program = self.unmask_program(parsed_counterfactual_program, potential_counterfactual, idx)
            prev_counterfactual_score = counterfactual_score
            counterfactual_label, counterfactual_score = self.blackbox(counterfactual_program)

            # check if last step has brought us closer to a solution
            if counterfactual_score < prev_counterfactual_score:
                parsed_counterfactual_program[idx] = potential_counterfactual
            else:
                counterfactual_score = prev_counterfactual_score

            if counterfactual_label != original_label:  # counterfactual found
                similarity_score = float(self.similarity_score(sample, counterfactual_program)[0][0])

                print(f"The correct label is: {target}")
                print(
                    f"Originally the model predicted {label2target[original_label]} with a confidence of {original_score}.")
                print(
                    f"After applying the counterfactual the model predicted {label2target[counterfactual_label]} "
                    f"with a confidence of {counterfactual_score}.")
                print(f"Similarity score: {similarity_score:.{4}f}")
                print()

                return counterfactual_program, True, similarity_score

        print(f"The correct label is: {target}")
        print(f"Originally the model predicted {label2target[original_label]} with a confidence of {original_score}.")
        print("No counterfactual was found!")
        print()

        return counterfactual_program, False, None
