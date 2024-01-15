import heapq

from BlackBox import BlackBox
from Explainer import Explainer
from SimilarityMetric import SimilarityMetric
from Parser import Parser

label2target = {'LABEL_0': False, 'LABEL_1': True}


class MultiShotCounterfactual:
    line_mask = "<MASK>"

    def __init__(self, explainer: Explainer, parser: Parser):
        self.blackbox = BlackBox()
        self.explainer = explainer
        self.similarity_score = SimilarityMetric()
        self.parser = parser

    def get_masked_program(self, split_program, idx):
        original_line = split_program[idx]
        split_program[idx] = self.line_mask
        return self.parser.unparse(split_program), original_line

    def unmask_program(self, program, replacement, idx):
        program[idx] = replacement
        return "\n".join(program)

    def get_counterfactual(self, sample, target):
        original_label, original_score = self.blackbox(sample)
        parsed_sample = self.parser.parse(sample)
        potential_counterfactuals = []
        heapq.heapify(potential_counterfactuals)
        explanation = []

        for idx, line in enumerate(parsed_sample):
            if not len(line.strip()) > 1:
                continue

            masked_program, original_line = self.get_masked_program(parsed_sample.copy(), idx)
            potential_counterfactual = self.explainer.explain(masked_program, label2target[original_label])
            unmasked_program = self.unmask_program(self.parser.parse(masked_program), potential_counterfactual, idx)
            counterfactual_label, counterfactual_score = self.blackbox(unmasked_program)

            # high confident samples with different labels should be considered first
            if counterfactual_label != original_label:
                counterfactual_score = 1 - counterfactual_score

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
                explanation.append((potential_counterfactual, idx))
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

                return explanation, True, similarity_score

        print(f"The correct label is: {target}")
        print(f"Originally the model predicted {label2target[original_label]} with a confidence of {original_score}.")
        print("No counterfactual was found!")

        return explanation, False, None

