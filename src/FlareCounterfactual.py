from src.BlackBox import BlackBox
from src.explainers import Explainer

label2target = {'LABEL_0': False, 'LABEL_1': True}


class FlareCounterfactual:
    def __init__(self, explainer: Explainer):
        self.blackbox = BlackBox()
        self.explainer = explainer

    def get_counterfactual(self, sample, target):
        original_label, original_score = self.blackbox(sample)
        candidate_counterfactual = self.explainer.explain(sample, target)
        counterfactual_label, counterfactual_score = self.blackbox(candidate_counterfactual)

        print(f"The correct label is: {target}")
        print(f"Originally the model predicted {label2target[original_label]} with a confidence of {original_score}.")
        print(
            f"After applying the counterfactual the model predicted {label2target[counterfactual_label]} with a confidence of {counterfactual_score}.")

        return candidate_counterfactual, original_label != counterfactual_label
