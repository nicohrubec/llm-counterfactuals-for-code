from BlackBox import BlackBox
from Explainer import Explainer
from SimilarityMetric import SimilarityMetric

label2target = {'LABEL_0': False, 'LABEL_1': True}


class FlareCounterfactual:
    def __init__(self, explainer: Explainer):
        self.blackbox = BlackBox()
        self.explainer = explainer
        self.similarity_score = SimilarityMetric()

    def get_counterfactual(self, sample, target):
        original_label, original_score = self.blackbox(sample)
        candidate_counterfactual = self.explainer.explain(sample, label2target[original_label])
        counterfactual_label, counterfactual_score = self.blackbox(candidate_counterfactual)
        similarity_score = float(self.similarity_score(sample, candidate_counterfactual)[0][0])

        print(f"The correct label is: {target}")
        print(f"Originally the model predicted {label2target[original_label]} with a confidence of {original_score}.")
        print(
            f"After applying the counterfactual the model predicted {label2target[counterfactual_label]} with a confidence of {counterfactual_score}.")
        print(f"Similarity score: {similarity_score:.{4}f}")

        return candidate_counterfactual, original_label != counterfactual_label, similarity_score
