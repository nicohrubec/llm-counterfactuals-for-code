from BlackBox import BlackBox
from Explainer import Explainer
from SimilarityMetric import SimilarityMetric
from helpers import get_dataset

label2target = {'LABEL_0': False, 'LABEL_1': True}


class OneShotCounterfactual:
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
        print()

        return candidate_counterfactual, original_label != counterfactual_label, similarity_score

    def run_experiment(self, n_samples=30):
        dataset = get_dataset()

        counterfactuals = []
        flippeds = []
        similarities = []

        for i in range(n_samples):
            try:
                print("Idx ", i)
                sample = dataset.iloc[i].func
                target = dataset.iloc[i].target

                counterfactual, flipped, similarity = self.get_counterfactual(sample, target)
                counterfactuals.append(counterfactual)
                flippeds.append(flipped)
                similarities.append(similarity)
            except:
                continue

            similarities = [v for v in similarities if v is not None]

            print("Experiment label flip score: ", sum(flippeds) / len(flippeds))
            print("Experiment similarity score: ", sum(similarities) / len(similarities))
            print()
