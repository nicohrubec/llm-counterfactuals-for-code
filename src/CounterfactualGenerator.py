from helpers import get_dataset
from typing import Tuple


class CounterfactualGenerator:
    def get_counterfactual(self, sample, target) -> Tuple[str, bool, float]:
        pass

    def run_experiment(self, n_samples=30, max_num_lines=25):
        dataset = get_dataset(max_num_lines=max_num_lines)

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
        counterfactual_similarities = [v for idx, v in enumerate(similarities) if flippeds[idx]]

        print("Experiment label flip score: ", sum(flippeds) / len(flippeds))
        print("Experiment similarity score: ", sum(similarities) / len(similarities))
        print("Experiment counterfactual similarity score: ",
              sum(counterfactual_similarities) / len(counterfactual_similarities))

