from CounterfactualGenerator import CounterfactualGenerator


class ExperimentRunner:
    def __init__(self, counterfactual_generator: CounterfactualGenerator):
        self.counterfactual_generator = counterfactual_generator

    def get_dataset(self, n_samples, max_num_lines):
        raise NotImplementedError

    def get_sample(self, dataset, idx):
        raise NotImplementedError

    def run_experiment(self, n_samples=30, max_num_lines=25):
        dataset = self.get_dataset(n_samples, max_num_lines)

        counterfactuals = []
        flippeds = []
        similarities = []

        for i in range(n_samples):
            try:
                print("Iteration ", i + 1)
                sample, target = self.get_sample(dataset, idx=i)

                counterfactual, flipped, similarity = self.counterfactual_generator.get_counterfactual(sample, target)
                counterfactuals.append(counterfactual)
                flippeds.append(flipped)
                similarities.append(similarity)
            except:
                continue

        similarities = [v for v in similarities if v is not None]
        counterfactual_similarities = [v for idx, v in enumerate(similarities) if flippeds[idx]]

        try:
            print("Experiment label flip score: ", sum(flippeds) / len(flippeds))
            print("Experiment similarity score: ", sum(similarities) / len(similarities))
            print("Experiment counterfactual similarity score: ",
                  sum(counterfactual_similarities) / len(counterfactual_similarities))
        except ZeroDivisionError:
            print("No counterfactuals were found in this experiment!")

        # very hacky but needed to print the one shot flip ratio after experiments which is only defined for the
        # MultiShotCounterfactual generator
        # TODO: should be moved but not sure yet where to
        try:
            print("Experiment one shot flip score: ", self.counterfactual_generator.get_one_shot_flip_ratio())
        except AttributeError:
            pass