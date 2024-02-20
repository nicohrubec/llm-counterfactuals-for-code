from CounterfactualGenerator import CounterfactualGenerator
from WrongPredictionError import WrongPredictionError
import traceback


class ExperimentRunner:
    def __init__(self, counterfactual_generator: CounterfactualGenerator):
        self.counterfactual_generator = counterfactual_generator

    def get_dataset(self, n_samples, max_num_lines):
        raise NotImplementedError

    def get_sample(self, dataset, idx):
        raise NotImplementedError

    def report_results(self, flippeds, idx, num_mispredictions, similarities, token_distances, targets):
        similarities = [v for v in similarities if v is not None]
        counterfactual_similarities = [v for idx, v in enumerate(similarities) if flippeds[idx]]
        counterfactual_distances = [v for idx, v in enumerate(similarities) if flippeds[idx]]
        true_labels_flipped = [flipped for idx, flipped in enumerate(flippeds) if targets[idx]]
        false_labels_flipped = [flipped for idx, flipped in enumerate(flippeds) if not targets[idx]]
        # report results
        try:
            print("Experiment label flip score: ", sum(flippeds) / len(flippeds))

            if len(true_labels_flipped) > 0:
                print("Experiment label flip score for true labels: ",
                      sum(true_labels_flipped) / len(true_labels_flipped))
                print("Number of True labels encountered: ", len(true_labels_flipped))
            else:
                print("No true labels were evaluated in this experiment!")

            if len(false_labels_flipped) > 0:
                print("Experiment label flip score for false labels: ",
                      sum(false_labels_flipped) / len(false_labels_flipped))
                print("Number of False labels encountered: ", len(false_labels_flipped))
            else:
                print("No false labels were evaluated in this experiment!")

            if len(similarities) > 0:
                print("Experiment similarity score: ", sum(similarities) / len(similarities))
            else:
                print("No similarity score was reported!")

            if len(counterfactual_similarities) > 0:
                print("Experiment counterfactual similarity score: ",
                      sum(counterfactual_similarities) / len(counterfactual_similarities))
            else:
                print("No counterfactuals were found in this experiment!")

            if len(token_distances) > 0:
                print("Experiment token distance: ", sum(token_distances) / len(token_distances))
            else:
                print("No token distance was reported!")

            if len(counterfactual_distances) > 0:
                print("Experiment counterfactual token distance: ",
                      sum(counterfactual_distances) / len(counterfactual_distances))

            print("Blackbox Accuracy: ", 1 - num_mispredictions / idx)
        except ZeroDivisionError:
            print("No results available, all samples failed!")
        # very hacky but needed to print the one shot flip ratio after experiments which is only defined for the
        # MultiShotCounterfactual generator
        # TODO: should be moved but not sure yet where to
        try:
            print("Experiment one shot flip score: ", self.counterfactual_generator.get_one_shot_flip_ratio())
        except AttributeError:
            pass

    def run_experiment(self, n_samples=30, max_num_lines=25):
        dataset = self.get_dataset(n_samples * 10, max_num_lines)

        counterfactuals = []
        flippeds = []
        similarities = []
        targets = []
        token_distances = []
        samples_done = 0
        idx = 0
        num_mispredictions = 0

        while samples_done < n_samples:
            try:
                print("Iteration ", samples_done + 1)
                sample, target = self.get_sample(dataset, idx=idx)

                counterfactual, flipped, similarity, token_distance = \
                    self.counterfactual_generator.get_counterfactual(sample, target)

                # track results
                counterfactuals.append(counterfactual)
                flippeds.append(flipped)
                similarities.append(similarity)
                token_distances.append(token_distance)
                targets.append(target)

                samples_done += 1
            except WrongPredictionError:
                num_mispredictions += 1
                print("Skipping sample because the blackbox mispredicted!")
            except:
                traceback.print_exc()
                samples_done += 1
                print("Something went wrong!")

            idx += 1

        self.report_results(flippeds, idx, num_mispredictions, similarities, token_distances, targets)
