from CounterfactualGenerator import CounterfactualGenerator
from WrongPredictionError import WrongPredictionError
from helpers import remove_empty_lines, remove_comments

import traceback
from time import perf_counter
from statistics import stdev


class ExperimentRunner:
    def __init__(self, counterfactual_generator: CounterfactualGenerator):
        self.counterfactual_generator = counterfactual_generator

    def get_dataset(self, n_samples, max_num_lines):
        raise NotImplementedError

    def get_sample(self, dataset, idx):
        raise NotImplementedError

    def report_results(self, flippeds, idx, num_mispredictions, similarities, token_distances, targets, times):
        similarities = [v for v in similarities if v is not None]
        counterfactual_similarities = [v for idx, v in enumerate(similarities) if flippeds[idx]]
        counterfactual_distances = [v for idx, v in enumerate(token_distances) if flippeds[idx]]
        true_labels_flipped = [flipped for idx, flipped in enumerate(flippeds) if targets[idx]]
        false_labels_flipped = [flipped for idx, flipped in enumerate(flippeds) if not targets[idx]]
        counterfactual_times = [t for idx, t in enumerate(times) if flippeds[idx]]

        # make results accessible outside of class
        self.similarities = similarities
        self.counterfactual_similarities = counterfactual_similarities
        self.token_distances = token_distances
        self.counterfactual_token_distances = counterfactual_distances
        self.times = times

        # report results
        try:
            label_flip_score = sum(flippeds) / len(flippeds)
            print(f"Experiment label flip score: {label_flip_score:.{2}f}")

            if len(true_labels_flipped) > 0:
                true_label_flip_score = sum(true_labels_flipped) / len(true_labels_flipped)
                print(f"Experiment label flip score for true labels: {true_label_flip_score:.{2}f}")
                print("Number of True labels encountered: ", len(true_labels_flipped))
            else:
                print("No true labels were evaluated in this experiment!")

            if len(false_labels_flipped) > 0:
                false_label_flip_score = sum(false_labels_flipped) / len(false_labels_flipped)
                print(f"Experiment label flip score for false labels: {false_label_flip_score:.{2}f}")
                print("Number of False labels encountered: ", len(false_labels_flipped))
            else:
                print("No false labels were evaluated in this experiment!")

            if len(similarities) > 0:
                similarity_score = sum(similarities) / len(similarities)
                print(f"Experiment similarity score: {similarity_score:.{4}f}")
            else:
                print("No similarity score was reported!")

            if len(counterfactual_similarities) > 0:
                counterfactual_similarity_score = sum(counterfactual_similarities) / len(counterfactual_similarities)
                print(f"Experiment counterfactual similarity score: {counterfactual_similarity_score:.{4}f}")
            else:
                print("No counterfactuals were found in this experiment!")

            if len(token_distances) > 0:
                avg_token_distance = sum(token_distances) / len(token_distances)
                print("Experiment token distance: ", avg_token_distance)
            else:
                print("No token distance was reported!")

            if len(counterfactual_distances) > 0:
                avg_counterfactual_token_distance = sum(counterfactual_distances) / len(counterfactual_distances)
                print("Experiment counterfactual token distance: ", avg_counterfactual_token_distance)

            print("Blackbox Accuracy: ", 1 - num_mispredictions / idx)

            if len(times) > 0:
                avg_runtime_in_s = sum(times) / len(times)
                print(f"Experiment avg runtime: {avg_runtime_in_s:.{2}f}")

                if len(times) > 1:
                    std_runtime_in_s = stdev(times)
                    print(f"Experiment std runtime: {std_runtime_in_s:.{2}f}")
            else:
                print("No runtime was reported!")

            if len(counterfactual_times) > 0:
                avg_counterfactual_runtime_in_s = sum(counterfactual_times) / len(counterfactual_times)
                print(f"Experiment avg counterfactual runtime: {avg_counterfactual_runtime_in_s:.{2}f}")

                if len(counterfactual_times) > 1:
                    std_counterfactual_runtime_in_s = stdev(counterfactual_times)
                    print(f"Experiment std counterfactual runtime: {std_counterfactual_runtime_in_s:.{2}f}")

        except ZeroDivisionError:
            print("No results available, all samples failed!")
        # very hacky but needed to print the one shot flip ratio after experiments which is only defined for the
        # MultiShotCounterfactual generator
        # TODO: should be moved but not sure yet where to
        try:
            one_shot_flip_score = self.counterfactual_generator.get_one_shot_flip_ratio()
            print(f"Experiment one shot flip score: {one_shot_flip_score:.{2}f}", )
        except AttributeError:
            pass

        try:
            print("Experiment found counterfactuals in iteration: ",
                  self.counterfactual_generator.get_counterfactual_iteration())
            print("Experiment found counterfactuals in iteration token distance: ",
                  self.counterfactual_generator.get_counterfactual_iteration_token_distance())
        except AttributeError:
            pass

    def run_experiment(self, n_samples=250, max_num_lines=50):
        # get more samples than we actually need in case of mispredictions
        dataset = self.get_dataset(n_samples * 4, max_num_lines)

        counterfactuals = []
        flippeds = []
        similarities = []
        targets = []
        token_distances = []
        times = []
        samples_done = 0
        idx = 0
        num_mispredictions = 0

        while samples_done < n_samples:
            try:
                print("Iteration ", samples_done + 1)
                sample, target = self.get_sample(dataset, idx=idx)
                sample = remove_empty_lines(sample)
                sample = remove_comments(sample)

                start_time = perf_counter()
                counterfactual, flipped, similarity, token_distance = \
                    self.counterfactual_generator.get_counterfactual(sample, target)
                end_time = perf_counter()

                # track results
                counterfactuals.append(counterfactual)
                flippeds.append(flipped)
                similarities.append(similarity)
                token_distances.append(token_distance)
                targets.append(target)

                # get and report runtime
                runtime_in_s = end_time - start_time
                times.append(runtime_in_s)
                print(f"Runtime in s: {runtime_in_s:.{2}f}")
                print()

                samples_done += 1
            except WrongPredictionError:
                num_mispredictions += 1
                print("Skipping sample because the blackbox mispredicted!")
            except:
                traceback.print_exc()
                samples_done += 1
                print("Something went wrong!")

            idx += 1

        self.report_results(flippeds, idx, num_mispredictions, similarities, token_distances, targets, times)
