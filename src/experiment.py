from CounterfactualGenerator import CounterfactualGenerator

from src.helpers import get_dataset


def run_defect_experiment(counterfactual_generator: CounterfactualGenerator, n_samples=30, max_num_lines=25):
    dataset = get_dataset(max_num_lines=max_num_lines)

    counterfactuals = []
    flippeds = []
    similarities = []

    for i in range(n_samples):
        try:
            print("Iteration ", i + 1)
            random_sample = dataset.sample(random_state=2024)
            sample = random_sample.iloc[0].func
            target = random_sample.iloc[0].target

            counterfactual, flipped, similarity = counterfactual_generator.get_counterfactual(sample, target)
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
        print("Experiment one shot flip score: ", counterfactual_generator.get_one_shot_flip_ratio())
    except AttributeError:
        pass
