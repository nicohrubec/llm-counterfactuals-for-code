from ExperimentRunner import ExperimentRunner
from helpers import get_dataset


class CloneExperimentRunner(ExperimentRunner):
    def get_dataset(self, n_samples, max_num_lines):
        return get_dataset("code_x_glue_cc_clone_detection_big_clone_bench",
                           n_samples=n_samples, max_num_lines=max_num_lines, filter_col='func1')

    def get_sample(self, dataset, idx):
        # random_sample = dataset.sample(random_state=2024)
        sample = "<s>" "</s> \n\n <s>".join([dataset.iloc[idx].func1, dataset.iloc[idx].func2]) + "</s>"
        target = dataset.iloc[idx].label

        return sample, target
