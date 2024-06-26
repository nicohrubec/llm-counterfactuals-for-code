from SimpleGPTDefectExplainer import SimpleGPTDefectExplainer
from SimpleGPTCloneExplainer import SimpleGPTCloneExplainer
from OneShotCounterfactual import OneShotCounterfactual
from MultiShotCounterfactual import MultiShotCounterfactual
from ReflectiveCounterfactual import ReflectiveCounterfactual
from MaskedGPTDefectExplainer import MaskedGPTDefectExplainer
from MaskedGPTCloneExplainer import MaskedGPTCloneExplainer
from MaskedGPTDefectTokenExplainer import MaskedGPTDefectTokenExplainer
from MaskedGPTCloneTokenExplainer import MaskedGPTCloneTokenExplainer
from LineParser import LineParser
from TokenParser import TokenParser
from DefectExperimentRunner import DefectExperimentRunner
from CloneExperimentRunner import CloneExperimentRunner

model_str = "gpt-3.5-turbo"
defect_blackbox_str = "uclanlp/plbart-c-cpp-defect-detection"
clone_blackbox_str = "uclanlp/plbart-java-clone-detection"

if __name__ == '__main__':
    line_parser = LineParser()
    token_parser = TokenParser()
    gpt_explainer = MaskedGPTCloneExplainer(model_str)
    clone_multi_shot_line_counterfactual_generator = MultiShotCounterfactual(gpt_explainer, clone_blackbox_str,
                                                                             line_parser)
    gpt_explainer = MaskedGPTCloneTokenExplainer(model_str)
    clone_multi_shot_token_counterfactual_generator = MultiShotCounterfactual(gpt_explainer, clone_blackbox_str,
                                                                              token_parser)

    gpt_explainer = SimpleGPTCloneExplainer(model_str)
    clone_single_shot_counterfactual_generator = OneShotCounterfactual(gpt_explainer, clone_blackbox_str)
    clone_reflective_counterfactual_generator = ReflectiveCounterfactual(gpt_explainer, clone_blackbox_str)

    gpt_explainer = SimpleGPTCloneExplainer(model_str, num_counterfactuals=3)
    clone_single_shot_multi_counterfactual_generator = OneShotCounterfactual(gpt_explainer, clone_blackbox_str)
    clone_reflective_multi_counterfactual_generator = ReflectiveCounterfactual(gpt_explainer, clone_blackbox_str)

    line_parser = LineParser()
    token_parser = TokenParser()
    gpt_explainer = MaskedGPTDefectExplainer(model_str)
    defect_multi_shot_line_counterfactual_generator = MultiShotCounterfactual(gpt_explainer, defect_blackbox_str,
                                                                              line_parser)
    gpt_explainer = MaskedGPTDefectTokenExplainer(model_str)
    defect_multi_shot_token_counterfactual_generator = MultiShotCounterfactual(gpt_explainer, defect_blackbox_str,
                                                                               token_parser)

    gpt_explainer = SimpleGPTDefectExplainer(model_str)
    defect_single_shot_counterfactual_generator = OneShotCounterfactual(gpt_explainer, defect_blackbox_str)
    defect_reflective_counterfactual_generator = ReflectiveCounterfactual(gpt_explainer, defect_blackbox_str)

    gpt_explainer = SimpleGPTDefectExplainer(model_str, num_counterfactuals=3)
    defect_single_shot_multi_counterfactual_generator = OneShotCounterfactual(gpt_explainer, defect_blackbox_str)
    defect_reflective_multi_counterfactual_generator = ReflectiveCounterfactual(gpt_explainer, defect_blackbox_str)

    print("Clone Multi shot line results: ")
    CloneExperimentRunner(clone_multi_shot_line_counterfactual_generator).run_experiment(n_samples=1)
    print()
    print("Clone Multi shot token results: ")
    CloneExperimentRunner(clone_multi_shot_token_counterfactual_generator).run_experiment(n_samples=1)
    print()
    print("Clone One shot results: ")
    CloneExperimentRunner(clone_single_shot_counterfactual_generator).run_experiment(n_samples=1)
    print()
    print("Clone One shot N=3 results: ")
    CloneExperimentRunner(clone_single_shot_multi_counterfactual_generator).run_experiment(n_samples=1)
    print()
    print("Defect Multi shot line results: ")
    DefectExperimentRunner(defect_multi_shot_line_counterfactual_generator).run_experiment(n_samples=1)
    print()
    print("Defect Multi shot token results: ")
    DefectExperimentRunner(defect_multi_shot_token_counterfactual_generator).run_experiment(n_samples=1)
    print()
    print("Defect One shot results: ")
    DefectExperimentRunner(defect_single_shot_counterfactual_generator).run_experiment(n_samples=1)
    print()
    print("Defect One shot N=3 results: ")
    DefectExperimentRunner(defect_single_shot_multi_counterfactual_generator).run_experiment(n_samples=1)
    print()
    print("Defect Reflection N=1 results: ")
    DefectExperimentRunner(defect_reflective_counterfactual_generator).run_experiment(n_samples=1)
    print()
    print("Defect Reflection N=3 results: ")
    DefectExperimentRunner(defect_reflective_multi_counterfactual_generator).run_experiment(n_samples=1)
    print()
