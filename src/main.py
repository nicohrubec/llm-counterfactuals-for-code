from SimpleGPTDefectExplainer import SimpleGPTDefectExplainer
from SimpleGPTCloneExplainer import SimpleGPTCloneExplainer
from CoTGPTDefectExplainer import CoTGPTDefectExplainer
from CoTGPTCloneExplainer import CoTGPTCloneExplainer
from OneShotCounterfactual import OneShotCounterfactual
from MultiShotCounterfactual import MultiShotCounterfactual
from MaskedGPTDefectExplainer import MaskedGPTDefectExplainer
from MaskedGPTCloneExplainer import MaskedGPTCloneExplainer
from LineParser import LineParser
from DefectExperimentRunner import DefectExperimentRunner
from CloneExperimentRunner import CloneExperimentRunner

model_str = "gpt-3.5-turbo"
defect_blackbox_str = "uclanlp/plbart-c-cpp-defect-detection"
clone_blackbox_str = "uclanlp/plbart-java-clone-detection"

if __name__ == '__main__':
    line_parser = LineParser()
    gpt_explainer = MaskedGPTCloneExplainer(model_str)
    clone_multi_shot_counterfactual_generator = MultiShotCounterfactual(gpt_explainer, clone_blackbox_str, line_parser)

    gpt_explainer = SimpleGPTCloneExplainer(model_str)
    clone_single_shot_counterfactual_generator = OneShotCounterfactual(gpt_explainer, clone_blackbox_str)

    gpt_cot_explainer = CoTGPTCloneExplainer(model_str)
    clone_single_shot_cot_counterfactual_generator = OneShotCounterfactual(gpt_cot_explainer, clone_blackbox_str)

    line_parser = LineParser()
    gpt_explainer = MaskedGPTDefectExplainer(model_str)
    defect_multi_shot_counterfactual_generator = MultiShotCounterfactual(gpt_explainer, defect_blackbox_str, line_parser)

    gpt_explainer = SimpleGPTDefectExplainer(model_str)
    defect_single_shot_counterfactual_generator = OneShotCounterfactual(gpt_explainer, defect_blackbox_str)

    gpt_cot_explainer = CoTGPTDefectExplainer(model_str)
    defect_single_shot_cot_counterfactual_generator = OneShotCounterfactual(gpt_cot_explainer, defect_blackbox_str)

    print("Clone Multi shot results: ")
    CloneExperimentRunner(clone_multi_shot_counterfactual_generator).run_experiment(n_samples=1)
    print()
    print("Clone One shot results: ")
    CloneExperimentRunner(clone_single_shot_counterfactual_generator).run_experiment(n_samples=1)
    print()
    print("Clone Cot results:")
    CloneExperimentRunner(clone_single_shot_cot_counterfactual_generator).run_experiment(n_samples=1)
    print()
    print("Defect Multi shot results: ")
    DefectExperimentRunner(defect_multi_shot_counterfactual_generator).run_experiment(n_samples=1)
    print()
    print("Defect One shot results: ")
    DefectExperimentRunner(defect_single_shot_counterfactual_generator).run_experiment(n_samples=1)
    print()
    print("Defect Cot results:")
    DefectExperimentRunner(defect_single_shot_cot_counterfactual_generator).run_experiment(n_samples=1)
