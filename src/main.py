from GPTExplainer import GPTExplainer
from GPTCoTExplainer import GPTCoTExplainer
from OneShotCounterfactual import OneShotCounterfactual
from MultiShotCounterfactual import MultiShotCounterfactual
from MaskedGPTExplainer import MaskedGPTExplainer
from LineParser import LineParser
from helpers import get_dataset

model_str = "gpt-3.5-turbo-1106"

if __name__ == '__main__':
    dataset = get_dataset()

    line_parser = LineParser()
    gpt_explainer = MaskedGPTExplainer(model_str)
    counterfactual_generator = MultiShotCounterfactual(gpt_explainer, line_parser)

    gpt_explainer = GPTExplainer(model_str)
    counterfactual_generator_flare = OneShotCounterfactual(explainer=gpt_explainer)

    gpt_cot_explainer = GPTCoTExplainer(model_str)
    counterfactual_generator_cot = OneShotCounterfactual(explainer=gpt_cot_explainer)

    print("Multi shot results: ")
    counterfactual_generator.run_experiment(n_samples=1)
    print("One shot results: ")
    counterfactual_generator_flare.run_experiment(n_samples=1)
    print("Cot results:")
    counterfactual_generator_cot.run_experiment(n_samples=1)
