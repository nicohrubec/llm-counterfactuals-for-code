from GPTExplainer import GPTExplainer
from GPTCoTExplainer import GPTCoTExplainer
from OneShotCounterfactual import OneShotCounterfactual
from helpers import get_dataset

model_str = "gpt-3.5-turbo-1106"

if __name__ == '__main__':
    dataset = get_dataset()
    gpt_explainer = GPTExplainer(model_str)
    counterfactual_generator_flare = OneShotCounterfactual(explainer=gpt_explainer)

    gpt_cot_explainer = GPTCoTExplainer(model_str)
    counterfactual_generator_cot = OneShotCounterfactual(explainer=gpt_cot_explainer)

    sample = dataset.iloc[0].func
    target = dataset.iloc[0].target

    print("One shot results: ")
    counterfactual_generator_flare.get_counterfactual(sample, target)
    print("Cot results:")
    counterfactual_generator_cot.get_counterfactual(sample, target)

