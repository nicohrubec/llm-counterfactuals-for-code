from GPTExplainer import GPTExplainer
from FlareCounterfactual import FlareCounterfactual
from helpers import get_dataset

if __name__ == '__main__':
    dataset = get_dataset()
    gpt_explainer = GPTExplainer("gpt-3.5-turbo-1106")
    counterfactual_generator = FlareCounterfactual(explainer=gpt_explainer)

    sample = dataset.iloc[0].func
    target = dataset.iloc[0].target

    counterfactual_generator.get_counterfactual(sample, target)
