class Explainer:
    system_prompt = "You are an oracle explanation module in a machine learning pipeline."

    def explain(self, sample: str, prediction: bool) -> str:
        raise NotImplementedError
