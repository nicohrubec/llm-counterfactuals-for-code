from transformers import pipeline
from transformers import AutoTokenizer
from typing import Tuple


class BlackBox:
    def __init__(self, model_str):
        if 'defect' in model_str:
            self.label2target = {'LABEL_0': False, 'LABEL_1': True}
        elif 'clone' in model_str:
            self.label2target = {'LABEL_0': True, 'LABEL_1': False}
        else:
            raise NotImplementedError

        t = AutoTokenizer.from_pretrained(model_str, use_fast=False)
        self.pipeline = pipeline(model=model_str, tokenizer=t)

    def classify(self, document: str) -> Tuple[str, float]:
        output = self.pipeline([document])
        return self.label2target[output[0]['label']], output[0]['score']

    def __call__(self, document: str) -> Tuple[str, float]:
        return self.classify(document)
