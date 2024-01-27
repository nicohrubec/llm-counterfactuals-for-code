from transformers import pipeline
from transformers import AutoTokenizer
from typing import Tuple


class BlackBox:
    def __init__(self, model_str):
        t = AutoTokenizer.from_pretrained(model_str, use_fast=False)
        self.defect_pipeline = pipeline(model=model_str, tokenizer=t)

    def classify(self, document: str) -> Tuple[str, float]:
        # TODO: check type and if not list then wrap in list
        output = self.defect_pipeline([document])
        return output[0]['label'], output[0]['score']

    def __call__(self, document: str) -> Tuple[str, float]:
        return self.classify(document)
