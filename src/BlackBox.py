from transformers import pipeline
from transformers import AutoTokenizer
from typing import Tuple


class BlackBox:
    def __init__(self):
        t = AutoTokenizer.from_pretrained("uclanlp/plbart-c-cpp-defect-detection", use_fast=False)
        self.defect_pipeline = pipeline(model="uclanlp/plbart-c-cpp-defect-detection", tokenizer=t)

    def classify(self, document: str) -> Tuple[str, float]:
        output = self.defect_pipeline([document])
        return output[0]['label'], output[0]['score']

    def __call__(self, document: str) -> Tuple[str, float]:
        return self.classify(document)
