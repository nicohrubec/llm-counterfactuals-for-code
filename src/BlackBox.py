from transformers import pipeline
from transformers import AutoTokenizer
from typing import Tuple
import torch


class BlackBox:
    def __init__(self, model_str):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if 'defect' in model_str:
            self.label2target = {'LABEL_0': False, 'LABEL_1': True}
        elif 'clone' in model_str:
            self.label2target = {'LABEL_0': True, 'LABEL_1': False}
        else:
            raise NotImplementedError

        t = AutoTokenizer.from_pretrained(model_str, use_fast=False, model_max_length=1024)
        self.pipeline = pipeline(model=model_str, tokenizer=t, device=device)

    def classify(self, document: str) -> Tuple[str, float]:
        output = self.pipeline([document], truncation=True)
        return self.label2target[output[0]['label']], output[0]['score']

    def __call__(self, document: str) -> Tuple[str, float]:
        return self.classify(document)
