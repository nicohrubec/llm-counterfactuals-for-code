# LLM-based Counterfactual Explanations for Models of Code

This repository contains the code for my bachelor thesis titled "LLM-based Counterfactual Explanations for Models of Code". The project explores the use of large language models to generate counterfactual explanations for models predicting source code properties. Counterfactual explanations are minimal changes to the input that flip the prediction of a classifier.

The project focuses on two main tasks:

1. **Code Clone Detection**: Identifying whether two functions are semantically equivalent
2. **Code Vulnerability Detection**: Detecting potential security vulnerabilities in source code

## Overview

- Multiple counterfactual generation strategies:
  - One-shot generation
  - Multi-shot generation with masked tokens/lines
  - Reflective generation with feedback
- Support for different LLM backends:
  - Main experiments use GPT-3.5/4 but other models can be used by implementing new explainers
- Evaluation metrics:
  - Label flip success rate
  - Semantic similarity
  - Token-level edit distance
  - Runtime performance

## Setup

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set up your OpenAI API key as an environment variable (if you want to run the experiments with GPT models)

## Usage

Multiple experiments are set up in `src/main.py` as demonstrations.

Example of running an experiment:
```python3
from DefectExperimentRunner import DefectExperimentRunner
from SimpleGPTDefectExplainer import SimpleGPTDefectExplainer
from OneShotCounterfactual import OneShotCounterfactual

explainer = SimpleGPTDefectExplainer("gpt-3.5-turbo")
generator = OneShotCounterfactual(explainer, "uclanlp/plbart-c-cpp-defect-detection")

runner = DefectExperimentRunner(generator)
runner.run_experiment(n_samples=10)
```


## Architecture

The project follows a modular architecture with four main components:

- **BlackBox**: Interface for the model to be explained
- **Explainers**: Interface with LLMs to generate counterfactual suggestions
  - SimpleExplainer: Base class for one-shot generation
  - MaskedExplainer: Base class for masked token/line generation
- **Generators**: Implement different counterfactual generation strategies
  - OneShotCounterfactual: Direct generation
  - MultiShotCounterfactual: Iterative masked generation
  - ReflectiveCounterfactual: Generation with feedback loop
- **Experiment Runners**: Handle dataset loading and evaluation
  - DefectExperimentRunner: For vulnerability detection
  - CloneExperimentRunner: For clone detection

## Evaluation Metrics

The experiment runners track multiple metrics:
- Label flip success rate (overall and per class)
- Semantic similarity between original and counterfactual
- Token-level edit distance
- Runtime performance statistics
- One-shot success rate
- Average iterations until success

## License

This project is licensed under the MIT License - see the LICENSE file for details.
