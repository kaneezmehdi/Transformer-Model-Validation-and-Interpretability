# Transformer-Model-Validation-and-Interpretability

Pythia Model Validation with Activation Hooks

This repository contains my implementation of a validation utility for the Pythia language model. The project focuses on capturing and verifying internal activations within the model to better understand the flow of computations across embedding, normalization, attention, and MLP layers.

Objective
The goal of this project was to gain a deeper understanding of transformer architectures by implementing forward hooks, collecting intermediate outputs, and verifying their consistency against manually computed tensors. This approach provides transparency into the hidden states of large language models and supports debugging, interpretability, and research-oriented analysis.

Key Features

* Tokenization of text input and conversion into embeddings.
* Registration of forward hooks at multiple points in the first transformer block, including residual connections, layer normalization, attention output, and MLP output.
* Storage of intermediate activations for later inspection and verification.
* Manual recomputation of selected tensors to compare against hook outputs.
* Reporting of shapes and consistency checks for captured versus recomputed activations.

Technical Highlights

* Implemented with PyTorch and the Pythia model framework.
* Demonstrates low-level handling of hooks for interpretability in transformer networks.
* Provides a foundation for debugging and research in model internals, including error tracing and transparency analysis.
* Verifies operations such as residual addition, normalization, and MLP transformations.

Repository Contents

* validate.py – The main script containing the validation function.
* Example code demonstrating how to run the validation on a given text prompt.
* Documentation describing the verification process and the role of each hook.

How to Run

1. Install dependencies including torch and the Pythia model library.
2. Import the validate\_model function and call it with a Pythia model and text prompt.
3. Observe printed results showing stored activation shapes and consistency checks.

Example Usage

```python
validate_model(pythia_model, "Hello world")
```

This project strengthened my understanding of how transformer-based models like Pythia operate internally, and provided valuable experience in model interpretability, debugging, and activation flow analysis.

---

Do you want me to also **add a sample expected output log** (the print statements from your code, e.g., shapes of tensors for each hook) to make the README look even more convincing and practical?
