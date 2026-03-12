<p align="center">
  <img src="Virgil/images/logo_app.png" alt="Virgil logo" width="180"/>
</p>

<h1 align="center">Virgil</h1>
<p align="center">
  <strong>Your Language Model Explainability Navigator</strong>
</p>

<p align="center">
  Discover, compare, and run explainability tools for transformer-based language models.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/Framework-Streamlit-ff4b4b" alt="Streamlit">
  <img src="https://img.shields.io/badge/Platform-Windows-informational" alt="Windows">
  <img src="https://img.shields.io/badge/Status-Active-success" alt="Status">
  <img src="https://img.shields.io/badge/XAI-LLMs-6f42c1" alt="XAI for LLMs">
</p>

---

## Overview

**Virgil** is an interactive toolkit for **explainable AI (XAI) for large language models**.

It is designed to help practitioners and researchers:
- discover suitable explainability methods for their task,
- compare tools side by side,
- and run selected methods through a unified interface.

Virgil combines **tool recommendation** with **runnable interpretability plugins**, making it easier to navigate a fragmented ecosystem of XAI methods for transformer models.

---

## Big Picture

Virgil addresses a simple problem:

> There are many explainability tools for language models, but it is often unclear  
> which method fits a specific need, what level of expertise it requires,  
> and how to actually run it in practice.

Virgil provides:
- a guided interface to **filter and rank methods**,
- a consistent UI to **run explanations**,
- support for both **practitioner-friendly** and **mechanistic interpretability** workflows,
- and comparison views to inspect multiple methods together.

<p align="center">
  <img src="images/virgil_screenshot.png" alt="Virgil screenshot" width="1000"/>
</p>

> Replace the screenshot above with a real app screenshot once ready.

---

## Features

### Tool discovery
- Filter methods by task, architecture, model access, explanation scope, and expertise level
- Rank tools based on user constraints and preferences
- Compare candidate methods side by side

### Runnable explainability plugins
Virgil currently supports plugins for:

- **Captum attribution methods**
  - Integrated Gradients
  - Saliency
  - DeepLift
  - Input × Gradient
  - GradientShap
  - Occlusion
  - Feature Ablation
  - NoiseTunnel variants
  - LIME
  - KernelSHAP
  - Shapley Value Sampling
  - Layer Integrated Gradients

- **Mechanistic interpretability**
  - Logit Lens
  - Direct Logit Attribution
  - Sparse Autoencoder feature exploration
  - Meta-transparency graph
  - Attention Rollout

- **Attention and visualization**
  - BertViz
  - PCA of embeddings / hidden states
  - Linear CKA across layers
  - CCA across layers

- **Example-based and training-data explanations**
  - TracIn
  - Gradient Similarity

- **Black-box explanations**
  - Anchors (Alibi)
  - Counterfactual explanations with Polyjuice

- **Generation-focused attribution**
  - Inseq methods for decoder and encoder-decoder models
  - Integrated Gradients
  - GradientSHAP
  - DeepLIFT
  - Input × Gradient
  - LIME
  - Discretized Integrated Gradients

- **Other interpretability utilities**
  - Ecco NMF
  - Ecco token ranking comparison
  - Probing on binary examples

---

## Libraries and projects used

Virgil builds on top of several excellent open-source projects in interpretability, visualization, and app development. :contentReference[oaicite:1]{index=1}

- [Streamlit](https://docs.streamlit.io/) — interactive app framework
- [Captum](https://captum.ai/) — model interpretability for PyTorch
- [BertViz](https://github.com/jessevig/bertviz) — attention visualization
- [Alibi](https://alibi.readthedocs.io/en/latest/) — black-box and white-box explanation methods
- [Inseq](https://inseq.org/) — interpretability for sequence generation models
- [SAELens](https://github.com/decoderesearch/SAELens) — sparse autoencoder analysis
- [Neuronpedia](https://github.com/hijohnnylin/neuronpedia) — feature dashboards and interpretability tooling
- [Ecco](https://github.com/jalammar/ecco) — interactive NLP model explanation and visualization
- [Polyjuice](https://github.com/tongshuangwu/polyjuice) — counterfactual text generation
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) — mechanistic interpretability tooling
- [PyTorch](https://pytorch.org/) — deep learning framework
- [Pandas](https://pandas.pydata.org/) — data handling
- [Matplotlib](https://matplotlib.org/) — plotting
- [Plotly](https://plotly.com/python/) — interactive plots
- [PyVis](https://pyvis.readthedocs.io/) — graph visualization

---

## Installation

> Installation instructions will be finalized soon.

### Option 1 — standard environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_new.txt