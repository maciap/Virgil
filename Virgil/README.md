<p align="center">
  <img src="images/logo_app.png" alt="Virgil logo" width="180"/>
</p>

<h1 align="center">Virgil</h1>
<p align="center">
  <strong>Your Language Model Explainability Navigator</strong>
</p>

<p align="center">
  Discover, compare, and run explainability methods for transformer-based language models.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue" alt="Python">
  <img src="https://img.shields.io/badge/Framework-Streamlit-ff4b4b" alt="Streamlit">
  <img src="https://img.shields.io/badge/Status-Active-success" alt="Status">
</p>



---

## Overview

> There are many explainability tools for language models, but it is often unclear  
> which method fits a specific need, what level of expertise it requires,  
> and how to actually run it in practice.


**Virgil** is an interactive interface for **explainable AI for transformer-based language models**.

It is designed to help practitioners and researchers, regardless of their level of expertise, to:
- discover suitable explainability methods for their needs,
- easily access the salient characteristics of the different methods, 
- compare methods side by side,
- run selected methods through a unified interface.
---


## Try Virgil Online

You can try Virgil directly in your browser via Hugging Face Spaces:

[![Hugging Face Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-yellow)](https://huggingface.co/spaces/XAI4LLMs/Virgil)

---


<p align="center">
  <img src="images/light_mode_screenshot.jpg" alt="Virgil screenshot" width="1000"/>
</p>

---

## Libraries and projects used

Virgil builds on top of several excellent open-source resources. :contentReference[oaicite:1]{index=1}

- [Captum](https://captum.ai/) — explainability for PyTorch
- [BertViz](https://github.com/jessevig/bertviz) — attention visualization
- [Alibi](https://alibi.readthedocs.io/en/latest/) — miscellaneous explanation methods
- [Inseq](https://inseq.org/) — explainability for text generation
- [SAELens](https://github.com/decoderesearch/SAELens) — sparse autoencoder analysis
- [Neuronpedia](https://github.com/hijohnnylin/neuronpedia) — feature dashboards and interpretability tooling
- [Ecco](https://github.com/jalammar/ecco) — explanation methods for transformer-based language models 
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) — mechanistic interpretability for transformer-based language models
- [LLM Transparency Tool] (https://github.com/facebookresearch/llm-transparency-tool) mechanistic interpretability for transformer-based language models
---

## Installation

### Option 1 — standard environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_new.txt
