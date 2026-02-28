# TAG-PF
**Tokenized Autoregressive Generation With Large Language Models for Renewable Power Forecasting**  
📄 *Status: Under review (submitted)*

TAG-PF is a two-stage generative forecasting framework that reformulates renewable power forecasting as **conditional next-token autoregressive generation** over a unified vocabulary. It first learns a **vector-quantization (VQ) temporal tokenizer** that maps multivariate power–meteorology sequences into discrete operational tokens, and then adapts a **pre-trained LLM** to generate future tokens autoregressively under token-type/length constraints, followed by de-tokenization to recover continuous forecasts.

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone [https://github.com/Code111/TAG-PF.git]
cd TAG-PF
```

### 2. Environment Setup
```bash
conda create -n tagpf python=3.12
conda activate tagpf
pip install -r requirements.txt
```

### 3. Data Preparation

#### Dataset Link

The dataset / feature-variable analysis repository you provided is:

https://github.com/Bob05757/Renewable-energy-generation-input-feature-variables-analysis
```
### 4. Train the Time-Series Tokenizer (Stage 1)

In Stage 1, we train a VQ-based time-series tokenizer to map continuous multivariate power–meteorology sequences into discrete tokens.

```bash
cd Stage1
sh scripts/run.sh
```
### 5. Build Tokenized Datasets (Stage 2 → build_tokens)
This step uses the trained tokenizer to convert continuous sequences into discrete token sequences for LLM training and evaluation.
```bash
cd Stage2/build_tokens
sh run.sh
```
### 6. LLM Fine-tuning / Adaptation and Forecasting (Stage 2 → pretrain)

In Stage 2, we **fine-tune (adapt)** a pre-trained LLM (e.g., via parameter-efficient tuning such as LoRA) to perform **conditional autoregressive generation** over the unified token vocabulary. The generated future tokens are then decoded back to continuous renewable power forecasts.

```bash
cd Stage2/pretrain
sh run.sh
```
## Acknowledgements

Our implementation is based on the following open-source codebases, with substantial modifications tailored to our research needs. We thank the authors for sharing their implementations and resources.

- Time-Series-Library: https://github.com/thuml/Time-Series-Library  
- VQGAN (taming-transformers): https://github.com/CompVis/taming-transformers



