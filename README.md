# TAG-PF
**Tokenized Autoregressive Generation With Large Language Models for Renewable Power Forecasting**  
📄 *Status: Under review (submitted)*

TAG-PF is a two-stage generative forecasting framework that reformulates renewable power forecasting as **conditional next-token autoregressive generation** over a unified vocabulary. It first learns a **vector-quantization (VQ) temporal tokenizer** that maps multivariate power–meteorology sequences into discrete operational tokens, and then adapts a **pre-trained LLM** to generate future tokens autoregressively under token-type/length constraints, followed by de-tokenization to recover continuous forecasts. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/TAG-PF.git
cd TAG-PF
