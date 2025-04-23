# GPT-2 Pytorch Implementation
---
## Overview

A PyTorch-based reproduction of OpenAI's GPT-2 model, inspired by the original research paper and popular tutorials by Andrej Karpathy, Vukrosic & Tunadorable.

This repository is a massive overhaul of Modded-NanoGPT, NanoGPT, TinyLlama, Meta's Lingua, and GPT Lab, designed to be an accessible, from-scratch base for amateurs to conduct cost-effective and rapid LLM research at a scale sufficient for ArXiv-worthy experiments. Existing lightweight GPT implementations often suffer from outdated architectures, narrow use cases, hidden dependencies, high computational costs, or complexity that hinders quick iteration—this project strikes a unique balance, offering a clean, modular, and affordable platform without sacrificing performance or educational clarity.

Under the hood, this project faithfully reproduce OpenAI's GPT-2 transformer architecture in PyTorch, paired with a custom BPE tokenizer, end-to-end training scripts, and inference tools. The goal is to enable anyone with consumer-grade hardware or low-cost cloud GPUs to experiment with large language models and contribute novel ideas to the field.

## Table of Contents
- [References](#references)
- [Installation](#installation)
- [Usage](#usage)
  - [Tokenizer Training](#tokenizer-training)
  - [Data Preparation](#data-preparation)
  - [Model Training](#model-training)
  - [Text Generation](#text-generation)
- [Contributing](#contributing)


## References

- **GPT-2: Language Models are Unsupervised Multitask Learners** (Radford et al., 2019)  
  [Read the paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- **Andrej Karpathy’s Tutorials**  
  - Reproducing GPT-2 (124M): https://youtu.be/l8pRSuU81PU?si=5FuXWc6VJghGZBr0  
  - Building the GPT Tokenizer: https://youtu.be/zduSFxRajkE?si=FtokVhpIZZPZSzOR
- **GPT Lab** by Evintunador: https://github.com/evintunador/gpt-lab
- **Tunadorable**
  - Talking thru GPT-Lab w/ @vukrosic : https://www.youtube.com/live/WDxdXn3tsNw?si=0VXl624Qjqshu3kY
  - Train your own GPT experiments easy & cheap : https://youtu.be/4cvBgHMDISs?si=CFcfR72ahrS5CRFD    
  
## Installation

1. **Fork or Template**  
   Fork this repository or use it as a GitHub template to start your own project.

2. **Clone locally**
   ```bash
   git clone https://github.com/prathamesh-mandavkar/gpt2-implementation.git
   cd gpt2-implementation
   ```

3. **Virtual environment** (recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Tokenizer Training

Train a BPE tokenizer on the FineWeb corpus. Adjust `--samples` (total characters, split evenly across GPUs) and `--vocabsize` (excluding special tokens).

- **Single GPU**
  ```bash
  python tokenizer/train_tokenizer.py \
    --samples 100000 \
    --vocabsize 1000 \
    --name readmetokenizer \
    --demo
  ```
- **Multi-GPU** (replace `G` with GPU count)
  ```bash
  torchrun --nproc_per_node=G tokenizer/train_tokenizer.py \
    --samples 100000 \
    --vocabsize 1000 \
    --name readmetokenizer \
    --demo
  ```

_For a detailed walkthrough of BPE tokenizers on CPU, see Andrej Karpathy’s video._

### Data Preparation

1. **Download FineWeb dataset** (versions: `10B`, `100B`, `10Bedu`, `100Bedu`)
   ```bash
   python scripts/download_fineweb.py \
     --version 10B \
     --num_shards 1 \
     --shard_size 10000000 \
     --tokenizer readmetokenizer_v1000_n100000.pkl
   ```
   - `shard_size`: number of tokens per shard
   - `num_shards`: total training shards (validation uses 1 shard by default)

2. **Download benchmarks**
   ```bash
   python scripts/download_hellaswag.py
   ```

### Model Training

Train GPT-2 on prepared shards. Vocabulary size = tokenizer size + special tokens (e.g., `1000 + 1`).

- **Single GPU**
  ```bash
  python train_gpt.py \
    --model_name ReadmeGPT \
    --tokenizer readmetokenizer_v1000_n100000.pkl \
    --vocab_size 1001 \
    --model_dim 128 \
    --num_heads 4 \
    --num_layers 6
  ```
- **Multi-GPU** (replace `G` with GPU count)
  ```bash
  torchrun --nproc_per_node=G train_gpt.py \
    --model_name ReadmeGPT \
    --tokenizer readmetokenizer_v1000_n100000.pkl \
    --vocab_size 1001 \
    --model_dim 128 \
    --num_heads 4 \
    --num_layers 6
  ```

> **Warning**: Using `--save_model` will create a `.pt` checkpoint. By default, `.pt` files are ignored by `.gitignore` to prevent large uploads.

### Text Generation

Generate samples from a trained checkpoint:
```bash
python generate.py \
  --checkpoint experiments/ReadmeGPT/<latest>.pt \
  --prompt "Once upon a time" \
  --max_length 100 \
  --temperature 0.9 \
  --top_k 50
```
## Contributing

Contributions welcome! Please open issues or PRs for bugs, features, or performance improvements. Follow PEP8, include tests, and document new functionality.
