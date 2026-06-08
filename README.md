# nanogpt

<div align="center">
  <p>
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/Tqdm-40B281?style=for-the-badge&logo=tqdm&logoColor=white" alt="tqdm" />
  </p>
</div>

nanogpt is a clean, character-level decoder-only Transformer language model built in PyTorch. Inspired by Andrej Karpathy's educational deep learning series, this repository provides a highly accessible, modular implementation of a GPT-style network from scratch, complete with custom attention heads, residual blocks, and text generation logic.

## Features

- **Decoder-Only Transformer**: Structured architecture utilizing 6 sequential block layers with a residual connection design.
- **Custom Attention Layers**: Implements Multi-Head Self-Attention using masked causal attention weights to enforce autoregressive token prediction.
- **Dynamic Training Pipeline**: Splits data dynamically, optimizes parameters using an AdamW optimizer, and schedules updates to print periodic evaluation metrics across training and validation splits.
- **Configurable Architecture**: Uses a clear `GPTConfig` dataclass configuration mapping dimensions ($N_{\text{embed}} = 32$, $N_{\text{head}} = 4$, dropout $= 0.4$).
- **Generative Decoding**: Features an autoregressive sampling helper (`.generate()`) with temperature-free Multinomial probability sampling to synthesize continuous character sequences.

## Tech Stack

- **Deep Learning Framework:** PyTorch (torch.nn, torch.nn.functional)
- **Serialization:** Pickle (for vocabulary-to-index metadata)
- **Utilities:** Tqdm

## Project Structure

```text
├── config.py               # Hyperparameter configurations and device configurations
├── model.py                # Core modules (Head, MultiHeadAttention, Block, FeedForward, BigramLanguageModel)
├── train.py                # Training loop, dynamic batching, and evaluation logic
├── generate.py             # Inference script loading weights to generate raw text
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites
- Python (v3.10+)
- CUDA-compatible GPU (recommended for acceleration, falls back to CPU automatically)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/abderrahmenex86/nanogpt.git
cd nanogpt
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### How to Use It

1. **Prepare your Training Dataset:**
Place a raw text file named `input.txt` in the root of the repository (e.g., the Tiny Shakespeare dataset or any other text file).

2. **Train the Transformer:**
Run the training script to process characters, serialize mappings into `meta.pkl`, and output checkpoints:
```bash
python train.py
```
This loop runs for 100,000 iterations (by default configuration), displaying progress bars and saving trained weights to `model.pth`.

3. **Generate Text:**
Use the generator to load model parameters and sample 500 characters starting from a blank context:
```bash
python generate.py
```
