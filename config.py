from dataclasses import dataclass

import torch


@dataclass
class GPTConfig:
    batch_size = 32 * 8
    block_size = 8
    max_iters = 10000 * 10
    eval_interval = 5000
    learning_rate = 1e-3
    eval_iters = 500
    n_embed = 32
    n_head = 4
    n_layers = 6
    dropout = 0.4
    vocab_size = 0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
