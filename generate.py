import pickle

import torch

from config import GPTConfig
from model import BigramLanguageModel


def main():
    config = GPTConfig()

    print("loading vocabulary")
    with open("meta.pkl", "rb") as f:
        meta = pickle.load(f)

    itos = meta["itos"]
    config.vocab_size = len(itos)
    decode = lambda indices: "".join([itos[i] for i in indices])

    print(f"loading model to {config.device}")
    model = BigramLanguageModel(config)

    state_dict = torch.load("model.pth", map_location=config.device)
    model.load_state_dict(state_dict)
    model.to(config.device)
    model.eval()

    print("generating text:\n" + "-" * 50)
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)

    with torch.no_grad():
        generated_tokens = model.generate(context, max_new_tokens=500)[0].tolist()

    print(decode(generated_tokens))
    print("-" * 50)


if __name__ == "__main__":
    main()
