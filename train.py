import pickle

import torch
from tqdm import tqdm

from config import GPTConfig
from model import BigramLanguageModel


def main():
    config = GPTConfig()
    torch.manual_seed(1337)

    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    config.vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    with open("meta.pkl", "wb") as f:
        pickle.dump({"stoi": stoi, "itos": itos}, f)

    encode = lambda s: [stoi[c] for c in s]

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.8 * len(data))
    train_set = data[:n]
    val_set = data[n:]

    def get_batch(split):
        d = train_set if split == "train" else val_set
        offsets = torch.randint(0, len(d) - config.block_size, (config.batch_size,))
        x = torch.stack([d[i : i + config.block_size] for i in offsets])
        y = torch.stack([d[i + 1 : i + config.block_size + 1] for i in offsets])
        # Move inputs to device
        return x.to(config.device), y.to(config.device)

    model = BigramLanguageModel(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(config.eval_iters)
            for k in range(config.eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    print(f"Training on {config.device}...")
    for iter in tqdm(range(config.max_iters)):
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            losses = estimate_loss()
            print(f"\nStep {iter}: Train loss: {losses['train']:.4f}, Val loss: {losses['val']:.4f}")

        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "model.pth")
    print("Training complete. Model weights saved to 'model.pth'.")


if __name__ == "__main__":
    main()
