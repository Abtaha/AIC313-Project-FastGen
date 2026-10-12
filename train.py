import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import PokemonDataset  # Adjust based on TA's dataset.py name/class
from model import ModelFewNFE  # We'll train one model and save weights for both


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset - Ensure this points to the TA's train split
    dataset = PokemonDataset(split='train')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)

    # Initialize our Flow matching model
    model = ModelFewNFE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    epochs = 400  # Flow matching usually needs a few hundred epochs

    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            # Handle if dataset yields (image, label) instead of just image
            x_1 = batch[0] if isinstance(batch, (tuple, list)) else batch
            x_1 = x_1.to(device)

            # 1. Sample Noise (x_0) and Time (t)
            batch_size = x_1.shape[0]
            x_0 = torch.randn_like(x_1)
            t = torch.rand(batch_size, device=device)
            t_expand = t.view(-1, 1, 1, 1)

            # 2. Linear Interpolation for Rectified Flow
            x_t = (1 - t_expand) * x_0 + t_expand * x_1

            # 3. The target is the velocity pointing from x_0 to x_1
            v_target = x_1 - x_0
            v_pred = model.net(x_t, t)

            # 4. Update
            loss = F.mse_loss(v_pred, v_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(dataloader):.4f}")

        # To Do: only save every 10 epochs and show fid

        # Save checkpoints frequently
        if (epoch + 1) % 50 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            # Both models share the exact same weights initially
            torch.save(model.state_dict(), "checkpoints/few_nfe.ckpt")
            torch.save(model.state_dict(), "checkpoints/one_nfe.ckpt")
            print("Checkpoints saved!")


if __name__ == "__main__":
    train()