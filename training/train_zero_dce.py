"""
Training script for Zero-DCE (Deep Curve Estimation) on non-reference low-light images.
Optimizes DCE_Net using non-reference loss functions:
- L_spatial (Spatial consistency)
- L_exposure (Well-exposed luminance target)
- L_color (Grey-world chromatic preservation)
- L_tv (Illumination smoothness)
"""

import os
import time
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.zero_dce.model import DCE_Net, L_spatial, L_exposure, L_color, L_tv
from datasets.dataset_setup import LOLDataset, generate_synthetic_samples

def train_zero_dce(data_dir, output_dir, epochs=50, batch_size=4, lr=1e-4, device='cuda' if torch.cuda.is_available() else 'cpu'):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)
    print(f"[*] Training Zero-DCE on device: {device}")

    if not os.path.exists(os.path.join(data_dir, 'our485', 'low')):
        generate_synthetic_samples(os.path.join(data_dir, 'our485'), num_samples=16)

    train_dataset = LOLDataset(data_dir, split='our485', patch_size=128)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = DCE_Net().to(device)
    loss_spa = L_spatial().to(device)
    loss_exp = L_exposure().to(device)
    loss_col = L_color().to(device)
    loss_tv = L_tv().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        total_epoch_loss = 0.0
        start = time.time()

        for batch in train_loader:
            low = batch['low'].to(device)
            optimizer.zero_grad()

            enh, A = model(low)
            l_spa = loss_spa(low, enh)
            l_exp = 10.0 * loss_exp(enh)
            l_col = 5.0 * loss_col(enh)
            l_tv = 200.0 * loss_tv(A)

            loss = l_spa + l_exp + l_col + l_tv
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()

        elapsed = time.time() - start
        avg_loss = total_epoch_loss / max(len(train_loader), 1)
        print(f"Epoch [{epoch:03d}/{epochs:03d}] Zero-DCE Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

    torch.save({'state_dict': model.state_dict()}, os.path.join(output_dir, 'zero_dce_model.pth'))
    print(f"[+] Zero-DCE training complete. Saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets/lol_data')
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    train_zero_dce(args.data_dir, args.output_dir, epochs=args.epochs)
