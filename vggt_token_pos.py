from typing import Tuple
import sys
sys.path.append("./vggt")

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn.functional as F


def make_position_labels(B: int, Hpos: int, Wpos: int, device: torch.device):
    """
    Produce per-token (row, col) labels for a batch.

    Returns:
      y_row, y_col with shape (B*Hpos*Wpos,)
    """
    rows = torch.arange(Hpos, device=device)
    cols = torch.arange(Wpos, device=device)
    grid_row = rows.view(Hpos, 1).expand(Hpos, Wpos)  # (Hpos, Wpos)
    grid_col = cols.view(1, Wpos).expand(Hpos, Wpos)  # (Hpos, Wpos)

    # Repeat for batch
    y_row = grid_row.flatten().unsqueeze(0).expand(B, -1).reshape(-1)
    y_col = grid_col.flatten().unsqueeze(0).expand(B, -1).reshape(-1)
    return y_row, y_col

# --------------------
# Model
# --------------------
class PositionPredictor(nn.Module):
    """
    Predict (row, col) position from a token or an NxN neighborhood of tokens.

    Inputs:
      - feats: (B, C, Hf, Wf) feature map from the frozen encoder
      - context_size: N (odd number, typically 1, 3, or 5)

    Internals:
      - Unfolds NxN neighborhoods (if N>1) to produce vectors of length C*N*N
      - Two classification heads: row in [0..Hpos-1], col in [0..Wpos-1]
        where Hpos = Hf - N + 1, Wpos = Wf - N + 1
    """
    def __init__(self, in_dim: int, hidden: int = 256, context_size: int = 1):
        super().__init__()
        self.context_size = context_size
        self.in_dim = in_dim * (context_size ** 2)

        self.backbone = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )

        # Heads will be materialized lazily after we see Hpos/Wpos
        self.row_head = None  # nn.Linear(hidden, Hpos)
        self.col_head = None  # nn.Linear(hidden, Wpos)

    def _maybe_build_heads(self, Hpos: int, Wpos: int, device: torch.device):
        # Create heads only once (or if shape changed across models)
        if (self.row_head is None) or (self.row_head.out_features != Hpos):
            self.row_head = nn.Linear(self.backbone[-2].out_features, Hpos).to(device)
        if (self.col_head is None) or (self.col_head.out_features != Wpos):
            self.col_head = nn.Linear(self.backbone[-2].out_features, Wpos).to(device)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Args:
          feat: (B, C, Hf, Wf)

        Returns:
          logits_row: (B*Hpos*Wpos, Hpos)
          logits_col: (B*Hpos*Wpos, Wpos)
          Hpos, Wpos
        """
        B, C, Hf, Wf = feat.shape
        N = self.context_size

        if N == 1:
            # tokens: (B, Hf, Wf, C) -> (B*Hf*Wf, C)
            tokens = feat.permute(0, 2, 3, 1).contiguous().view(B * Hf * Wf, C)
            Hpos, Wpos = Hf, Wf
        else:
            # unfold neighborhoods: (B, C*N*N, (Hf-N+1)*(Wf-N+1))
            neigh = F.unfold(feat, kernel_size=N, stride=1)  # (B, C*N*N, L)
            Hpos, Wpos = Hf - N + 1, Wf - N + 1
            tokens = neigh.permute(0, 2, 1).contiguous()     # (B, L, C*N*N)
            tokens = tokens.view(B * Hpos * Wpos, C * N * N)

        self._maybe_build_heads(Hpos, Wpos, feat.device)

        h = self.backbone(tokens)
        logits_row = self.row_head(h)  # (B*Hpos*Wpos, Hpos)
        logits_col = self.col_head(h)  # (B*Hpos*Wpos, Wpos)
        return logits_row, logits_col, Hpos, Wpos


def compute_semantic_similarity(feat1, feat2):
    # feat1, feat2: B x C x N
    feat1 = F.normalize(feat1, dim=1)  # normalize along channel dim
    feat2 = F.normalize(feat2, dim=1)
    # compute similarity along channel dim, resulting in B x N
    sim = F.cosine_similarity(feat1, feat2, dim=1)
    # average over batch and spatial dims
    return sim.mean().item()


def build_imagenette_loader(imagenette_root: str, batch_size: int = 32, image_size: int = 256, patch_size: int = 8, workers: int = 4):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size + patch_size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.Imagenette(root=imagenette_root, split="train", transform=transform)
    train_loder = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    dataset = datasets.Imagenette(root=imagenette_root, split="val", transform=transform)
    test_loder = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    return train_loder, test_loder


device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

n_patches = 3  # NOTE: change this for different delta_x

patch_size = 14 * n_patches
train_loder, test_loder = build_imagenette_loader(
    imagenette_root="../data", batch_size=8, image_size=518, patch_size=patch_size, workers=4)

epochs = 5
infer_dino_tokens = False
sims = [[] for _ in range(1 if infer_dino_tokens else 24)]
ce = nn.CrossEntropyLoss()
predictors = [
    PositionPredictor(
        in_dim=1024 if infer_dino_tokens else 2048, hidden=512, context_size=1).to(device) for _ in range(1 if infer_dino_tokens else 24)
]
optimizers = [
    torch.optim.Adam(predictor.parameters(), lr=1e-3) for predictor in predictors
]
schedulers = [
    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loder)) for optimizer in optimizers
]


for epoch in range(epochs):

    for data_loader, is_train in [(train_loder, True), (test_loder, False)]:

        acc_rows = [[] for _ in range(1 if infer_dino_tokens else 24)]
        acc_cols = [[] for _ in range(1 if infer_dino_tokens else 24)]
        acc_boths = [[] for _ in range(1 if infer_dino_tokens else 24)]
        
        total_losses = [0.0 for _ in range(1 if infer_dino_tokens else 24)]
        total_counts = [0 for _ in range(1 if infer_dino_tokens else 24)]
        correct_rows = [0 for _ in range(1 if infer_dino_tokens else 24)]
        correct_cols = [0 for _ in range(1 if infer_dino_tokens else 24)]
        correct_boths = [0 for _ in range(1 if infer_dino_tokens else 24)]
                
        for predictor in predictors:
            predictor.train(is_train)

        pbar = tqdm(data_loader)
        for images, _ in pbar:
            images = images.to(device)
            images = torch.stack([images[..., :-patch_size], images[..., patch_size:]], dim=1)

            with torch.no_grad():
                if infer_dino_tokens:
                    B, S, C_in, H, W = images.shape
                    # Normalize images and reshape for patch embed
                    images = (images - model.aggregator._resnet_mean) / model.aggregator._resnet_std
                    # Reshape to [B*S, C, H, W] for patch embedding
                    images = images.view(B * S, C_in, H, W)
                    patch_tokens = model.aggregator.patch_embed(images)["x_norm_patchtokens"]
                    _, P, C = patch_tokens.shape
                    aggregated_tokens_list = [patch_tokens.reshape(B, S, P, C)]
                    patch_start_idx = 0
                else:
                    aggregated_tokens_list, patch_start_idx = model.aggregator(images)

            for i, a in enumerate(aggregated_tokens_list):
                b, s, n, c = a.shape

                tokens = a[..., patch_start_idx:, :].reshape(-1, n - patch_start_idx, c).permute(0, 2, 1)
                feat_map = tokens.reshape(-1, c, int(n ** (1/2)), int(n ** (1/2)))
                # feat_map = F.interpolate(feat_map, size=(16, 16), mode="bilinear")
                feat_map = F.adaptive_avg_pool2d(feat_map, (16,16))

                B, C, Hf, Wf = feat_map.shape
                
                # Predictor forward (builds heads lazily)
                logits_row, logits_col, Hpos, Wpos = predictors[i](feat_map)
                # Labels
                y_row, y_col = make_position_labels(B, Hpos, Wpos, device)
                loss = ce(logits_row, y_row) + ce(logits_col, y_col)
                # Backprop
                if is_train:
                    optimizers[i].zero_grad()
                    loss.backward()
                    optimizers[i].step()
                    if schedulers[i] is not None:
                        schedulers[i].step()

                # Metrics
                with torch.no_grad():
                    pred_row = logits_row.argmax(dim=1)
                    pred_col = logits_col.argmax(dim=1)
                    both = (pred_row == y_row) & (pred_col == y_col)

                    # bs_tokens = y_row.numel()
                    # total_losses[i] += loss.item() * bs_tokens
                    # total_counts[i] += bs_tokens
                    # correct_rows[i] += (pred_row == y_row).sum().item()
                    # correct_cols[i] += (pred_col == y_col).sum().item()
                    # correct_boths[i] += both.sum().item()

                    # acc_row = correct_rows[i] / total_counts[i]
                    # acc_col = correct_cols[i] / total_counts[i]
                    # acc_both = correct_boths[i] / total_counts[i]
                    acc_rows[i].append((pred_row == y_row).sum().item() / y_row.numel())
                    acc_cols[i].append((pred_col == y_col).sum().item() / y_col.numel())
                    acc_boths[i].append(both.sum().item() / both.numel())

                if i == 0:
                    pbar.set_postfix_str(f"loss={loss.item():.4f} row={acc_rows[i][-1]:.3f} col={acc_cols[i][-1]:.3f} both={acc_boths[i][-1]:.3f}")

        for i in range(len(acc_rows)):
            print(f"Epoch {epoch} Index {i} {'train' if is_train else 'test'} Row {np.array(acc_rows[i]).mean():.3f} Col {np.array(acc_cols[i]).mean():.3f} Both {np.array(acc_boths[i]).mean():.3f}")
