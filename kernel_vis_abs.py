import os
import math
import time
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt

from transformers import (
    AutoModel,
    AutoFeatureExtractor,
    AutoImageProcessor,
)

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
# === COPY-PASTE: Case B spatial field (absolute PE) for HF ViT/DeiT/DINO/IJEPA-style models ===
# Computes K_i(Δ) = p_i^T (WQ^T WK) p_{i+Δ} and saves a PNG.
# Works when model has additive absolute pos_embed + ViT-style attn.qkv Linear(C->3C).
# Skips Swin/BEiT(relative bias) and RoPE-only models.

import os
import torch
import matplotlib.pyplot as plt

def _get_attr_by_path(obj, path: str):
    cur = obj
    for p in path.split("."):
        cur = getattr(cur, p)
    return cur

def _get_abs_pos_embed_patches(model):
    # Try common HF/timm attribute paths
    paths = [
        "embeddings.position_embeddings",
        "vision_model.embeddings.position_embedding",
    ]
    P = None
    for p in paths:
        try:
            P = _get_attr_by_path(model, p)
            break
        except Exception:
            pass
    if P is None:
        raise ValueError("No absolute pos_embed found (likely relative-bias or RoPE-only model).")

    if isinstance(P, nn.Embedding):
        P = P.weight
    if isinstance(P, torch.nn.Parameter):
        P = P.data
    # [1, T, C] or [T, C]
    if P.dim() == 3:
        P = P[0]
    if P.dim() != 2:
        raise ValueError(f"Unexpected pos_embed shape: {tuple(P.shape)}")

    # Drop CLS if present (heuristic)
    T, C = P.shape
    sT = int(T ** 0.5)
    sT1 = int((T - 1) ** 0.5) if T > 1 else -1
    if sT * sT != T and (T - 1) > 0 and sT1 * sT1 == (T - 1):
        P = P[1:]  # drop CLS
    return P  # [N, C] patches only

def _find_all_vit_qkv_linears(model, names=["attention.query", "attention.key", "attention.value"]):
    print(model)
    qkvs = []
    for name, m in model.named_modules():
        for _name in names:
            if name.endswith(_name):
                if hasattr(m, "weight") and m.weight is not None and m.weight.dim() == 2:
                    out_dim, in_dim = m.weight.shape
                    if out_dim == 3 * in_dim:
                        qkvs.append((name, m))
    return qkvs

@torch.no_grad()
def compute_spatial_field_abs_pe(model, device, grid=None, ref_xy=None, use_full_M=True, head_idx=0):
    """
    Returns:
      K_offset: [H, W] offset-centered field (Δy,Δx) on CPU
      (H,W), ref_xy
    """
    model = model.to(device).eval()

    # 1) P: [N, C]
    P = _get_abs_pos_embed_patches(model).to(device)
    N, C = P.shape

    # 2) infer grid
    if grid is None:
        S = int(N ** 0.5)
        if S * S != N:
            raise ValueError(f"Cannot infer square grid from N={N}. Pass grid=(H,W).")
        H = W = S
    else:
        H, W = grid
        if H * W != N:
            raise ValueError(f"grid {grid} mismatches N={N} patches")

    # 3) reference location
    if ref_xy is None:
        ref_xy = (H // 2, W // 2)
    ry, rx = ref_xy
    ref_i = ry * W + rx

    # 4) get qkv, take the last one
    Wqs, Wks, Wvs = [], [], []
    for name, m in model.named_modules():
        if name.endswith("attention.query") or ("vision_model" in name and name.endswith("self_attn.q_proj")):
            Wqs.append(m.weight.to(device).detach())
        elif name.endswith("attention.key") or ("vision_model" in name and name.endswith("self_attn.k_proj")):
            Wks.append(m.weight.to(device).detach())
        elif name.endswith("attention.value") or ("vision_model" in name and name.endswith("self_attn.v_proj")):
            Wvs.append(m.weight.to(device).detach())

    assert len(Wqs) == len(Wks) == len(Wvs) and len(Wqs) > 0, model

    results = []
    for i in range(len(Wqs)):
        Wq, Wk, Wv = Wqs[i], Wks[i], Wvs[i]

        # 5) build M
        M = Wq.T @ Wk  # [C, C] (clean + robust, no need num_heads)

        # 6) K in absolute coords then roll to offset coords
        p_ref = P[ref_i]          # [C]
        v = p_ref @ M             # [C]
        K_abs = (P @ v).view(H, W)  # [H, W]
        K_offset = torch.roll(torch.roll(K_abs, shifts=-ry, dims=0), shifts=-rx, dims=1)
        
        results.append((K_offset.detach().float().cpu(), (H, W), ref_xy))

    return results


def save_field_png(K_offset, out_png, title="Spatial field (offset-centered)"):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.imshow(K_offset.numpy())
    plt.title(title)
    plt.xlabel("Δx (wrapped)")
    plt.ylabel("Δy (wrapped)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

"""
NOTES / EXPECTED FIXES (based on HF model wrappers):
- Some models may not expose last attention module paths exactly as assumed.
  If capture["x"] is None, print spec.name and inspect model.named_modules().
- For Swin/SwinV2: attention is windowed; this code will capture window attention input,
  but N won't be grid^2. I recommend starting with ViT-like models first.
- For SAM / SigLIP2 / MetaCLIP: config fields or module names may differ; the q/k finder
  heuristic often still works, but you may need to special-case W_Q/W_K paths.

Once you have the direct-logits kernel K (N,N), you can compare it against your
weight-implied kernels and show that the empirical logit bias matches the predicted
positional kernel structure.
"""


def build_processor_and_model(model_id: str, device: torch.device):
    """
    Returns (processor, model). Processor can be AutoImageProcessor or AutoFeatureExtractor.
    Model is AutoModel.from_pretrained(model_id). We set model to eval and to device.
    """
    # Try AutoImageProcessor (newer HF), fallback to AutoFeatureExtractor
    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
    except Exception:
        processor = AutoFeatureExtractor.from_pretrained(model_id)

    model = AutoModel.from_pretrained(model_id)
    model = model.to(device).eval()
    return processor, model


def build_imagenette_loader(imagenette_root: str, batch_size: int = 32, image_size: int = 256, workers: int = 4):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.Imagenette(root=imagenette_root, split="train", transform=transform)
    train_loder = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    dataset = datasets.Imagenette(root=imagenette_root, split="val", transform=transform)
    test_loder = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    return train_loder, test_loder


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    print("Using device:", device)
    # build dataloader
    train_loder, test_loder = build_imagenette_loader(args.imagenette_root, batch_size=args.batch_size, image_size=args.image_size, workers=args.workers)

    results = []
    for model_id in args.models:
        print("Loading model:", model_id)

        processor, model = build_processor_and_model(model_id, device)

        image_size = args.image_size
        if model_id in [
            "google/siglip2-so400m-patch14-224",
            "openai/clip-vit-large-patch14", "DeepGlint-AI/mlcd-vit-bigG-patch14-224",
        ]:
            feature_grid = 16
            patch_size = 14
        elif model_id in ["facebook/dinov2-large"]:
            feature_grid = 37
            patch_size = 14
        else:
            feature_grid = args.feature_grid
            patch_size = args.patch_size

        if args.load_weights == "best":
            decoer_weights_path = os.path.join(args.save_dir, f"best_decoder_{model_id.replace('/', '_')}.pth")
        elif args.load_weights == "epoch":
            raise Exception("Not implemented")
        else:
            decoer_weights_path = None

        res_list = compute_spatial_field_abs_pe(
            model, device,
            # grid=(feature_grid, feature_grid),
            use_full_M=True,   # recommended: robust across HF models
        )

        for j, (K, (H,W), ref) in enumerate(res_list):
            out_png = os.path.join(args.kernel_save_dir, f"field_{model_id.replace('/','_')}_layer_{j}.png")
            save_field_png(K, out_png, title=f"{model_id} field ref={ref}")
            print("Saved:", out_png)

        # results.append({
        #     "model_id": model_id,
        #     **spectral_localization_coefficient(K)
        # })
        # print(f"{model_id} kernel → {summarize_spectral_localization_coefficient(results[-1])}")

    # Print summary
    print(f"\n=== Summary ===")
    for r in results:
        print(f"{r['model_id']}: {summarize_spectral_localization_coefficient(r)}")


if __name__ == "__main__":
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenette_root", type=str, default="./data", help="Path to imagenette train/ (ImageFolder structure)")
    parser.add_argument("--models", type=str, nargs="+",
                        default=[
                            # "facebook/dinov2-large",
                            # "facebook/ijepa_vitg16_22k",
                            # "facebook/dino-vitb16",
                            # "google/vit-base-patch16-224",
                            # "openai/clip-vit-large-patch14",
                            # "facebook/metaclip-b16-fullcc2.5b",
                            # "google/siglip2-so400m-patch14-224",

                            # "facebook/dinov3-vit7b16-pretrain-lvd1689m",
                            # "DeepGlint-AI/mlcd-vit-bigG-patch14-224",
                            # "facebook/deit-base-patch16-224",
                            # "microsoft/beit-base-patch16-224-pt22k",
                            # "facebook/data2vec-vision-large",
                            # "microsoft/swin-base-patch4-window7-224",
                            # "microsoft/swinv2-base-patch4-window16-256",
                            # "facebook/sam-vit-base",
                        ],
                        help="Hugging Face model ids to probe.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--feature_grid", type=int, default=14)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--load_weights", type=str, default=None, help="best | epoch num")
    parser.add_argument("--baseline_eval", action="store_true", help="Whether to evaluate the baseline model.")

    parser.add_argument("--kernel_layers", type=str, default="last",
                        help='Which layers to use for kernel: "last" | "all" | "indices:0,3,6"')
    parser.add_argument("--kernel_png", action="store_true", help="Also save kernel heatmap PNG if matplotlib is available.")


    parser.add_argument("--kernel_batches", type=int, default=8)
    parser.add_argument("--kernel_random_images", action="store_true")
    parser.add_argument("--kernel_save_dir", type=str, default="./kernels_direct")


    args = parser.parse_args()

    for offset in [1]:
        main(args)
        torch.cuda.empty_cache()
        time.sleep(10)
            
