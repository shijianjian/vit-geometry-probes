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


def _find_relative_bias_modules(model, module_name="relative_position_bias_table"):
    hits = []
    for name, m in model.named_modules():
        if hasattr(m, module_name):
            tab = getattr(m, module_name)
            if isinstance(tab, torch.nn.Parameter) or torch.is_tensor(tab):
                hits.append((name, m))
    return hits


def _find_continuous_modules(model, module_name="continuous_position_bias_mlp"):
    hits = []
    for name, m in model.named_modules():
        if hasattr(m, module_name):
            hits.append((name, m))
    return hits


def _guess_window_size_from_table(table_len: int):
    """
    If table_len == (2Wh-1)*(2Ww-1) and Wh==Ww, guess Wh.
    Returns (Wh,Ww) or None.
    """
    # try square window
    s = int(round(table_len ** 0.5))
    if s * s == table_len:
        # s = (2W-1) => W = (s+1)/2
        if (s + 1) % 2 == 0:
            W = (s + 1) // 2
            return (W, W)
    return None

@torch.no_grad()
def extract_relative_bias_field(model, device="cuda", average_heads=True):
    """
    Returns:
      field: [Hf, Wf] (offset grid) where Hf=2Wh-1, Wf=2Ww-1, averaged across heads if requested
      meta: dict with module name, num_heads, window_size guess
    """
    model = model.to(device).eval()

    _module_name = "relative_position_bias_table"
    hits = _find_relative_bias_modules(model)
    if len(hits) == 0:
        hits = _find_relative_bias_modules(model, module_name="relative_coords_table")
        _module_name = "relative_coords_table"
    if not hits:
        raise ValueError("No module with relative_position_bias_table found.")

    res = []
    for i in range(len(hits)):
        name, attn = hits[i]

        tab = getattr(attn, _module_name)
        if isinstance(tab, torch.nn.Parameter):
            tab = tab.data
        tab = tab.to(device)

        if _module_name == "relative_position_bias_table":
            # Common shapes:
            # Swin: [ (2Wh-1)*(2Ww-1), num_heads ]
            # Some variants: [ num_heads, (2Wh-1)*(2Ww-1) ]
            if tab.dim() != 2:
                raise ValueError(f"Unexpected bias table shape: {tuple(tab.shape)}")

            # normalize shape to [L, num_heads]
            if tab.shape[0] < tab.shape[1]:
                # likely [num_heads, L]
                tab = tab.transpose(0, 1)

            L, num_heads = tab.shape

            # Try to get window_size from module (Swin often has window_size)
            window_size = getattr(attn, "window_size", None)
            if window_size is not None:
                # could be (Wh,Ww) or list/tuple
                if isinstance(window_size, (list, tuple)) and len(window_size) == 2:
                    Wh, Ww = int(window_size[0]), int(window_size[1])
                else:
                    # sometimes scalar
                    Wh = Ww = int(window_size)
                expected_L = (2 * Wh - 1) * (2 * Ww - 1) + 3  # BEIT
                if expected_L != L:
                    # fall back to guessing
                    ws_guess = _guess_window_size_from_table(L)
                    if ws_guess is None:
                        raise ValueError(f"window_size={window_size} inconsistent with table_len={L}")
                    Wh, Ww = ws_guess
            else:
                ws_guess = _guess_window_size_from_table(L)
                if ws_guess is None:
                    raise ValueError(
                        f"Cannot infer window size from table_len={L}. "
                        f"If this is BEiT-like with CLS distances, use the 'index-based' method (see note below)."
                    )
                Wh, Ww = ws_guess

            # Reshape to spatial offset grid: [2Wh-1, 2Ww-1, num_heads]
            try:
                field = tab[:-3].view(2 * Wh - 1, 2 * Ww - 1, num_heads)
            except Exception as e:
                field = tab.view(2 * Wh - 1, 2 * Ww - 1, num_heads)

            meta = {
                "module": name,
                "num_heads": int(num_heads),
                "window_size": (int(Wh), int(Ww)),
                "field_shape": tuple(field.shape),
            }
        else:
            field = tab[0]
            meta = {"module": name}

        # average heads -> [Hf,Wf]
        if average_heads:
            field = field.mean(dim=-1)
        res.append((field.detach().float().cpu(), meta))
    return res

def save_field_png(field, out_png, title):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.imshow(field.numpy())
    plt.title(title)
    plt.xlabel("Δx")
    plt.ylabel("Δy")
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

        res_list = extract_relative_bias_field(
            model, device,
        )

        for j, (K, meta) in enumerate(res_list):
            out_png = os.path.join(args.kernel_save_dir, f"field_{model_id.replace('/','_')}_layer_{j}.png")
            save_field_png(K, out_png, title=f"{model_id}")
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
                            "microsoft/swinv2-base-patch4-window16-256",
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
    parser.add_argument("--kernel_save_dir", type=str, default="./kernels_direct_rel")


    args = parser.parse_args()

    for offset in [1]:
        main(args)
        torch.cuda.empty_cache()
        time.sleep(10)
            
