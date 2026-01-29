import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, datasets
from feature import FeatureExtractor
import pandas as pd
import matplotlib.pyplot as plt
import os
from utils import compute_fundamental_matrix, sample_epipolar_line, cosine_similarity, draw_matches
from data import SpringDataset, read_dsp5
from metric import evaluate_epipolar_consistency_new


def run_model(extractor, imageL, imageR, disp_gt, valid_mask, upsample=True, reshuffle_pos=False):
    featL = extractor.extract(imageL, reshuffle_pos=reshuffle_pos)
    featR = extractor.extract(imageR, reshuffle_pos=reshuffle_pos)
    # Step 2: Upsample features
    upsampler = torch.hub.load("wimmerth/anyup", "anyup", verbose=False).to(device).eval()
    
    if upsample:
        with torch.no_grad():
            # q_chunk_size can reduce memory if needed
            featL_up = upsampler(imageL.to(device), featL, q_chunk_size=256)  # (1, C, 448, 448)
            featR_up = upsampler(imageR.to(device), featR, q_chunk_size=256)  # (1, C, 448, 448)
    else:
        featL_up = F.interpolate(featL, size=(image_size, image_size), mode='bilinear')
        featR_up = F.interpolate(featR, size=(image_size, image_size), mode='bilinear')

    return featL_up, featR_up, disp_gt[None], valid_mask[None, None]


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def plot_epipolar_response(featL, featR, d_gt, model_name, tau=20, s=+1, max_disp=64):
    """
    Visualize cost volume and feature similarity along an epipolar line.
    featL, featR: [C, H, W] normalized feature maps
    d_gt: ground truth disparity map [H, W]
    """
    H, W = d_gt.shape
    y = H // 2  # center row
    x = W // 2  # center pixel
    gt_d = int(d_gt[y, x].item())

    # build cost curve for one pixel
    disp_range = range(0, max_disp)
    cost_curve = []
    for d in disp_range:
        shifted = torch.roll(featR, shifts=-s*d, dims=-1)
        sim = (featL[:, y, x] * shifted[:, y, x]).sum().item()
        cost_curve.append(sim)

    cost_curve = np.array(cost_curve)
    prob = F.softmax(torch.tensor(tau * cost_curve), dim=0).numpy()

    plt.figure(figsize=(4,3))
    plt.plot(disp_range, cost_curve, label='raw sim', color='gray', lw=2)
    plt.plot(disp_range, prob, label='softmax(τC)', color='blue', lw=2)
    plt.axvline(gt_d, color='red', linestyle='--', label='GT disparity')
    plt.title(f"{model_name} – Epipolar Response")
    plt.xlabel("Disparity (pixels)")
    plt.ylabel("Similarity / Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"epipolar_response_{model_name}.png")

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


@torch.no_grad()
def plot_epipolar_response(
    featL: torch.Tensor,
    featR: torch.Tensor,
    d_gt: torch.Tensor,
    model_name: str,
    tau: float = 20,
    s: int = +1,
    max_disp: int = 64,
    upsample: bool = False,
    if_zero_pos: bool = False,
    output_dir: str = "./epipolar_response_plots"
):
    """
    Visualize feature-space epipolar structure and the effect of upsampling.

    Args:
        featL: [1, C, H, W] left feature map
        featR: [1, C, H, W] right feature map
        d_gt: [1, 1, H, W] ground-truth disparity
        model_name: string for labeling
        tau: temperature for softmax
        s: stereo direction (+1 or -1)
        max_disp: max disparity range to probe
    """

    B, C, H, W = featL.shape
    assert B == 1, "visualization is only for a single sample"

    # ----------------------------------------
    # 1. Normalize features (important!)
    featL = F.normalize(featL, dim=1)
    featR = F.normalize(featR, dim=1)

    # ----------------------------------------
    # 2. Build cost volume in feature space
    cost_vol = []
    for d in range(max_disp + 1):
        if s == +1:
            shifted = torch.roll(featR, shifts=-d, dims=-1)
            shifted[..., :, W - d:] = 0
        else:
            shifted = torch.roll(featR, shifts=d, dims=-1)
            shifted[..., :, :d] = 0
        cost_vol.append((featL * shifted).sum(1, keepdim=True))
    cost_vol = torch.cat(cost_vol, dim=1)  # [1, D, H, W]

    # ----------------------------------------
    # 3. Soft argmin to get expected disparity
    prob = F.softmax(tau * cost_vol, dim=1)
    disp_pred = torch.sum(prob * torch.arange(max_disp + 1, device=featL.device).view(1, -1, 1, 1), dim=1)

    # ----------------------------------------
    # 4. PCA projection for visualization
    def pca_project(feat):
        feat_flat = feat[0].permute(1, 2, 0).reshape(-1, C)
        mean = feat_flat.mean(dim=0, keepdim=True)
        X = feat_flat - mean
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        pcs = Vh[:3].T
        proj = X @ pcs
        proj = proj.reshape(H, W, 3)
        proj = (proj - proj.min(0).values) / (proj.max(0).values - proj.min(0).values + 1e-6)
        return proj.cpu().numpy()

    lr_rgb = pca_project(featL)
    hr_rgb = pca_project(featR)

    # ----------------------------------------
    # 5. Visualizations
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(lr_rgb)
    axs[0].set_title(f"{model_name}: Left feature space")
    axs[0].axis("off")

    axs[1].imshow(hr_rgb)
    axs[1].set_title(f"Right / AnyUp features")
    axs[1].axis("off")

    axs[2].imshow(disp_pred[0].cpu(), cmap="magma")
    axs[2].set_title("Feature-space disparity (soft-argmin)")
    axs[2].axis("off")

    plt.suptitle(f"Epipolar Feature Response — {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"epipolar_response_{'no_pos' if if_zero_pos else 'pos'}_{'anyup' if upsample else 'bilinear'}_pca_{model_name}.png"), bbox_inches='tight')

    # Optionally overlay cost-volume response along one scanline
    # mid_y = H // 2
    # plt.figure(figsize=(8, 4))
    # plt.imshow(prob[0, :, mid_y, :].cpu(), aspect="auto", cmap="viridis")
    # plt.title(f"Epipolar response along scanline y={mid_y}")
    # plt.xlabel("x (pixel)")
    # plt.ylabel("disparity index d")
    # plt.colorbar(label="softmax prob")
    # plt.savefig(f"epipolar_response_{'anyup' if upsample else 'bilinear'}_{model_name}.png", bbox_inches='tight')

    return disp_pred



device = "cuda" if torch.cuda.is_available() else "cpu"
image_size = 448
reshuffle_pos = False
output_dir = f"./epipolar_response_plots_{'reshuffle' if reshuffle_pos else 'no_reshuffle'}"
if_zero_pos = False
upsample = True
pos_shuffle = True

dataset = SpringDataset(
    "/ibex/ai/home/shij0c/git/stereo_dataset/OpenStereo/data/Spring",
    "/ibex/ai/home/shij0c/git/stereo_dataset/feature_analysis/epipolar/spring_train.txt",
    image_size,
)

for backbone in [
    "facebook/dinov2-large",
    "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    "facebook/ijepa_vitg16_22k",
    "facebook/deit-base-patch16-224",
    "google/siglip2-so400m-patch14-224",

    "facebook/dino-vitb16",
    "microsoft/beit-base-patch16-224-pt22k",
    "facebook/data2vec-vision-large",
    "DeepGlint-AI/mlcd-vit-bigG-patch14-224",
    "facebook/sam-vit-base",
    "openai/clip-vit-large-patch14",

    "microsoft/swin-base-patch4-window7-224",
    "microsoft/swinv2-base-patch4-window16-256",
    "google/vit-base-patch16-224",
]:
    results = None
    extractor = FeatureExtractor(
        backbone, if_zero_pos=if_zero_pos, pos_shuffle=pos_shuffle).to(device)
    for i, data in enumerate(tqdm(dataset)):
        if i < 40:
            continue
        imageL, imageR, disp_gt = data["left"][None], data["right"][None], data["disp"]
        import torchvision
        torchvision.utils.save_image(imageL, "temp_L.png")
        torchvision.utils.save_image(imageR, "temp_R.png")
        valid_mask = torch.as_tensor(data["valid"]) * (1 - torch.as_tensor(data["occ_mask"], dtype=torch.long))
        valid_mask = valid_mask[0]
        featL, featR, disp_gt, valid_mask = run_model(
            extractor, imageL, imageR, disp_gt, valid_mask, upsample=upsample, reshuffle_pos=reshuffle_pos)
        os.makedirs(output_dir, exist_ok=True)
        plot_epipolar_response(
            featL, featR, disp_gt[0, 0], backbone.split("/")[1], max_disp=192, upsample=upsample, if_zero_pos=if_zero_pos,
            output_dir=output_dir
        )
        break
