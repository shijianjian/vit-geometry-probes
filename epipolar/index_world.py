import torch
import math
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, datasets
import pandas as pd

from utils import compute_fundamental_matrix, sample_epipolar_line, cosine_similarity, draw_matches
from feature import build_processor_and_model, extract_patch_feature_map, FeatureExtractor
from data import SpringDataset, SceneFlowDataset
from metric import evaluate_epipolar_consistency_new
from pe_ops import GlobalPEPlugin

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="spring")
parser.add_argument("--shift_patch_num", type=int, default=0)
parser.add_argument("--upsample", action="store_true", default=False)
parser.add_argument("--image_size", type=int, default=448)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

image_size = args.image_size
upsample = args.upsample
dataset_name = args.dataset
shift_patch_num = (0, args.shift_patch_num)


def run_model(extractor, plugin, imageL, imageR, disp_gt, valid_mask, upsample=True, upsampler=None):
    
    # plugin = GlobalPEPlugin(
    #     extractor.model, backbone, image_size, patch_size=patch_size, shift_patch_num=shift_patch_num, grid_scale_factor=1)

    if shift_patch_num[1] != 0:
        if plugin is None:
            raise RuntimeError("Plugin is not supported for this model")
        plugin.update(0)
    featL = extractor.extract(imageL.to(device))
    if shift_patch_num[1] != 0:
        if plugin is None:
            raise RuntimeError("Plugin is not supported for this model")
        plugin.update(shift_patch_num[1])
    featR = extractor.extract(imageR.to(device))
    # Step 2: Upsample features

    if upsample:
        with torch.no_grad():
            # q_chunk_size can reduce memory if needed
            featL_up = upsampler(imageL.to(device), featL, q_chunk_size=256)  # (1, C, 448, 448)
            featR_up = upsampler(imageR.to(device), featR, q_chunk_size=256)  # (1, C, 448, 448)
    else:
        featL_up = F.interpolate(featL, size=(image_size, image_size), mode='bilinear')
        featR_up = F.interpolate(featR, size=(image_size, image_size), mode='bilinear')

    return evaluate_epipolar_consistency_new(featL_up, featR_up, disp_gt[None], valid_mask[None, None])


if dataset_name == "spring":
    dataset = SpringDataset(
        "/ibex/ai/home/shij0c/git/stereo_dataset/OpenStereo/data/Spring",
        "/ibex/ai/home/shij0c/git/stereo_dataset/feature_analysis/epipolar/spring_train.txt",
        image_size,
    )
else:
    raise ValueError(f"Invalid dataset name: {dataset_name}")

for backbone in [
    # "facebook/dinov2-large",
    "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    # "facebook/ijepa_vitg16_22k",
    # "facebook/deit-base-patch16-224",
    # "google/siglip2-so400m-patch14-224",
    # "openai/clip-vit-large-patch14",
    # "facebook/dino-vitb16",
    # "google/vit-base-patch16-224",

    # "microsoft/beit-base-patch16-224-pt22k",
    # "facebook/data2vec-vision-large",

    # "facebook/vit-mae-base",  # Contains Random Masking, breaks the offset settings.
    # "DeepGlint-AI/mlcd-vit-bigG-patch14-224",
    # "microsoft/swin-base-patch4-window7-224",
    # "microsoft/swinv2-base-patch4-window16-256",
    # "facebook/sam-vit-base",
]:
    print(f"Running {backbone} for {dataset_name} with shift {shift_patch_num[1]} and upsample {upsample}")
    results = None
    if backbone in ["timm/twins_pcpvt_base.in1k"]:
        force_image_size = image_size
    else:
        force_image_size = None
    extractor = FeatureExtractor(backbone, force_image_size=force_image_size).to(device)

    plugin = None
    if backbone == "facebook/sam-vit-base":
        patch_size = 16
    elif backbone in ["google/siglip2-so400m-patch14-224", "openai/clip-vit-large-patch14"]:
        patch_size = 14
    else:
        patch_size = extractor.model.config.patch_size
    
    if upsample:
        upsampler = torch.hub.load("wimmerth/anyup", "anyup", verbose=False).to(device).eval()
    else:
        upsampler = None

    if shift_patch_num[1] != 0:
        if backbone == "facebook/sam-vit-base":
            _input_size = 1024
        elif backbone in ["google/siglip2-so400m-patch14-224", "openai/clip-vit-large-patch14"]:
            _input_size = 224
        else:
            _input_size = extractor.model.config.image_size
        plugin = GlobalPEPlugin(
            extractor.model,
            backbone,
            _input_size,
            patch_size=patch_size,
            shift_patch_num=(0, shift_patch_num[1] if backbone != "facebook/sam-vit-base" else shift_patch_num[1] * 4)
        )
    for data in tqdm(dataset):
        imageL, imageR, disp_gt = data["left"][None], data["right"][None], data["disp"]
        valid_mask = torch.as_tensor(data["valid"]) * (1 - torch.as_tensor(data["occ_mask"], dtype=torch.long))
        valid_mask = valid_mask[0]

        result = run_model(
            extractor, plugin, imageL, imageR, disp_gt, valid_mask, upsample=upsample,
            upsampler=upsampler
        )
        result["name"] = data["name"]
        if results is None:
            results = pd.DataFrame([result])
        else:
            results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)

        results.to_csv(f"{dataset_name}_worldx_{shift_patch_num[1]}_{'upsample' if upsample else 'bilinear'}_{image_size}_train_{backbone.split('/')[1]}.csv", index=False)
