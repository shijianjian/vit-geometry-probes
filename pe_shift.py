
import os
import math
import argparse
from pathlib import Path
from typing import List, Tuple
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from einops import rearrange

from transformers import (
    AutoModel,
    AutoFeatureExtractor,
    AutoImageProcessor,
)


def generate_relative_position_index_swin(window_size):
    coords_h = torch.arange(window_size[0])
    coords_w = torch.arange(window_size[1])
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += window_size[0] - 1
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    relative_position_index = relative_coords.sum(-1)
    return relative_position_index


class GlobalPEPlugin(nn.Module):
    """
    Plugin to reindex positional embeddings onto a larger global grid.
    """
    def __init__(self, model, model_name, image_size=448, patch_size=16, shift_patch_num=(0, 1)):
        super().__init__()
        self.model = model
        self.model_name = model_name

        if model_name in ["facebook/dinov2-large"]:
            image_size = 518
            print(f"Setting image_size to {image_size} for {model_name}")

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_cols, self.num_rows = image_size // patch_size, image_size // patch_size
        self.num_global_cols, self.num_global_rows = self.num_cols + shift_patch_num[1], self.num_rows + shift_patch_num[0]

        # Extract the pretrained pos embedding
        if model_name in [
            "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        ]:
            pass
        elif model_name == "DeepGlint-AI/mlcd-vit-bigG-patch14-224":
            inv_freq = self.model.vision_model.vision_rotary_embedding.inv_freq
            # Generate position IDs for height and width dimensions
            self.hpos_ids = (
                torch.arange(
                    self.num_global_rows, device=inv_freq.device, dtype=inv_freq.dtype
                ).unsqueeze(1).expand(-1, self.num_cols)
            )
            self.wpos_ids = (
                torch.arange(
                    self.num_global_cols, device=inv_freq.device, dtype=inv_freq.dtype
                ).unsqueeze(0).expand(self.num_global_rows, -1)
            )

            # Generate the full rotary positional embeddings for the maximum grid size
            max_grid_size = max(self.num_global_cols, self.num_global_rows)
            seq = torch.arange(max_grid_size, device=inv_freq.device, dtype=inv_freq.dtype)
            self.rotary_pos_emb_full = torch.outer(seq, inv_freq)

        elif model_name == "microsoft/swinv2-base-patch4-window16-256":
            for blk in self.model.encoder.layers:
                for layer in blk.blocks:
                    layer.attention.self.continuous_position_bias_mlp[0].weight = nn.Parameter(
                        pos_weight * layer.attention.self.continuous_position_bias_mlp[0].weight.clone())
                    layer.attention.self.continuous_position_bias_mlp.requires_grad = False

        elif model_name == "microsoft/swin-base-patch4-window7-224":
            self.global_pe_swin = []
            for blk in self.model.encoder.layers:
                for layer in blk.blocks:
                    rpb = layer.attention.self.relative_position_bias_table
                    num_heads = rpb.shape[1]
                    hw = layer.attention.self.window_size

                    h, w = hw if isinstance(hw, (list, tuple)) else (hw, hw)
                    new_h = h + shift_patch_num[0]
                    new_w = w + shift_patch_num[1]
                    num_global_rows = new_h * 2 - 1
                    num_global_cols = new_w * 2 - 1
                    num_rows = h * 2 - 1
                    num_cols = w * 2 - 1

                    bias_2d = rpb.reshape(
                        num_rows, num_cols, num_heads).permute(2, 0, 1)  # (heads, h2, w2)

                    # interpolate to new size
                    self.global_pe_swin.append((
                        F.interpolate(
                            bias_2d.unsqueeze(0),
                            size=(num_global_rows, num_global_cols),
                            mode="bilinear",
                            align_corners=True
                        ).reshape(num_heads, -1).permute(1, 0),
                        (h, w),
                        (new_h, new_w),
                    ))

        elif model_name in [
            "facebook/data2vec-vision-large",
            "microsoft/beit-base-patch16-224-pt22k"
        ]:
            rpb = model.encoder.relative_position_bias.relative_position_bias_table  # (2H-1)*(2W-1), num_heads
            num_heads = rpb.shape[1]

            # infer local grid size
            hw = model.encoder.relative_position_bias.window_size
            h, w = hw if isinstance(hw, (list, tuple)) else (hw, hw)
            new_h = h + shift_patch_num[0]
            new_w = w + shift_patch_num[1]

            self.num_global_rows = new_h * 2 - 1
            self.num_global_cols = new_w * 2 - 1
            self.num_rows = h * 2 - 1
            self.num_cols = w * 2 - 1

            self.original_bias_index_fn = self.model.encoder.relative_position_bias.generate_relative_position_index
            # reshape to 2D bias map
            bias_2d = rpb[:-3].reshape(
                self.num_rows, self.num_cols, num_heads).permute(2, 0, 1)  # (heads, h2, w2)

            # interpolate to new size
            self.global_pe = F.interpolate(
                bias_2d.unsqueeze(0),
                size=(self.num_global_rows, self.num_global_cols),
                mode="bilinear",
                align_corners=True
            )

            self.global_pe_cls = rpb[-3:]
        else:
            if model_name in [
                "facebook/metaclip-b16-fullcc2.5b",
                "google/siglip2-so400m-patch14-224",
                "google/siglip2-base-patch16-224",
                "openai/clip-vit-large-patch14",
            ]:
                pos_embed = model.vision_model.embeddings.position_embedding.weight
                pos_embed = pos_embed[None]
            elif model_name in ["facebook/sam-vit-base"]:
                pos_embed = model.vision_encoder.pos_embed
                pos_embed = pos_embed.reshape(pos_embed.shape[0], -1, pos_embed.shape[-1])
            else:
                pos_embed = model.embeddings.position_embeddings

            if model_name in [
                "facebook/ijepa_vitg16_22k",
                "google/siglip2-so400m-patch14-224",
                "google/siglip2-base-patch16-224",
                "facebook/sam-vit-base",
                "facebook/data2vec-vision-large",
                "microsoft/beit-base-patch16-224-pt22k"
            ]:
                self.cls_token = None
                patch_embed = pos_embed
            else:
                self.cls_token, patch_embed = pos_embed[:, :1, :], pos_embed[:, 1:, :]

            self.seq_len = pos_embed.shape[1]
            # Reshape to 2D grid
            h = w = int(patch_embed.shape[1] ** 0.5)
            try:
                patch_embed = patch_embed.reshape(1, h, w, -1)
            except Exception:
                raise ValueError(f"Patch embed shape: {patch_embed.shape}, h: {h}, w: {w}")

            # Create a big global grid, initialized as zeros or interpolated
            self.global_pe = F.interpolate(
                patch_embed.permute(0, 3, 1, 2),
                size=(self.num_global_rows, self.num_global_cols),
                mode='bilinear',
                align_corners=True
            )

    def generate_world_grid_coords(
        self,
        view_h: int,           # Height of the view (e.g., 37)
        view_w: int,           # Width of the view (e.g., 37)
        world_h: int,          # Total height of the world grid (e.g., 37)
        world_w: int,          # Total width of the world grid (e.g., 38)
        top_offset: int,       # Y-patch offset in the world (e.g., 0)
        left_offset: int,      # X-patch offset in the world (e.g., 1 for right img)
        device: torch.device = "cpu",
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Generates normalized [-1, 1] patch coordinates for a 'view'
        sliced from a larger 'world grid'.

        For RoPE
        """
        
        # 1. Create coordinates for the *entire* world grid
        # These are normalized from -1 to 1 across the *world* dimensions
        y_world, x_world = torch.meshgrid(
            torch.linspace(-1.0, 1.0, world_h, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, world_w, device=device, dtype=dtype),
            indexing="ij"
        )
        
        # 2. Stack them into a (y, x) grid
        # Shape: (world_h, world_w, 2)
        scale_y = world_h / 2
        scale_x = world_w / 2
        world_coords_grid = torch.stack([
            (y_world + 1) * scale_y,
            (x_world + 1) * scale_x
        ], dim=-1)

        # 3. Slice the grid to get the coordinates for our specific view
        # e.g., world_coords[0:37, 1:38, :]
        view_coords_grid = world_coords_grid[
            top_offset : top_offset + view_h,
            left_offset : left_offset + view_w,
            :
        ]

        # 4. Flatten to the (num_patches, 2) format expected by the RoPE module
        # Shape: (view_h * view_w, 2)
        view_coords_flat = view_coords_grid.flatten(0, 1)
        
        return view_coords_flat

    def update(self, offset_col=0):
        """
        Forward one image whose top-left patch starts at `offset_col`
        in the global grid.
        """
        if self.model_name in ["facebook/dinov3-vit7b16-pretrain-lvd1689m"]:
            world_patch_coords = self.generate_world_grid_coords(
                view_h=self.num_rows,
                view_w=self.num_cols,
                world_h=self.num_global_rows,
                world_w=self.num_global_cols,
                top_offset=0,
                left_offset=offset_col,
            )
            def fake_forward(pixel_values: torch.Tensor):
                device, dtype = pixel_values.device, pixel_values.dtype
                device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"
 
                patch_coords = world_patch_coords.to(device=device, dtype=torch.float32)

                angles = 2 * math.pi * patch_coords[:, :, None] * self.model.rope_embeddings.inv_freq[None, None, :]
                angles = angles.flatten(1, 2)
                angles = angles.tile(2)

                cos = torch.cos(angles)
                sin = torch.sin(angles)

                return cos.to(dtype=dtype), sin.to(dtype=dtype)
            self.model.rope_embeddings.forward = fake_forward

        elif self.model_name == "DeepGlint-AI/mlcd-vit-bigG-patch14-224":
            # Flatten and stack the position IDs
            pos_ids = torch.stack([self.hpos_ids.flatten(), self.wpos_ids[:, offset_col:offset_col + self.num_cols].flatten()], dim=-1)
            
            # assert False, (self.hpos_ids.shape, self.wpos_ids.shape, self.pos_ids.shape, (self.num_global_rows, self.num_global_cols))
            # Select and flatten the embeddings based on the position IDs
            rotary_pos_emb = self.rotary_pos_emb_full[pos_ids.long()].flatten(1)

            def fake_forward(a, b):
                return rotary_pos_emb
            self.model.vision_model.vision_rotary_embedding.forward = fake_forward

        elif self.model_name == "microsoft/swin-base-patch4-window7-224":
            count = 0
            for blk in self.model.encoder.layers:
                for layer in blk.blocks:
                    global_pe, (num_rows, num_cols), (num_global_rows, num_global_cols) = self.global_pe_swin[count]
                    count += 1
                    base_index = generate_relative_position_index_swin((num_rows, num_cols))
                    shifted = base_index.clone()
                    shifted[1:, 1:] += offset_col
                    # assert False, (shifted, base_index, shifted.shape, offset_col, global_pe.shape, (num_rows, num_cols), (num_global_rows, num_global_cols))

                    layer.attention.self.relative_position_index = shifted
                    layer.attention.self.relative_position_bias_table = nn.Parameter(global_pe)
                    layer.attention.self.relative_position_bias_table.requires_grad = False
        elif self.model_name in ["facebook/data2vec-vision-large", "microsoft/beit-base-patch16-224-pt22k"]:
            pe_slice = self.global_pe.reshape(self.global_pe.shape[1], -1).permute(1, 0)
            pe = torch.cat([pe_slice, self.global_pe_cls], dim=0)

            def shifted_relative_index(*args, **kwargs):
                base_index = self.original_bias_index_fn(*args, **kwargs)
                shifted = base_index.clone()
                # assert False, (shifted.shape, offset_col)
                shifted[1:, 1:] += offset_col  # small offset inside domain (tune if W large)
                return shifted
            self.model.encoder.relative_position_bias.generate_relative_position_index = shifted_relative_index
            self.model.encoder.relative_position_bias.relative_position_bias_table = nn.Parameter(pe)
            self.model.encoder.relative_position_bias.relative_position_bias_table.requires_grad = False
        else:
            # slice out the corresponding region from global PE
            pe_slice = self.global_pe[..., offset_col:offset_col + self.num_cols]
            pe_slice = pe_slice.permute(0, 2, 3, 1).reshape(1, self.num_rows * self.num_cols, -1)
            if self.cls_token is not None:
                pe = torch.cat([self.cls_token, pe_slice], dim=1)
            else:
                pe = pe_slice
            assert pe.shape[1] == self.seq_len, f"pe.shape: {pe.shape}, seq_len: {self.seq_len}"
            
            if self.model_name in [
                "facebook/metaclip-b16-fullcc2.5b", "google/siglip2-so400m-patch14-224",
                "openai/clip-vit-large-patch14", "google/siglip2-base-patch16-224",
            ]:
                pe = pe[0]

            if self.model_name in [
                "facebook/metaclip-b16-fullcc2.5b",
                "google/siglip2-so400m-patch14-224",
                "google/siglip2-base-patch16-224",
                "openai/clip-vit-large-patch14",
            ]:
                self.model.vision_model.embeddings.position_embedding.weight = nn.Parameter(pe)
                self.model.vision_model.embeddings.position_embedding.weight.requires_grad = False
            elif self.model_name == "facebook/sam-vit-base":
                self.model.vision_encoder.pos_embed = nn.Parameter(pe.reshape(1, self.num_rows, self.num_cols, -1))
                self.model.vision_encoder.pos_embed.requires_grad = False
            else:
                self.model.embeddings.position_embeddings = nn.Parameter(pe)
                self.model.embeddings.position_embeddings.requires_grad = False

# ---------------------------
# Utilities / Models
# ---------------------------

def psnr(mse):
    if mse == 0:
        return float('inf')
    return 10.0 * math.log10(1.0 / mse)


class PatchDecoder(nn.Module):
    """
    MLP decoder that reconstructs a patch (3 x patch_size x patch_size)
    from a feature vector of dim `in_dim`.
    The decoder is applied independently at each spatial location.
    """
    def __init__(self, in_dim: int, patch_size: int = 8, hidden: int = 512):
        super().__init__()
        self.patch_size = patch_size
        self.out_dim = 3 * patch_size * patch_size
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.out_dim)
        )

    def forward(self, feat):  # feat: (B, C, H, W)
        B, C, H, W = feat.shape
        # (B*H*W, C)
        x = feat.permute(0, 2, 3, 1).reshape(B * H * W, C)
        x = self.net(x)  # (B*H*W, out_dim)
        x = x.view(B, H, W, 3, self.patch_size, self.patch_size)
        # return (B, 3, H*ps, W*ps) is done by reassembly function
        return x


class HiddenDimDecoder(nn.Module):
    """
    MLP decoder that reconstructs a patch (3 x patch_size x patch_size)
    from a feature vector of dim `in_dim`.
    The decoder is applied independently at each spatial location.
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1, padding=0, bias=False),
        )

    def forward(self, feat):  # feat: (B, C, H, W)
        B, C, H, W = feat.shape
        x = self.net(feat)  # feat: (B, C, H, W)
        return x


def patches_to_image(patches: torch.Tensor) -> torch.Tensor:
    """
    patches: (B, Hf, Wf, 3, ps, ps)
    returns: (B, 3, Hf*ps, Wf*ps)
    """
    B, Hf, Wf, Cc, ps, ps2 = patches.shape
    assert ps == ps2
    # reorder to (B, 3, Hf, ps, Wf, ps)
    x = patches.permute(0, 3, 1, 4, 2, 5)
    # combine spatial dims
    x = x.reshape(B, Cc, Hf * ps, Wf * ps)
    return x


def assemble_patches_from_image(img: torch.Tensor, patch_size=8) -> torch.Tensor:
    """
    For debug/testing: split images into non-overlapping patches.
    img: (B, 3, H, W)
    returns patches (B, H/ps, W/ps, 3, ps, ps)
    """
    B, C, H, W = img.shape
    assert H % patch_size == 0 and W % patch_size == 0
    Hf = H // patch_size
    Wf = W // patch_size
    # reshape
    patches = img.reshape(B, C, Hf, patch_size, Wf, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5)  # (B, Hf, Wf, C, ps, ps)
    return patches


# ---------------------------
# Feature extraction helpers
# ---------------------------

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


def extract_patch_feature_map(
    processor,
    model,
    images_tensor: torch.Tensor,
    device: torch.device,
    target_grid_size: int = 16,
):
    """
    - images_tensor: (B, 3, H, W), pixel values in [0,1], dtype=float32
    - processor: AutoImageProcessor or AutoFeatureExtractor (used for normalization/resizing if necessary)
    - model: AutoModel
    Returns: feature_map (B, C, target_grid_size, target_grid_size)
    """
    # prepare pixel_values using the processor's .preprocess or __call__ if available
    # Many HF processors expect PIL images; but they also accept tensors if we pass "images="
    # We'll call processor(images=..., return_tensors="pt") but must move to device afterwards.
    # Convert channels-first to channels-last for processor which expects PIL or HWC
    B, C, H, W = images_tensor.shape
    # Convert to list of PIL-like arrays by permuting to (B, H, W, C) and to numpy if needed
    # But processors accept torch tensors: pass images_tensor.permute(0,2,3,1)
    input_kwargs = {"images": images_tensor.permute(0, 2, 3, 1).to(device), "return_tensors": "pt"}
    with torch.no_grad():
        try:
            proc_out = processor(**input_kwargs)
        except Exception:
            # some processors require CPU numpy; fallback to CPU numpy
            imgs_cpu = (images_tensor.permute(0,2,3,1).cpu().numpy() * 255).astype('uint8')
            proc_out = processor(images=list(imgs_cpu), return_tensors="pt")
        # move pixel_values (if present) to device
        if "pixel_values" in proc_out:
            pixel_values = proc_out["pixel_values"].to(device)
        elif "pixel_values" in proc_out.keys():
            pixel_values = proc_out["pixel_values"].to(device)
        else:
            # Try last key
            # fallback: assume processor returned images as "images"
            pixel_values = proc_out[list(proc_out.keys())[0]].to(device)

        # call model
        # Many vision AutoModels accept pixel_values as kwarg
        with torch.no_grad():
            
            if hasattr(model, "vision_model"):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=torch.zeros((1,), dtype=torch.long, device=pixel_values.device),
                    output_hidden_states=True
                )
            else:
                outputs = model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)

        # Try to extract a spatial token map:
        feat = None
        # 1) standard ViT-style: last_hidden_state
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            feat = outputs.last_hidden_state  # (B, seq_len, C)
        # 2) some models put it under 'vision_model_output' or nested
        elif hasattr(outputs, "vision_model_output") and getattr(outputs.vision_model_output, "last_hidden_state", None) is not None:
            feat = outputs.vision_model_output.last_hidden_state
        elif isinstance(outputs, dict) and "last_hidden_state" in outputs:
            feat = outputs["last_hidden_state"]

        if feat is None:
            # fallback: try hidden_states[-1]
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                feat = outputs.hidden_states[-1]
            elif hasattr(outputs, "vision_hidden_states"):
                feat = outputs.vision_hidden_states[-1]
                B, C, H, W = feat.shape
                feat = feat.reshape(B, C, H * W).permute(0, 2, 1)
            else:
                raise RuntimeError("Could not find feature map in model outputs. Output keys: " + ", ".join([k for k in dir(outputs) if not k.startswith("_")]))

        if feat.dim() == 4:
            B, C, H, W = feat.shape
            feat = feat.permute(0, 2, 3, 1).reshape(B, -1, C)

        # feat shape (B, seq_len, C)
        Bf, seq_len, Cdim = feat.shape

        # Remove cls token if present (heuristic: seq_len is not a square number)
        sq = int(round(math.sqrt(seq_len)))
        has_cls = False
        if sq * sq != seq_len:
            # normally the first token is CLS
            has_cls = True
            if model.__class__.__name__ == "DINOv3ViTModel":
                other_tokens = model.config.num_register_tokens
            else:
                other_tokens = 0
            feat_no_cls = feat[:, 1 + other_tokens :, :]
            seq_len = seq_len - 1 - other_tokens
            sq = int(round(math.sqrt(seq_len)))
            feat = feat_no_cls

        Hf = sq
        Wf = sq
        # reshape to (B, Hf, Wf, C)
        feat_map = feat.view(Bf, Hf, Wf, Cdim).permute(0, 3, 1, 2).contiguous()  # (B, C, Hf, Wf)

        # If feature grid does not match target_grid_size, resize spatially via bilinear
        if target_grid_size is not None:
            assert feat_map.shape[2] == target_grid_size and feat_map.shape[3] == target_grid_size, f"Feature map shape: {feat_map.shape}, target grid size: {target_grid_size}"
                # feat_map = F.interpolate(feat_map, size=(target_grid_size, target_grid_size), mode='bilinear', align_corners=False)

    return feat_map  # (B, C, target, target)


def baseline_eval(
    model_id: str,
    processor,
    model,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    device: torch.device,
    epochs: int = 3,
    lr: float = 1e-4,
    image_size: int = 256,
    feature_grid: int = 16,
    patch_size: int = 8,
    shift_num: int = 1,
    global_pe: bool = False,
    **kwargs
):
    # extract a sample to get feature dim
    model_device = device
    model.to(model_device).eval()
    # get one batch
    sample_batch = next(iter(train_dataloader))
    imgs, _ = sample_batch  # imgs in range [0,1]
    imgs = imgs.to(device)
    feat_map = extract_patch_feature_map(
        processor, model, imgs, device, target_grid_size=None)
    B, C, Hf, Wf = feat_map.shape
    print(f"[{model_id}] Feature map shape sample: {feat_map.shape} from image {imgs.shape}")

    plugin = None
    if shift_num != 0 and global_pe:
        plugin = GlobalPEPlugin(
            model,
            model_id,
            1024 if model_id == "facebook/sam-vit-base" else image_size,
            patch_size=patch_size,
            shift_patch_num=(0, shift_num if model_id != "facebook/sam-vit-base" else shift_num * 4)
        )

    with torch.no_grad():
        total_mse = 0.0
        total_l1 = 0.0
        total_sim = 0.0
        best_loss = 0.0
        nimgs = 0
        for imgs, _ in tqdm(test_dataloader, desc=f"Eval {model_id}"):
            imgs = imgs.to(device)
            if plugin is not None:
                plugin.update(0)
            feat_map_1 = extract_patch_feature_map(
                processor, model, imgs[..., :image_size, :image_size], device, target_grid_size=None)
            if model_id == "facebook/sam-vit-base":
                feat_map_1 = F.interpolate(
                    feat_map_1, size=(feature_grid, feature_grid), mode='bilinear', align_corners=False)
            if plugin is not None:
                if model_id == "facebook/sam-vit-base":
                    plugin.update(shift_num * 4)  # We use smaller token feature map for SAM
                else:
                    plugin.update(shift_num)
            feat_map_2 = extract_patch_feature_map(
                processor, model, imgs[..., :image_size, -image_size:], device, target_grid_size=None)
            if model_id == "facebook/sam-vit-base":
                feat_map_2 = F.interpolate(
                    feat_map_2, size=(feature_grid, feature_grid), mode='bilinear', align_corners=False)

            if shift_num != 0:
                feat_map_1 = feat_map_1[..., shift_num:]
                feat_map_2 = feat_map_2[..., :-shift_num]

            mse = torch.mean((feat_map_1 - feat_map_2) ** 2).item()
            l1 = torch.mean(torch.abs(feat_map_1 - feat_map_2)).item()
            B, C, H, W = feat_map_1.shape
            sim = compute_semantic_similarity(
                feat_map_1.reshape(B, C, -1), feat_map_2.reshape(B, C, -1))

            total_mse += mse * imgs.shape[0]
            total_l1 += l1 * imgs.shape[0]
            nimgs += imgs.shape[0]
            total_sim += sim * imgs.shape[0]
        avg_mse = total_mse / nimgs
        avg_l1 = total_l1 / nimgs
        avg_sim = total_sim / nimgs
        avg_psnr = psnr(avg_mse)
        print(f"[{model_id}] Eval results: MSE={avg_mse:.6f}, L1={avg_l1:.6f}, PSNR={avg_psnr:.3f} dB, SIM={avg_sim:.6f}")

    return {"model_id": model_id, "mse": avg_mse, "l1": avg_l1, "psnr": avg_psnr, "sim": avg_sim, "best_loss": best_loss}


def compute_semantic_similarity(feat1, feat2):
    # feat1, feat2: B x C x N
    feat1 = F.normalize(feat1, dim=1)  # normalize along channel dim
    feat2 = F.normalize(feat2, dim=1)
    # compute similarity along channel dim, resulting in B x N
    sim = F.cosine_similarity(feat1, feat2, dim=1)
    # average over batch and spatial dims
    return sim.mean().item()

# ---------------------------
# Data loader
# ---------------------------

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

# ---------------------------
# Main
# ---------------------------

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    print("Using device:", device)
    # build dataloader
    train_loder, test_loder = build_imagenette_loader(
        args.imagenette_root, batch_size=args.batch_size, image_size=args.image_size, patch_size=args.patch_size * args.shift_num, workers=args.workers)

    results = []
    for model_id in args.models:
        print("Loading model:", model_id)
        try:
            processor, model = build_processor_and_model(model_id, device)
        except Exception as e:
            print(f"Failed to load {model_id}: {e}")
            continue

        if model_id in [
            "facebook/dinov2-large", "google/siglip2-so400m-patch14-224",
            "openai/clip-vit-large-patch14", "DeepGlint-AI/mlcd-vit-bigG-patch14-224",
        ]:
            feature_grid = 16
            patch_size = 14
        else:
            feature_grid = args.feature_grid
            patch_size = args.patch_size
            
        if args.load_weights == "best":
            decoer_weights_path = os.path.join(args.save_dir, f"best_decoder_{args.shift_num}_{model_id.replace('/', '_')}.pth")
        elif args.load_weights == "epoch":
            raise Exception("Not implemented")
        else:
            decoer_weights_path = None
            
        fn = baseline_eval

        res = fn(
            model_id=model_id,
            processor=processor,
            model=model,
            train_dataloader=train_loder,
            test_dataloader=test_loder,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            image_size=args.image_size,
            feature_grid=feature_grid,
            patch_size=patch_size,
            save_dir=args.save_dir,
            decoer_weights_path=decoer_weights_path,
            shift_num=args.shift_num,
            global_pe=args.global_pe
        )
        results.append(res)

    # Print summary
    for r in results:
        print(f"{r['model_id']}: MSE={r['mse']:.6f}, L1={r['l1']:.6f}, PSNR={r['psnr']:.2f} dB, SIM={r['sim']:.6f}, best_loss={r['best_loss']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenette_root", type=str, default="./data", help="Path to imagenette train/ (ImageFolder structure)")
    parser.add_argument("--models", type=str, nargs="+",
                        default=[
                            # "facebook/dinov2-large",
                            # "facebook/dinov3-vit7b16-pretrain-lvd1689m",
                            # "facebook/ijepa_vitg16_22k",
                            # "facebook/metaclip-b16-fullcc2.5b",
                            # "facebook/deit-base-patch16-224",
                            # "google/siglip2-so400m-patch14-224",
                            # "google/siglip2-base-patch16-224",
                            # "facebook/sam-vit-base",
                            # "openai/clip-vit-large-patch14",
                            # "facebook/dino-vitb16",
                            # "google/vit-base-patch16-224",

                            # "microsoft/beit-base-patch16-224-pt22k",
                            # "facebook/data2vec-vision-large",

                            # "facebook/sam2-hiera-large",
                            # "facebook/vit-mae-base",  # Contains Random Masking, breaks the offset settings.
                            # "nyu-visionx/moco-v3-vit-b",

                            # "microsoft/swin-base-patch4-window7-224",
                            "microsoft/swinv2-base-patch4-window16-256",
                            # "DeepGlint-AI/mlcd-vit-bigG-patch14-224",

                            # "nvidia/segformer-b4-finetuned-ade-512-512",  # NO PE in the model
                        ],
                        help="Hugging Face model ids to probe.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--feature_grid", type=int, default=14)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_shift")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--load_weights", type=str, default=None, help="best | epoch num")
    parser.add_argument("--shift_num", type=int, default=1)
    parser.add_argument("--global_pe", action="store_true", default=False)
    args = parser.parse_args()

    for shift_num in [1, 2, 3]:
        args.shift_num = shift_num
        print(f"Probing with shift_num={shift_num}")
        main(args)
