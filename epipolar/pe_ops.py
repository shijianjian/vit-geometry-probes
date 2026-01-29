import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def zero_out_pos_embedding(model, model_name):
    if model_name == "facebook/dinov3-vit7b16-pretrain-lvd1689m":
        orig_forward = model.rope_embeddings.forward
        def fake_forward(x):
            a, b = orig_forward(x)
            return torch.zeros_like(a), torch.zeros_like(b)
        model.rope_embeddings.forward = fake_forward
    elif model_name == "DeepGlint-AI/mlcd-vit-bigG-patch14-224":
        # Flatten and stack the position IDs
        # assert False, (self.hpos_ids.shape, self.wpos_ids.shape, self.pos_ids.shape, (self.num_global_rows, self.num_global_cols))
        # Select and flatten the embeddings based on the position IDs
        orig_forward = model.vision_model.vision_rotary_embedding.forward
        def fake_forward(a, b):
            return orig_forward(a, b) * 0.
        model.vision_model.vision_rotary_embedding.forward = fake_forward
    elif model_name == "facebook/sam-vit-base":
        model.vision_encoder.pos_embed = nn.Parameter(
            torch.zeros_like(model.vision_encoder.pos_embed.clone()))
        # feature_grid = 1024 // args.patch_size
    elif model_name == "microsoft/swin-base-patch4-window7-224":
        for blk in model.encoder.layers:
            for layer in blk.blocks:
                layer.attention.self.relative_position_bias_table = nn.Parameter(
                    torch.zeros_like(layer.attention.self.relative_position_bias_table)
                )
                layer.attention.self.relative_position_bias_table.requires_grad = False
    elif model_name == "microsoft/swinv2-base-patch4-window16-256":
        for blk in model.encoder.layers:
            for layer in blk.blocks:
                layer.attention.self.continuous_position_bias_mlp[0].weight = nn.Parameter(
                    torch.zeros_like(layer.attention.self.continuous_position_bias_mlp[0].weight.clone()))
                # layer.attention.self.continuous_position_bias_mlp[2].weight = nn.Parameter(
                #     pos_weight * layer.attention.self.continuous_position_bias_mlp[2].weight.clone())
                layer.attention.self.continuous_position_bias_mlp.requires_grad = False
    elif model_name in ["facebook/data2vec-vision-large", "microsoft/beit-base-patch16-224-pt22k"]:
        model.encoder.relative_position_bias.relative_position_bias_table = nn.Parameter(
            torch.zeros_like(model.encoder.relative_position_bias.relative_position_bias_table)
        )
        model.encoder.relative_position_bias.relative_position_bias_table.requires_grad = False
    elif model_name in [
        "facebook/metaclip-b16-fullcc2.5b", "google/siglip2-so400m-patch14-224",
        "openai/clip-vit-large-patch14", "google/siglip2-base-patch16-224",
    ]:
        model.vision_model.embeddings.position_embedding.weight = nn.Parameter(
            torch.zeros_like(model.vision_model.embeddings.position_embedding.weight))
        model.vision_model.embeddings.position_embedding.requires_grad = False
    else:
        model.embeddings.position_embeddings = nn.Parameter(torch.zeros_like(model.embeddings.position_embeddings.data))
        model.embeddings.position_embeddings.requires_grad = False
    return model


def _tensor_shuffle(tensors, model_name, seed=None):
    generator=torch.Generator().manual_seed(seed) if seed is not None else None
    if model_name == "facebook/dinov3-vit7b16-pretrain-lvd1689m":
        a, b = tensors
        ind = torch.randperm(a.shape[0], generator=generator)
        return a[ind], b[ind]
    if model_name in [
        "facebook/dino-vitb16",
        "facebook/dinov2-large",
        "microsoft/swin-base-patch4-window7-224",
        "google/vit-base-patch16-224"
    ]:
        # assert False, (tensors.shape, tensors, torch.cat([
        #     tensors[:, :1],
        #     tensors[:, 1:][:, torch.randperm(tensors.shape[1] - 1, generator=generator)]
        # ], dim=1))
        # Keep the cls token fixed
        return torch.cat([
            tensors[:, :1],
            tensors[:, 1:][:, torch.randperm(tensors.shape[1] - 1, generator=generator)]
        ], dim=1)
    if model_name == "google/siglip2-so400m-patch14-224":
        return tensors[:, torch.randperm(tensors.shape[1], generator=generator)]
    if model_name in [
        "facebook/ijepa_vitg16_22k",
        "facebook/deit-base-patch16-224"
    ]:
        return tensors[:, torch.randperm(tensors.shape[1], generator=generator)]
    if model_name in [
        "microsoft/swinv2-base-patch4-window16-256",
        "DeepGlint-AI/mlcd-vit-bigG-patch14-224",
    ]:
        return tensors[torch.randperm(tensors.shape[0], generator=generator)]
    if model_name in [
        "openai/clip-vit-large-patch14"
    ]:
        return torch.cat([
            tensors[:1],
            tensors[1:][torch.randperm(tensors.shape[0] - 1, generator=generator)]
        ], dim=0)
    if model_name in [
        "facebook/sam-vit-base"
    ]:
        b, h, w, c = tensors.shape
        tensors = tensors.reshape(b, -1, c)
        tensors = tensors[:, torch.randperm(tensors.shape[1], generator=generator)]
        return tensors.reshape(b, h, w, c)
    if model_name in [
        "facebook/data2vec-vision-large",
        "microsoft/beit-base-patch16-224-pt22k",
    ]:
        return torch.cat([
            tensors[:, :-3][:, torch.randperm(tensors.shape[1] - 3, generator=generator)],
            tensors[:, -3:],
        ], dim=1)
    raise NotImplementedError(f"Model name: {model_name}")


def pos_shuffle_embedding(model, model_name):
    if model_name == "facebook/dinov3-vit7b16-pretrain-lvd1689m":
        orig_forward = model.rope_embeddings.forward
        def fake_forward(x):
            a, b = orig_forward(x)
            return _tensor_shuffle((a, b), model_name)
        model.rope_embeddings.forward = fake_forward
    elif model_name == "DeepGlint-AI/mlcd-vit-bigG-patch14-224":
        # Flatten and stack the position IDs
        # assert False, (self.hpos_ids.shape, self.wpos_ids.shape, self.pos_ids.shape, (self.num_global_rows, self.num_global_cols))
        # Select and flatten the embeddings based on the position IDs
        orig_forward = model.vision_model.vision_rotary_embedding.forward
        def fake_forward(a, b):
            return _tensor_shuffle(orig_forward(a, b), model_name)
        model.vision_model.vision_rotary_embedding.forward = fake_forward
    elif model_name == "facebook/sam-vit-base":
        model.vision_encoder.pos_embed = nn.Parameter(
            _tensor_shuffle(model.vision_encoder.pos_embed.clone(), model_name))
        # feature_grid = 1024 // args.patch_size
    elif model_name == "microsoft/swin-base-patch4-window7-224":
        for blk in model.encoder.layers:
            for layer in blk.blocks:
                layer.attention.self.relative_position_bias_table = nn.Parameter(
                    _tensor_shuffle(layer.attention.self.relative_position_bias_table, model_name, seed=42)
                )
                layer.attention.self.relative_position_bias_table.requires_grad = False
    elif model_name == "microsoft/swinv2-base-patch4-window16-256":
        for blk in model.encoder.layers:
            for layer in blk.blocks:
                layer.attention.self.continuous_position_bias_mlp[0].weight = nn.Parameter(
                    _tensor_shuffle(layer.attention.self.continuous_position_bias_mlp[0].weight.clone(), model_name, seed=42))
                # layer.attention.self.continuous_position_bias_mlp[2].weight = nn.Parameter(
                #     pos_weight * layer.attention.self.continuous_position_bias_mlp[2].weight.clone())
                layer.attention.self.continuous_position_bias_mlp.requires_grad = False
    elif model_name in ["facebook/data2vec-vision-large", "microsoft/beit-base-patch16-224-pt22k"]:
        model.encoder.relative_position_bias.relative_position_bias_table = nn.Parameter(
            _tensor_shuffle(model.encoder.relative_position_bias.relative_position_bias_table, model_name)
        )
        model.encoder.relative_position_bias.relative_position_bias_table.requires_grad = False
    elif model_name in [
        "facebook/metaclip-b16-fullcc2.5b", "google/siglip2-so400m-patch14-224",
        "openai/clip-vit-large-patch14", "google/siglip2-base-patch16-224",
    ]:
        model.vision_model.embeddings.position_embedding.weight = nn.Parameter(
            _tensor_shuffle(model.vision_model.embeddings.position_embedding.weight, model_name))
        model.vision_model.embeddings.position_embedding.requires_grad = False
    else:
        model.embeddings.position_embeddings = nn.Parameter(
            _tensor_shuffle(model.embeddings.position_embeddings.data, model_name))
        model.embeddings.position_embeddings.requires_grad = False
    return model


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
            "microsoft/beit-base-patch16-224-pt22k",
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

    # def generate_world_grid_coords(
    #     self,
    #     view_h: int,           # Height of the view (e.g., 37)
    #     view_w: int,           # Width of the view (e.g., 37)
    #     world_h: int,          # Total height of the world grid (e.g., 37)
    #     world_w: int,          # Total width of the world grid (e.g., 38)
    #     top_offset: int,       # Y-patch offset in the world (e.g., 0)
    #     left_offset: int,      # X-patch offset in the world (e.g., 1 for right img)
    #     device: torch.device = "cpu",
    #     dtype: torch.dtype = torch.float32
    # ) -> torch.Tensor:
    #     """
    #     Generates normalized [-1, 1] patch coordinates for a 'view'
    #     sliced from a larger 'world grid'.

    #     For RoPE
    #     """
        
    #     # 1. Create coordinates for the *entire* world grid
    #     # These are normalized from -1 to 1 across the *world* dimensions
    #     y_world, x_world = torch.meshgrid(
    #         torch.linspace(-1.0, 1.0, world_h, device=device, dtype=dtype),
    #         torch.linspace(-1.0, 1.0, world_w, device=device, dtype=dtype),
    #         indexing="ij"
    #     )
        
    #     # 2. Stack them into a (y, x) grid
    #     # Shape: (world_h, world_w, 2)
    #     scale_y = world_h / 2
    #     scale_x = world_w / 2
    #     world_coords_grid = torch.stack([
    #         (y_world + 1) * scale_y,
    #         (x_world + 1) * scale_x
    #     ], dim=-1)

    #     # 3. Slice the grid to get the coordinates for our specific view
    #     # e.g., world_coords[0:37, 1:38, :]
    #     view_coords_grid = world_coords_grid[
    #         top_offset : top_offset + view_h,
    #         left_offset : left_offset + view_w,
    #         :
    #     ]

    #     # 4. Flatten to the (num_patches, 2) format expected by the RoPE module
    #     # Shape: (view_h * view_w, 2)
    #     view_coords_flat = view_coords_grid.flatten(0, 1)
        
    #     return view_coords_flat

    def update(self, offset_col=0):
        """
        Forward one image whose top-left patch starts at `offset_col`
        in the global grid.
        """
        if self.model_name in ["facebook/dinov3-vit7b16-pretrain-lvd1689m"]:
            from transformers.models.dinov3_vit.modeling_dinov3_vit import (
                get_patches_center_coordinates,
                augment_patches_center_coordinates,
            )
            
            def fake_forward(pixel_values: torch.Tensor):
                device, dtype = pixel_values.device, pixel_values.dtype
                device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"
 
                world_patch_coords = get_patches_center_coordinates(
                    self.num_global_rows,
                    self.num_global_cols,
                    dtype=torch.float32,
                    device=device_type,
                )
                world_patch_coords = world_patch_coords.reshape(self.num_global_rows, self.num_global_cols, 2)
                patch_coords = world_patch_coords[..., offset_col:offset_col + self.num_cols, :].reshape(-1, 2)

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

            if self.model_name == "facebook/dinov3-vit7b16-pretrain-lvd1689m":
                raise NotImplementedError("Dinov3 is not supported right now")
            elif self.model_name in [
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
