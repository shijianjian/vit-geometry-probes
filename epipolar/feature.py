

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from torchvision import transforms, datasets
from PIL import Image
import timm
from pe_ops import zero_out_pos_embedding, pos_shuffle_embedding


class FeatureExtractor(nn.Module):
    def __init__(
        self, model_name="facebook/dinov2-base", device="cuda", if_zero_pos=False,
        pos_shuffle=False, force_image_size: int = None
    ):
        super().__init__()
        name = model_name.split("/")
        self.is_timm = name[0] == "timm"
        self.model_name = model_name
        self.force_image_size = force_image_size
        if self.is_timm:
            self.processor, self.model = build_timm_model(name[1], device)
        else:
            self.processor, self.model = build_processor_and_model(model_name, device)
            self.model.eval()

            if if_zero_pos:
                self.model = zero_out_pos_embedding(self.model, model_name)

            if pos_shuffle:
                self.model = pos_shuffle_embedding(self.model, model_name)

    @torch.no_grad()
    def extract(self, images_tensor, return_raw_tokens=False, reshuffle_pos=False):
        if reshuffle_pos:
            self.model = pos_shuffle_embedding(self.model, self.model_name)
        if self.is_timm:
            return extract_timm_feature(self.processor, self.model, images_tensor, self.force_image_size)
        assert self.force_image_size is None, "image_size must be None for non-timm models"
        return extract_patch_feature_map(
            processor=self.processor, model=self.model, images_tensor=images_tensor, device=self.model.device, return_raw_tokens=return_raw_tokens)


def build_timm_model(model_id: str, device: torch.device):
    model = timm.create_model(model_id, pretrained=True)
    model = model.to(device).eval()

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    return transforms, model


def extract_timm_feature(processor, model, images_tensor, image_size: int = None):
    image_tensor = processor(images_tensor)
    if image_size is not None:
        image_tensor = F.interpolate(image_tensor, size=(image_size, image_size), mode='bilinear')
    feat = model.forward_features(image_tensor)
    
    return _token_to_feature_map(feat)


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
    return_raw_tokens: bool = False
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
            imgs_cpu = (images_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
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
            try:
                outputs = model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
            except Exception:
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=torch.zeros((1,), dtype=torch.long, device=pixel_values.device),
                    output_hidden_states=True
                )

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
                B, H, W, C = feat.shape
                feat = feat.reshape(B, H * W, C)
            else:
                raise RuntimeError("Could not find feature map in model outputs. Output keys: " + ", ".join([k for k in dir(outputs) if not k.startswith("_")]))

    if return_raw_tokens:
        return feat

    return _token_to_feature_map(model, feat)


def _token_to_feature_map(model, feat):
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
    return feat_map  # (B, C, target, target)
