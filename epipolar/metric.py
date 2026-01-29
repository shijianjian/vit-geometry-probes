import torch
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn.functional as F

import torch, torch.nn.functional as F

import torch
import torch.nn.functional as F

def _norm_features(feat):
    # mean-center per map then L2 norm per channel
    feat = feat - feat.mean(dim=[2,3], keepdim=True)
    return F.normalize(feat, dim=1)

def _build_cost_vol(featL, featR, max_disp, tau, sign, invalid_fill=-1e4):
    """
    sign = +1  -> assume xR = x + d  (shift right image LEFT by d)
    sign = -1  -> assume xR = x - d  (shift right image RIGHT by d)
    """
    B, C, H, W = featL.shape
    vols = []
    for d in range(max_disp + 1):
        if sign == +1:
            # shift LEFT by d (negative roll)
            shifted = torch.roll(featR, shifts=-d, dims=-1)
            # rightmost d columns become invalid after left-shift
            if d > 0:
                shifted[..., :, W - d:] = 0
        else:
            # shift RIGHT by d (positive roll)
            shifted = torch.roll(featR, shifts=+d, dims=-1)
            # leftmost d columns invalid after right-shift
            if d > 0:
                shifted[..., :, :d] = 0
        sim = (featL * shifted).sum(1, keepdim=True)
        # mark invalid borders as very low so softmax ignores them
        if d > 0:
            if sign == +1:
                sim[..., :, W - d:] = invalid_fill
            else:
                sim[..., :, :d] = invalid_fill
        vols.append(sim)
    Cvol = torch.cat(vols, dim=1)  # (B, D+1, H, W)
    P = F.softmax(tau * Cvol, dim=1)
    disp_vals = torch.arange(0, max_disp + 1, device=featL.device, dtype=P.dtype).view(1, -1, 1, 1)
    d_hat = (P * disp_vals).sum(1, keepdim=True)  # (B,1,H,W)
    return Cvol, d_hat

@torch.no_grad()
def evaluate_epipolar_consistency_new(
    featL, featR, disp_gt, valid_mask,
    max_disp=192, tau=20.0, device='cuda',
    auto_sign=True, sign_if_known=None, encoder_stride=16
):
    """
    Dense zero-shot epipolar probe with auto sign detection.
    Outputs pixel-level EPE/D1 after accounting for encoder stride.
    """
    featL, featR = featL.to(device), featR.to(device)
    disp_gt = disp_gt.to(device)
    valid_mask = valid_mask.to(device).bool()
    
    # --- Compute feature scaling factor ---
    scale = featR.shape[-1] / disp_gt.shape[-1]

    # --- Resize GT and mask to feature resolution (but store pixel-scale copy) ---
    disp_gt_pix = disp_gt.clone()               # original pixel-level disparity
    disp_gt_feat = F.interpolate(disp_gt * scale, size=(featR.shape[-2], featR.shape[-1]), mode='nearest')
    valid_mask = F.interpolate(valid_mask.float(), size=(featR.shape[-2], featR.shape[-1]), mode='nearest').bool()

    # --- Normalize features ---
    featL = _norm_features(featL)
    featR = _norm_features(featR)

    B, C, H, W = featL.shape

    # --- Mask guard ---
    mask = valid_mask & (disp_gt_feat >= 0) & (disp_gt_feat <= max_disp)
    if mask.sum() == 0:
        return {"EPE": 0.0, "D1": 0.0, "Recall@1": 0.0, "Recall@2": 0.0, "Recall@5": 0.0, "EC-SIM": 0.0, "sign": 0}

    # --- Auto detect sign ---
    def ec_sim_given_sign(sgn):
        x_coords = torch.arange(W, device=device).view(1, 1, 1, W).repeat(B, 1, H, 1)
        xR = torch.clamp(x_coords + sgn * disp_gt_feat.round().long(), 0, W - 1)
        featR_warp = torch.gather(featR, dim=-1, index=xR.expand(-1, C, -1, -1))
        ec = (featL * featR_warp).sum(1, keepdim=True)
        return ec[mask].mean().item()

    if sign_if_known in (+1, -1):
        sign = sign_if_known
    elif auto_sign:
        valid_idx = mask.view(-1).nonzero(as_tuple=False).view(-1)
        if valid_idx.numel() > 4096:
            rand_sel = valid_idx[torch.randperm(valid_idx.numel(), device=device)[:4096]]
            m_samp = torch.zeros_like(mask.view(-1), dtype=torch.bool)
            m_samp[rand_sel] = True
            m_samp = m_samp.view_as(mask)
        else:
            m_samp = mask
        old_mask = mask
        mask = m_samp
        ec_pos = ec_sim_given_sign(+1)
        ec_neg = ec_sim_given_sign(-1)
        sign = +1 if ec_pos >= ec_neg else -1
        mask = old_mask
    else:
        sign = +1

    # --- Build cost volume ---
    cost_vol, disp_pred_feat = _build_cost_vol(featL, featR, max_disp, tau, sign)

    # --- Convert disparity back to pixel scale ---
    disp_pred_pix = disp_pred_feat / scale

    # --- Metrics ---
    num_valid = mask.sum().item()
    if num_valid == 0:
        return {"EPE": 0.0, "D1": 0.0, "Recall@1": 0.0, "Recall@2": 0.0, "Recall@5": 0.0, "EC-SIM": 0.0, "sign": sign}

    # EPE in pixels
    epe = (disp_pred_pix - disp_gt_pix).abs()
    epe_mean = epe[valid_mask].mean().item()

    # D1 in pixels
    disp_abs = disp_gt_pix.abs()
    thr = torch.maximum(torch.full_like(disp_gt_pix, 3.0), 0.05 * disp_abs)
    bad = epe > thr
    d1 = (bad & valid_mask).float().sum() / (valid_mask.float().sum() + 1e-6)

    # EC-SIM
    disp_int = disp_gt_feat.round().long()
    x_coords = torch.arange(W, device=device).view(1, 1, 1, W).repeat(B, 1, H, 1)
    xR = torch.clamp(x_coords + sign * disp_int, 0, W - 1)
    featR_warp = torch.gather(featR, dim=-1, index=xR.expand(-1, C, -1, -1))
    ec_sim = (featL * featR_warp).sum(1, keepdim=True)
    ec_sim_mean = ec_sim[mask].mean().item()

    # Recall@k
    gt_idx = disp_gt_feat.round().long().clamp(0, max_disp)
    topk_idx = torch.topk(cost_vol, k=5, dim=1).indices
    eq = (topk_idx == gt_idx)
    R1 = ((eq[:, :1].any(1)) & mask.squeeze(1)).float().sum() / (mask.float().sum() + 1e-6)
    R2 = ((eq[:, :2].any(1)) & mask.squeeze(1)).float().sum() / (mask.float().sum() + 1e-6)
    R5 = ((eq[:, :5].any(1)) & mask.squeeze(1)).float().sum() / (mask.float().sum() + 1e-6)

    return {
        "EPE": float(epe_mean),
        "D1": float(d1.item()),
        "Recall@1": float(R1.item()),
        "Recall@2": float(R2.item()),
        "Recall@5": float(R5.item()),
        "EC-SIM": float(ec_sim_mean),
        "sign": int(sign),
    }


@torch.no_grad()
def evaluate_epipolar_consistency_new_old(
    featL, featR, disp_gt, valid_mask,
    max_disp=192, tau=20.0, device='cuda',
    auto_sign=True, sign_if_known=None
):
    """
    Dense zero-shot epipolar probe with auto sign detection.

    Args:
        featL, featR: (1, C, H, W) raw features (any encoder). Will be mean-centered + channel L2-normalized.
        disp_gt:      (1, 1, H, W) disparity in pixels
        valid_mask:   (1, 1, H, W) binary mask for valid GT
        max_disp:     int
        tau:          temperature for softmax over disparities
        auto_sign:    if True, pick sign (+1 or -1) that yields higher EC-SIM on a sample
        sign_if_known: set to +1 to use xR = x + d, or -1 to use xR = x - d

    Returns:
        dict with EPE, D1, Recall@1/2/5, EC-SIM, and 'sign' used (+1 or -1)
    """
    featL, featR = featL.to(device), featR.to(device)
    disp_gt = disp_gt.to(device)
    valid_mask = valid_mask.to(device).bool()
    
    scale = featR.shape[-1] / disp_gt.shape[-1]
    disp_gt = F.interpolate(disp_gt * scale, size=(featR.shape[-2], featR.shape[-1]), mode='nearest')
    valid_mask = F.interpolate(valid_mask.float(), size=(featR.shape[-2], featR.shape[-1]), mode='nearest').bool()

    # 1) Channel normalization (more stable with mean-centering)
    featL = _norm_features(featL)
    featR = _norm_features(featR)

    B, C, H, W = featL.shape

    # Guard: valid pixels within disparity range
    mask = valid_mask & (disp_gt >= 0) & (disp_gt <= max_disp)
    if mask.sum() == 0:
        return {"EPE": 0.0, "D1": 0.0, "Recall@1": 0.0, "Recall@2": 0.0, "Recall@5": 0.0, "EC-SIM": 0.0, "sign": 0}

    # 2) Auto-detect sign if needed
    def ec_sim_given_sign(sgn):
        # warp featR to L using GT with a given sign
        x_coords = torch.arange(W, device=device).view(1, 1, 1, W).repeat(B, 1, H, 1)
        xR = x_coords + sgn * disp_gt.round().long()
        xR = torch.clamp(xR, 0, W - 1)
        featR_warp = torch.gather(featR, dim=-1, index=xR.expand(-1, C, -1, -1))
        ec = (featL * featR_warp).sum(1, keepdim=True)
        return ec[mask].mean().item()

    if sign_if_known in (+1, -1):
        sign = sign_if_known
    elif auto_sign:
        # sample up to ~4096 valid pixels for quick sign test
        valid_idx = mask.view(-1).nonzero(as_tuple=False).view(-1)
        if valid_idx.numel() > 4096:
            rand_sel = valid_idx[torch.randperm(valid_idx.numel(), device=device)[:4096]]
            m_samp = torch.zeros_like(mask.view(-1), dtype=torch.bool)
            m_samp[rand_sel] = True
            m_samp = m_samp.view_as(mask)
        else:
            m_samp = mask
        # temporarily use sampled mask for EC-SIM comparison
        old_mask = mask
        mask = m_samp
        ec_pos = ec_sim_given_sign(+1)
        ec_neg = ec_sim_given_sign(-1)
        sign = +1 if ec_pos >= ec_neg else -1
        # restore full mask
        mask = old_mask
    else:
        sign = +1  # default

    # 3) Build cost volume and soft-argmin with chosen sign
    cost_vol, disp_pred = _build_cost_vol(featL, featR, max_disp, tau, sign)

    # 4) Metrics
    num_valid = mask.sum().item()
    if num_valid == 0:
        return {"EPE": 0.0, "D1": 0.0, "Recall@1": 0.0, "Recall@2": 0.0, "Recall@5": 0.0, "EC-SIM": 0.0, "sign": sign}

    # EPE
    epe = torch.abs(disp_pred - disp_gt)
    epe_mean = epe[mask].mean().item()

    # D1 (>3 px or >5% of GT)
    thr = torch.maximum(torch.full_like(disp_gt, 3.0), 0.05 * torch.clamp(disp_gt, min=0))
    bad = (torch.abs(disp_pred - disp_gt) > thr)
    d1 = (bad & mask).sum().float() / (mask.sum().float() + 1e-6)

    # EC-SIM with chosen sign
    disp_int = disp_gt.round().long()
    x_coords = torch.arange(W, device=device).view(1, 1, 1, W).repeat(B, 1, H, 1)
    xR = x_coords + sign * disp_int
    xR = torch.clamp(xR, 0, W - 1)
    featR_warp = torch.gather(featR, dim=-1, index=xR.expand(-1, C, -1, -1))
    ec_sim = (featL * featR_warp).sum(1, keepdim=True)
    ec_sim_mean = ec_sim[mask].mean().item()

    # Recall@k
    gt_idx = disp_gt.round().long().clamp(0, max_disp)
    topk_idx = torch.topk(cost_vol, k=5, dim=1).indices  # (B,5,H,W)
    R1 = ((topk_idx[:, :1] == gt_idx).any(1) & mask).float().sum() / (mask.sum().float() + 1e-6)
    R2 = ((topk_idx[:, :2] == gt_idx).any(1) & mask).float().sum() / (mask.sum().float() + 1e-6)
    R5 = ((topk_idx[:, :5] == gt_idx).any(1) & mask).float().sum() / (mask.sum().float() + 1e-6)

    return {
        "EPE": float(epe_mean),
        "D1": float(d1.item()),
        "Recall@1": float(R1.item()),
        "Recall@2": float(R2.item()),
        "Recall@5": float(R5.item()),
        "EC-SIM": float(ec_sim_mean),
        "sign": int(sign),
    }

def evaluate_epipolar_consistency(featL_up, featR_up, disp_gt, thresholds=[1,3,5]):
    """
    Evaluate epipolar consistency using GT disparity.
    
    Args:
        featL_up: (1, C, H, W) left upsampled feature map from AnyUp
        featR_up: (1, C, H, W) right upsampled feature map from AnyUp
        disp_gt:  (1, H, W) ground truth disparity map
        num_samples: number of random valid pixels to test
        thresholds: list of pixel thresholds for recall
    """
    # Prepare data
    featL_up = F.normalize(featL_up, dim=1)
    featR_up = F.normalize(featR_up, dim=1)
    disp = disp_gt.squeeze().cpu().numpy()
    H, W = disp.shape

    # sample valid coordinates
    valid = np.where(disp > 0)
    coords = list(zip(valid[1], valid[0]))  # (x, y)
    np.random.shuffle(coords)
    coords = coords

    errors = []
    recalls = {t: 0 for t in thresholds}

    for (x, y) in coords:
        gt_disp = disp[y, x]
        x_gt = int(round(x - gt_disp))  # GT corresponding pixel in right view
        if x_gt < 0 or x_gt >= W:
            continue

        # Extract left feature (1, C)
        fa = featL_up[0, :, y, x]

        # Sample along epipolar line (same row)
        search_range = 64  # ± search range in pixels
        x_min = max(0, x - search_range)
        x_max = min(W, x + search_range)

        fb_line = featR_up[0, :, y, x_min:x_max].permute(1, 0)  # (range, C)
        sims = torch.matmul(fb_line, fa)  # (range,)
        best_idx = torch.argmax(sims).item()
        x_pred = x_min + best_idx

        # disparity prediction
        disp_pred = x - x_pred
        err = abs(disp_pred - gt_disp)
        errors.append(err)

        for t in thresholds:
            if err <= t:
                recalls[t] += 1

    mean_err = np.mean(errors)
    median_err = np.median(errors)
    recall_results = {t: recalls[t] / len(errors) for t in thresholds}

    return {
        "mean_err": mean_err,
        "median_err": median_err,
        **{f"recall@{t}": v for t, v in recall_results.items()}
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 1, 64, 64, 64
    featL = torch.rand(B, C, H, W)
    # Create perfect 8px disparity: right = left shifted RIGHT by +8
    featR = torch.roll(featL, shifts=+8, dims=-1)
    disp_gt = torch.full((B,1,H,W), 8.0)
    valid_mask = torch.ones_like(disp_gt)

    out = evaluate_epipolar_consistency_new(
        featL, featR, disp_gt, valid_mask,
        max_disp=32, tau=20.0, device='cpu', auto_sign=True
    )
    print(out)  # Expect ~ EPE≈0, D1≈0, Recall@{1,2,5}≈1, EC-SIM≈1, sign=+1