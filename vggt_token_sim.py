import sys
sys.path.append("./vggt")

import numpy as np
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn.functional as F


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


sims = [[] for _ in range(24)]
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        for images, _ in test_loder:
            images = images.to(device)
            images = torch.stack([images[..., :-patch_size], images[..., patch_size:]], dim=1)
            aggregated_tokens_list, patch_start_idx = model.aggregator(images)
            
            for i, a in enumerate(aggregated_tokens_list):
                b, s, n, c = a.shape
                # n_token_wh = int((n - patch_start_idx) ** (1/2))
                tokens = a[..., patch_start_idx:, :]
                
                tokens = tokens.permute(0, 3, 2, 1)

                sim = compute_semantic_similarity(tokens[..., 0], tokens[..., 1])
                sims[i].append(sim)

print([(i, a) for i, a in enumerate(np.array(sims).mean(axis=1))])


# VGGT, delta_x=1
# [(0, 0.8770315914192899), (1, 0.8198645116112625), (2, 0.7988911916913423), (3, 0.7905532698281671), (4, 0.8467383713683383), (5, 0.7690014697140929), (6, 0.758098654494509), (7, 0.7477587999248699), (8, 0.7389269279365385), (9, 0.7318569522525531), (10, 0.7298484816085054), (11, 0.745523635821527), (12, 0.7422187063698856), (13, 0.7897737522717648), (14, 0.8269222048538037), (15, 0.807065166669562), (16, 0.8406469205246437), (17, 0.9290337499435953), (18, 0.9295404455083937), (19, 0.9197582346358756), (20, 0.9090439864185823), (21, 0.925110226007683), (22, 0.9537618825489051), (23, 0.9902856227336736)]
# VGGT, delta_x=2
# [(0, np.float64(0.8300649391415162)), (1, np.float64(0.7538915185481614)), (2, np.float64(0.7274246902912551)), (3, np.float64(0.7181287628812848)), (4, np.float64(0.7951239453071004)), (5, np.float64(0.6924735180229373)), (6, np.float64(0.6790306654335768)), (7, np.float64(0.6665620659373686)), (8, np.float64(0.6565068127424314)), (9, np.float64(0.6498441732585308)), (10, np.float64(0.652317312487274)), (11, np.float64(0.6785101028663806)), (12, np.float64(0.6800347592097435)), (13, np.float64(0.7437689559280023)), (14, np.float64(0.7920820486035707)), (15, np.float64(0.7685950684935886)), (16, np.float64(0.8070020704793833)), (17, np.float64(0.9131539421508851)), (18, np.float64(0.9117292512944177)), (19, np.float64(0.8905046937907544)), (20, np.float64(0.8860658623293315)), (21, np.float64(0.9048269200470686)), (22, np.float64(0.9407650167976037)), (23, np.float64(0.9872641329124358))
# VGGT, delta_x=3
# [(0, np.float64(0.801331842382668)), (1, np.float64(0.7142230573351175)), (2, np.float64(0.6850645398899396)), (3, np.float64(0.675663906178018)), (4, np.float64(0.7651889369104157)), (5, np.float64(0.6481655562967973)), (6, np.float64(0.6334859422899313)), (7, np.float64(0.6200467522906674)), (8, np.float64(0.6096443527827681)), (9, np.float64(0.6037628989355627)), (10, np.float64(0.6096429282196185)), (11, np.float64(0.6421333319543578)), (12, np.float64(0.6466311612100076)), (13, np.float64(0.7197873960201706)), (14, np.float64(0.7727665739729546)), (15, np.float64(0.7459645868561661)), (16, np.float64(0.7869508721677931)), (17, np.float64(0.9026820879361295)), (18, np.float64(0.8990268144005428)), (19, np.float64(0.8686172123353496)), (20, np.float64(0.8701257668056216)), (21, np.float64(0.8904030423785907)), (22, np.float64(0.9311924125172213)), (23, np.float64(0.984803654875629))]