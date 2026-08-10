"""Evaluate the recovery-check reconstructions with multiple encoders.

For each encoder, pairwise identification asks: is recovered_i more similar to
source_i than to source_j (j != i)? Encoders:
  - clip_vitb16  : the inversion encoder (matched -> expected high by construction)
  - clip_laion   : independent-training CLIP (same family)
  - alexnet_conv5: non-CLIP vision net
  - pixel        : raw pixel correlation
  - lpips        : perceptual distance (AlexNet backbone)

A high matched-CLIP score together with low pixel/LPIPS/non-CLIP scores shows the
identification reflects CLIP feature-matching (optimization-evaluation coupling),
not perceptual reconstruction -- no human ground truth needed.
"""
import os
import sys
import glob
import argparse
import pickle

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torchvision
import yaml

# Load config for mental_img_recon path
with open("./scripts/config/config_recon.yaml", "rb") as f:
    dt_cfg = yaml.safe_load(f)
mental_img_recon_dir = dt_cfg["file_path"]["mental_img_recon_dir"]
sys.path.append(mental_img_recon_dir)

import recon_func  # noqa: E402  (createCrops)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

CLIP_MEAN = [0.4814, 0.4578, 0.4082]
CLIP_STD = [0.2686, 0.2613, 0.2757]
NUM_CROP_EVAL = 320           # crop augmentation for CLIP evaluators (recovered side); 320 is very slow (createCrops python loop)
CLIP_EVAL_MODEL = "ViT-B/16"  # switch to "ViT-B/32" to match Koide-Majima

_clip_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                               transforms.Normalize(CLIP_MEAN, CLIP_STD)])


def _center_norm(f):
    f = f - f.mean(dim=1, keepdim=True)
    return f / (f.norm(dim=1, keepdim=True) + 1e-8)


def clip_crop_matrix(enc, recs, srcs, device, ncrop=NUM_CROP_EVAL):
    """M[i,j] = mean over crops of corr(recovered_j crops, source_i single).

    Matches Koide-Majima convention: recon side uses createCrops augmentation,
    target/source side is the single image (no augmentation).
    """
    src_f = []
    with torch.no_grad():
        for s in srcs:
            src_f.append(_center_norm(enc(_clip_tf(s).unsqueeze(0).to(device)).float()))
    src_f = torch.cat(src_f, 0)  # (n, D) centered+normalized

    rec_cf = []
    with torch.no_grad():
        for r in recs:
            crops = recon_func.createCrops(_clip_tf(r).unsqueeze(0).to(device), ncrop, DEVICE=device)
            chunks = [enc(crops[k:k + 64]).float() for k in range(0, ncrop, 64)]
            rec_cf.append(_center_norm(torch.cat(chunks, 0)))  # (ncrop, D)

    n = len(srcs)
    M = np.zeros((n, n))
    for j in range(n):
        sims = (rec_cf[j] @ src_f.T).mean(0).cpu().numpy()  # mean over crops -> (n,)
        M[:, j] = sims
    return M


def load_pairs(out_dir):
    src = sorted(glob.glob(os.path.join(out_dir, "source", "*.png")))
    rec = sorted(glob.glob(os.path.join(out_dir, "recovered", "*.png")))
    assert len(src) == len(rec) and len(src) > 0, (len(src), len(rec))
    return [Image.open(p).convert("RGB") for p in src], [Image.open(p).convert("RGB") for p in rec]


def pearson(a, b):
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    return float(torch.nn.functional.cosine_similarity(a, b, dim=1, eps=1e-8).cpu())


ALEX_RAND_SEED = 1234  # must match recovery_matrix_invert.py so the random AlexNet is identical

OPENAI_CLIP_NAME = {"clip_vitb16": "ViT-B/16", "clip_vitb32": "ViT-B/32", "clip_rn50": "RN50"}


def load_clip_enc(name, device):
    """Return encode_image fn for a CLIP evaluator (crop-augmented matrix path)."""
    if name in OPENAI_CLIP_NAME:
        import clip
        model, _ = clip.load(OPENAI_CLIP_NAME[name], jit=False, device=device)
    else:  # clip_laion
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
    model.eval()

    def enc(x):
        return model.encode_image(x)
    return enc


def build_encoder(name, device):
    """Return sim_fn(recovered_pil, source_pil) -> similarity (higher = more similar)."""
    if name in OPENAI_CLIP_NAME or name == "clip_laion":
        enc = load_clip_enc(name, device)

        def sim(rec, src):
            with torch.no_grad():
                fr = enc(_clip_tf(rec).unsqueeze(0).to(device)).float()
                fs = enc(_clip_tf(src).unsqueeze(0).to(device)).float()
            return pearson(fr, fs)
        return sim

    if name.startswith("clip_rn50_layer"):
        import clip
        model, _ = clip.load("RN50", jit=False, device=device)
        model.eval()
        block = getattr(model.visual, name.split("clip_rn50_")[1])  # layer1..layer4
        _cap = {}
        block.register_forward_hook(lambda m, i, o: _cap.__setitem__("f", o))

        def feat(pil):
            model.encode_image(_clip_tf(pil).unsqueeze(0).to(device))
            return _cap["f"].reshape(1, -1).float()

        def sim(rec, src):
            with torch.no_grad():
                return pearson(feat(rec), feat(src))
        return sim

    if name in ("alexnet_conv2", "alexnet_conv5", "alexnet_rand_conv2", "alexnet_rand_conv5"):
        sl = 5 if name.endswith("conv2") else 12  # post-ReLU conv2 / conv5
        if name.startswith("alexnet_rand"):
            torch.manual_seed(ALEX_RAND_SEED)
            model = torchvision.models.alexnet(weights=None).eval().to(device)
        else:
            model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1).eval().to(device)
        tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        def sim(rec, src):
            with torch.no_grad():
                fr = model.features[:sl](tf(rec).unsqueeze(0).to(device)).reshape(1, -1).float()
                fs = model.features[:sl](tf(src).unsqueeze(0).to(device)).reshape(1, -1).float()
            return pearson(fr, fs)
        return sim

    if name == "dreamsim":
        from dreamsim import dreamsim
        model, preprocess = dreamsim(pretrained=True, device=device)

        def sim(rec, src):
            with torch.no_grad():
                d = model(preprocess(rec).to(device), preprocess(src).to(device))
            return -float(d.cpu())  # similarity = -distance
        return sim

    if name == "pixel":
        tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

        def sim(rec, src):
            fr = tf(rec).reshape(1, -1)
            fs = tf(src).reshape(1, -1)
            return pearson(fr, fs)
        return sim

    if name == "lpips":
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        net = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(device).eval()
        tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

        def sim(rec, src):
            with torch.no_grad():
                d = net(tf(rec).unsqueeze(0).to(device), tf(src).unsqueeze(0).to(device))
            return -float(d.cpu())  # similarity = -distance
        return sim

    raise ValueError(name)


def identification(sim_fn, recs, srcs):
    n = len(srcs)
    M = np.zeros((n, n))
    for i, s in enumerate(srcs):       # row i = source i
        for j, r in enumerate(recs):   # col j = recovered j
            M[i, j] = sim_fn(r, s)
    cr = np.sum(M - np.diag(M)[:, np.newaxis] < 0, axis=1) / (n - 1)
    return float(np.mean(cr)), M


ALL_EVALUATORS = [
    "clip_vitb16", "clip_vitb32", "clip_rn50", "clip_laion",
    "clip_rn50_layer1", "clip_rn50_layer2", "clip_rn50_layer3", "clip_rn50_layer4",
    "alexnet_conv2", "alexnet_conv5", "alexnet_rand_conv2", "alexnet_rand_conv5",
    "lpips", "dreamsim", "pixel",
]
_CLIP_EMB_EVALS = set(OPENAI_CLIP_NAME) | {"clip_laion"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True,
                    help="one inversion run: <dir>/{source,recovered}/*.png. The "
                         "identification pickle is written back into it.")
    ap.add_argument("--clip_aug", action="store_true", help="CLIP eval with crop aug (default: single image, no aug)")
    ap.add_argument("--eval_crops", type=int, default=NUM_CROP_EVAL, help="num crops for CLIP-aug eval")
    ap.add_argument("--evaluators", default=",".join(ALL_EVALUATORS),
                    help="comma-separated evaluator names (default: full set)")
    args = ap.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    srcs, recs = load_pairs(args.out_dir)
    print(f"[info] {len(srcs)} recovered/source pairs")

    results = {}
    for name in [e.strip() for e in args.evaluators.split(",") if e.strip()]:
        if name in _CLIP_EMB_EVALS and args.clip_aug:
            enc = load_clip_enc(name, device)
            M = clip_crop_matrix(enc, recs, srcs, device, ncrop=args.eval_crops)
            cr = np.sum(M - np.diag(M)[:, np.newaxis] < 0, axis=1) / (M.shape[1] - 1)
            acc = float(np.mean(cr))
            note = " (crop-aug recovered, single source)"
        else:
            sim_fn = build_encoder(name, device)
            acc, M = identification(sim_fn, recs, srcs)
            note = " (no aug, single image)" if name in _CLIP_EMB_EVALS else ""
        results[name] = {"acc": acc, "matrix": M}
        print(f"[recovery_check | {name}] identification accuracy = {acc:.4f}{note}")

    with open(os.path.join(args.out_dir, "recovery_check_identification.pkl"), "wb") as f:
        pickle.dump(results, f)
    print("RECOVERY_EVAL_DONE")


if __name__ == "__main__":
    main()
