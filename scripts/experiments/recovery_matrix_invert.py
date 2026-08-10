"""Generalized feature inversion using the ORIGINAL Koide-Majima procedure
(withoutLangevin Adam + Langevin SGLD on the VQGAN latent), with a *pluggable*
optimization space.

For the optimization-space x evaluation-space matrix (recovery check):
targets are RANDOM RGB-noise images; we invert their true feature (in opt_space)
back to an image, then identify with every evaluator. The diagonal (eval space ==
opt space) being highest = inflation from evaluating in the optimization space.

opt_space choices (chosen so each has a matching evaluator in recovery_check_eval):
  clip_vitb16     <-> eval clip_vitb16
  openclip_laion  <-> eval clip_laion
  alexnet_conv2   <-> eval alexnet_conv2
  alexnet_conv5   <-> eval alexnet_conv5

Loss = correlation (centered cosine), no mean-feature subtraction (uniform across
spaces; mu shown to be immaterial in Phase 0). CLIP-type spaces use 32 augmented
crops (as in the original CLIP loss); conv spaces use the single image.
"""
import os
import sys
import argparse

import numpy as np
import torch
from PIL import Image
import yaml

# Load config for mental_img_recon path
with open("./scripts/config/config_KS_mod.yaml", "rb") as f:
    dt_cfg = yaml.safe_load(f)
mental_img_recon_dir = dt_cfg["file_path"]["mental_img_recon_dir"]
sys.path.append(mental_img_recon_dir)

import recon_func  # noqa: E402  createCrops, convert*, set_initInput, get_recImg

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True

# original_all procedure params (config_KS_mod.yaml)
N_WO = 1000          # withoutLangevin (Adam) reps
LR_WO = 0.5
N_LANG = 500         # Langevin (SGLD) reps
LR_GAMMA, LR_A, LR_B, T_LANG = 0.055, 0.00015, 0.15, 1e-6
NUM_CROP = 32


def centered_cos_loss(feat, target):
    """-mean correlation (centered cosine) between rows of feat and target(1,D)."""
    f = feat - feat.mean(dim=1, keepdim=True)
    t = target - target.mean(dim=1, keepdim=True)
    return -torch.nn.functional.cosine_similarity(f, t, dim=1, eps=1e-6).mean()


VGG_LAYER_IDX = [2, 7, 16, 25, 34]  # 5 conv blocks (post-ReLU)
CLIP_COEF_VGGCLIP = 0.25            # original_all
ALEX_RAND_SEED = 1234              # fixed seed so random-weight AlexNet is identical in invert & eval


def _imagenet_prep(device):
    from torchvision import transforms
    return transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def build_space(opt_space, device, no_crop=False):
    """Return (loss_fn(vqgan_out, target)->scalar, target_from_pil(pil)->target).

    no_crop=True: CLIP loss uses the single 224 image (no createCrops augmentation),
    matching the earlier vqgan_clip_feature_recovery setup.
    """
    if opt_space in ("clip_vitb16", "clip_vitb32", "clip_rn50", "openclip_laion"):
        if opt_space == "openclip_laion":
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
        else:
            import clip
            name = {"clip_vitb16": "ViT-B/16", "clip_vitb32": "ViT-B/32", "clip_rn50": "RN50"}[opt_space]
            model, preprocess = clip.load(name, jit=False, device=device)
        model.eval()

        def enc(x):
            return model.encode_image(x)

        def loss_fn(out, target):
            img = recon_func.convertVQGANoutputIntoCLIPinput(out)
            if no_crop:
                return centered_cos_loss(enc(img).float(), target)
            crops = recon_func.createCrops(img, NUM_CROP, DEVICE=device)
            return centered_cos_loss(enc(crops).float(), target)

        def target_from_pil(pil):
            with torch.no_grad():
                return enc(preprocess(pil).unsqueeze(0).to(device)).float()
        return loss_fn, target_from_pil

    if opt_space in ("alexnet_conv2", "alexnet_conv5"):
        import torchvision
        sl = 5 if opt_space == "alexnet_conv2" else 12
        model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1).eval().to(device)
        prep = _imagenet_prep(device)

        def loss_fn(out, target):
            img = recon_func.convertVQGANoutputIntoVGGinput(out)
            return centered_cos_loss(model.features[:sl](img).reshape(1, -1).float(), target)

        def target_from_pil(pil):
            with torch.no_grad():
                return model.features[:sl](prep(pil.convert("RGB")).unsqueeze(0).to(device)).reshape(1, -1).float()
        return loss_fn, target_from_pil

    if opt_space == "vgg_clip":
        # original_all: VGG19 (5 conv layers, layer-averaged corr) + CLIP ViT-B/32 (clip_coef 0.25)
        import torchvision
        import clip
        vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).eval().to(device)
        cmodel, cprep = clip.load("ViT-B/32", jit=False, device=device)
        cmodel.eval()
        prep = _imagenet_prep(device)

        def vgg_feats(img):  # list of flattened activations at VGG_LAYER_IDX
            feats, h = [], img
            for i, layer in enumerate(vgg.features):
                h = layer(h)
                if i in VGG_LAYER_IDX:
                    feats.append(h.reshape(1, -1).float())
            return feats

        def loss_fn(out, target):
            vgg_t, clip_t = target
            vimg = recon_func.convertVQGANoutputIntoVGGinput(out)
            vf = vgg_feats(vimg)
            vgg_loss = sum(centered_cos_loss(vf[i], vgg_t[i])
                           for i in range(len(VGG_LAYER_IDX))) / len(VGG_LAYER_IDX)
            cimg = recon_func.convertVQGANoutputIntoCLIPinput(out)
            crops = recon_func.createCrops(cimg, NUM_CROP, DEVICE=device)
            clip_loss = centered_cos_loss(cmodel.encode_image(crops).float(), clip_t)
            return vgg_loss + CLIP_COEF_VGGCLIP * clip_loss

        def target_from_pil(pil):
            with torch.no_grad():
                vimg = prep(pil.convert("RGB")).unsqueeze(0).to(device)
                vgg_t = vgg_feats(vimg)
                clip_t = cmodel.encode_image(cprep(pil).unsqueeze(0).to(device)).float()
            return (vgg_t, clip_t)
        return loss_fn, target_from_pil

    if opt_space.startswith("clip_rn50_layer"):
        # OpenAI CLIP RN50 visual, intermediate residual-stage output (single image, no crop)
        import clip
        model, preprocess = clip.load("RN50", jit=False, device=device)
        model.eval()
        block = getattr(model.visual, opt_space.split("clip_rn50_")[1])  # layer1..layer4
        _cap = {}
        block.register_forward_hook(lambda m, i, o: _cap.__setitem__("f", o))

        def extract(x):
            model.encode_image(x)  # runs full visual forward, hook captures the block output
            return _cap["f"].reshape(1, -1).float()

        def loss_fn(out, target):
            img = recon_func.convertVQGANoutputIntoCLIPinput(out)
            return centered_cos_loss(extract(img), target)

        def target_from_pil(pil):
            with torch.no_grad():
                return extract(preprocess(pil).unsqueeze(0).to(device))
        return loss_fn, target_from_pil

    if opt_space in ("alexnet_rand_conv2", "alexnet_rand_conv5"):
        # AlexNet with RANDOM (untrained) weights; fixed seed -> identical net in invert & eval
        import torchvision
        sl = 5 if opt_space == "alexnet_rand_conv2" else 12
        torch.manual_seed(ALEX_RAND_SEED)
        model = torchvision.models.alexnet(weights=None).eval().to(device)
        prep = _imagenet_prep(device)

        def loss_fn(out, target):
            img = recon_func.convertVQGANoutputIntoVGGinput(out)
            return centered_cos_loss(model.features[:sl](img).reshape(1, -1).float(), target)

        def target_from_pil(pil):
            with torch.no_grad():
                return model.features[:sl](prep(pil.convert("RGB")).unsqueeze(0).to(device)).reshape(1, -1).float()
        return loss_fn, target_from_pil

    raise ValueError(opt_space)


def invert_one(vqgan, loss_fn, target, init_img, device, n_wo=N_WO, n_lang=N_LANG,
               optimizer="Adam", lr_wo=LR_WO, clamp=True):
    # ---- withoutLangevin (Adam/AdamW) ----
    z = recon_func.set_initInput(init_img, "PIL", vqgan, DEVICE=device)
    op = (torch.optim.AdamW([z], lr=lr_wo) if optimizer == "AdamW"
          else torch.optim.Adam([z], lr=lr_wo))
    for _ in range(n_wo):
        out = vqgan.decode(z)
        if clamp:
            out = out.clamp(-1, 1)
        loss = loss_fn(out, target)
        op.zero_grad()
        loss.backward(retain_graph=True)
        op.step()
    # ---- Langevin (SGLD) ----
    lrs = [LR_A * ((LR_B + t) ** -LR_GAMMA) for t in range(n_lang)]
    z = z.detach().requires_grad_()
    for t in range(n_lang):
        out = vqgan.decode(z)
        if clamp:
            out = out.clamp(-1, 1)
        if z.grad is not None:
            z.grad = None
        loss = loss_fn(out, target)
        loss.backward(retain_graph=True)
        z = (z - z.grad * (lrs[t] / T_LANG)).detach()
        noise = torch.randn_like(z) * np.sqrt(lrs[t])
        z = (z + noise).requires_grad_()
    return recon_func.get_recImg(vqgan, z, DEVICE=device), float(loss.detach().cpu())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt_space", required=True,
                    choices=["clip_vitb16", "clip_vitb32", "clip_rn50", "openclip_laion",
                             "clip_rn50_layer1", "clip_rn50_layer2", "clip_rn50_layer3", "clip_rn50_layer4",
                             "alexnet_conv2", "alexnet_conv5", "alexnet_rand_conv2", "alexnet_rand_conv5",
                             "vgg_clip"])
    ap.add_argument("--n_images", type=int, default=25)
    ap.add_argument("--noise_size", type=int, default=224)
    ap.add_argument("--out_root", default="results/recovery_matrix")
    ap.add_argument("--n_wo", type=int, default=N_WO)
    ap.add_argument("--n_lang", type=int, default=N_LANG)
    ap.add_argument("--no_crop", action="store_true", help="CLIP loss without createCrops augmentation")
    ap.add_argument("--optimizer", choices=["Adam", "AdamW"], default="Adam")
    ap.add_argument("--lr_wo", type=float, default=LR_WO)
    ap.add_argument("--no_clamp", action="store_true", help="disable clamping VQGAN output to [-1,1] during optimization")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dt_cfg = yaml.safe_load(open("./scripts/config/config_KS_mod.yaml", "rb"))
    taming_dir = dt_cfg["file_path"]["taming_transformer_dir"]
    sys.path.insert(0, taming_dir)
    import model_loading
    cfg = model_loading.load_config(taming_dir + "/logs/vqgan_imagenet_f16_1024/configs/model.yaml", display=False)
    vqgan = model_loading.load_vqgan(cfg, ckpt_path=taming_dir + "/logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt").to(device)
    vqgan.eval()

    loss_fn, target_fn = build_space(args.opt_space, device, no_crop=args.no_crop)

    # shared random RGB-noise sources (same seed -> identical across opt_spaces)
    rng = np.random.default_rng(SEED)
    sources = [Image.fromarray(rng.integers(0, 256, (args.noise_size, args.noise_size, 3), dtype=np.uint8), "RGB")
               for _ in range(args.n_images)]

    out_dir = os.path.join(args.out_root, args.opt_space)
    src_out, rec_out = os.path.join(out_dir, "source"), os.path.join(out_dir, "recovered")
    os.makedirs(src_out, exist_ok=True)
    os.makedirs(rec_out, exist_ok=True)
    init_img = Image.fromarray(np.uint8(np.ones([240, 240, 3]) * 128))

    for k, src in enumerate(sources):
        src.save(os.path.join(src_out, f"{k:02d}.png"))
        target = target_fn(src)
        rec, loss = invert_one(vqgan, loss_fn, target, init_img, device, n_wo=args.n_wo, n_lang=args.n_lang,
                               optimizer=args.optimizer, lr_wo=args.lr_wo, clamp=not args.no_clamp)
        rec.save(os.path.join(rec_out, f"{k:02d}.png"))
        print(f"[{args.opt_space}][{k+1}/{len(sources)}] final corr={-loss:.4f}")
    print(f"MATRIX_INVERT_DONE {args.opt_space}")


if __name__ == "__main__":
    main()
