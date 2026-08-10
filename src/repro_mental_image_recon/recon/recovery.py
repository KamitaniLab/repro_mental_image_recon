"""Feature inversion with a *pluggable* optimization space.

The optimization procedure is the original Koide-Majima one (withoutLangevin Adam
+ Langevin SGLD on the VQGAN latent); what varies is the feature space the loss is
computed in. That is what the circular-evaluation analysis needs: invert a target's
true feature in one space, then identify the result in every space, and read the
diagonal (eval space == opt space) as the inflation.

opt_space choices (chosen so each has a matching evaluator in recovery_check_eval):
  clip_vitb16     <-> eval clip_vitb16
  openclip_laion  <-> eval clip_laion
  alexnet_conv2   <-> eval alexnet_conv2
  alexnet_conv5   <-> eval alexnet_conv5

Loss = correlation (centered cosine), no mean-feature subtraction (uniform across
spaces; mu shown to be immaterial in Phase 0). CLIP-type spaces use 32 augmented
crops (as in the original CLIP loss); conv spaces use the single image.

Used by ``scripts/experiments/recovery_matrix_invert.py`` (single run) and
``recovery_matrix_invert_reps.py`` (repetitions with per-rep seeds).
"""

import numpy as np
import recon_func  # createCrops, convert*, set_initInput, get_recImg
import torch

SEED = 42

# Importing this module fixes the global RNG state and pins cuDNN to deterministic
# kernels. Both callers depend on that happening before they build anything, and
# the reps runner deliberately re-seeds afterwards with its own --seed.
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True

# original_all procedure params (config_recon.yaml)
N_WO = 1000  # withoutLangevin (Adam) reps
LR_WO = 0.5
N_LANG = 500  # Langevin (SGLD) reps
LR_GAMMA, LR_A, LR_B, T_LANG = 0.055, 0.00015, 0.15, 1e-6
NUM_CROP = 32

VGG_LAYER_IDX = [2, 7, 16, 25, 34]  # 5 conv blocks (post-ReLU)
CLIP_COEF_VGGCLIP = 0.25  # original_all
ALEX_RAND_SEED = (
    1234  # fixed seed so random-weight AlexNet is identical in invert & eval
)

OPT_SPACES = (
    "clip_vitb16",
    "clip_vitb32",
    "clip_rn50",
    "openclip_laion",
    "clip_rn50_layer1",
    "clip_rn50_layer2",
    "clip_rn50_layer3",
    "clip_rn50_layer4",
    "alexnet_conv2",
    "alexnet_conv5",
    "alexnet_rand_conv2",
    "alexnet_rand_conv5",
    "vgg_clip",
)


def centered_cos_loss(feat, target):
    """-mean correlation (centered cosine) between rows of feat and target(1,D)."""
    f = feat - feat.mean(dim=1, keepdim=True)
    t = target - target.mean(dim=1, keepdim=True)
    return -torch.nn.functional.cosine_similarity(f, t, dim=1, eps=1e-6).mean()


def _imagenet_prep(device):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def build_space(opt_space, device, no_crop=False):
    """Return (loss_fn(vqgan_out, target)->scalar, target_from_pil(pil)->target).

    no_crop=True: CLIP loss uses the single 224 image (no createCrops augmentation),
    matching the earlier vqgan_clip_feature_recovery setup.
    """
    if opt_space in ("clip_vitb16", "clip_vitb32", "clip_rn50", "openclip_laion"):
        if opt_space == "openclip_laion":
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
            )
        else:
            import clip

            name = {
                "clip_vitb16": "ViT-B/16",
                "clip_vitb32": "ViT-B/32",
                "clip_rn50": "RN50",
            }[opt_space]
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
        model = (
            torchvision.models.alexnet(
                weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1
            )
            .eval()
            .to(device)
        )
        prep = _imagenet_prep(device)

        def loss_fn(out, target):
            img = recon_func.convertVQGANoutputIntoVGGinput(out)
            return centered_cos_loss(
                model.features[:sl](img).reshape(1, -1).float(), target
            )

        def target_from_pil(pil):
            with torch.no_grad():
                return (
                    model.features[:sl](
                        prep(pil.convert("RGB")).unsqueeze(0).to(device)
                    )
                    .reshape(1, -1)
                    .float()
                )

        return loss_fn, target_from_pil

    if opt_space == "vgg_clip":
        # original_all: VGG19 (5 conv layers, layer-averaged corr) + CLIP ViT-B/32 (clip_coef 0.25)
        import clip
        import torchvision

        vgg = (
            torchvision.models.vgg19(
                weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1
            )
            .eval()
            .to(device)
        )
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
            vgg_loss = sum(
                centered_cos_loss(vf[i], vgg_t[i]) for i in range(len(VGG_LAYER_IDX))
            ) / len(VGG_LAYER_IDX)
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
        block = getattr(
            model.visual, opt_space.split("clip_rn50_")[1]
        )  # layer1..layer4
        _cap = {}
        block.register_forward_hook(lambda m, i, o: _cap.__setitem__("f", o))

        def extract(x):
            model.encode_image(
                x
            )  # runs full visual forward, hook captures the block output
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
            return centered_cos_loss(
                model.features[:sl](img).reshape(1, -1).float(), target
            )

        def target_from_pil(pil):
            with torch.no_grad():
                return (
                    model.features[:sl](
                        prep(pil.convert("RGB")).unsqueeze(0).to(device)
                    )
                    .reshape(1, -1)
                    .float()
                )

        return loss_fn, target_from_pil

    raise ValueError(opt_space)


def invert_one(
    vqgan,
    loss_fn,
    target,
    init_img,
    device,
    n_wo=N_WO,
    n_lang=N_LANG,
    optimizer="Adam",
    lr_wo=LR_WO,
    clamp=True,
):
    # ---- withoutLangevin (Adam/AdamW) ----
    z = recon_func.set_initInput(init_img, "PIL", vqgan, DEVICE=device)
    op = (
        torch.optim.AdamW([z], lr=lr_wo)
        if optimizer == "AdamW"
        else torch.optim.Adam([z], lr=lr_wo)
    )
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
