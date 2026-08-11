"""Invert the features of random RGB-noise targets in one optimization space.

One run of the circular-evaluation analysis: generate ``--n_images`` noise
sources, invert each one's true feature back to an image with the original
Koide-Majima procedure, and write both to
``<out_root>/<opt_space>/{source,recovered}/``. ``recovery_check_eval.py`` then
identifies the recovered images in every evaluation space.

The procedure and the space definitions live in
``repro_mental_image_recon.recon.recovery``; this script is the CLI around them.
For the repetitions that Figure 4 averages over, use
``recovery_matrix_invert_reps.py``.
"""

import argparse
import os

import numpy as np
import torch
import yaml
from PIL import Image

from repro_mental_image_recon.recon.recovery import (
    LR_WO,
    N_LANG,
    N_WO,
    OPT_SPACES,
    SEED,
    build_space,
    invert_one,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--opt_space", required=True, choices=list(OPT_SPACES))
    ap.add_argument("--n_images", type=int, default=25)
    ap.add_argument("--noise_size", type=int, default=224)
    ap.add_argument("--out_root", default="results/recovery_matrix")
    ap.add_argument("--n_wo", type=int, default=N_WO)
    ap.add_argument("--n_lang", type=int, default=N_LANG)
    ap.add_argument(
        "--no_crop",
        action="store_true",
        help="CLIP loss without createCrops augmentation",
    )
    ap.add_argument("--optimizer", choices=["Adam", "AdamW"], default="Adam")
    ap.add_argument("--lr_wo", type=float, default=LR_WO)
    ap.add_argument(
        "--no_clamp",
        action="store_true",
        help="disable clamping VQGAN output to [-1,1] during optimization",
    )
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open("./scripts/config/config_recon.yaml", "rb") as f:
        dt_cfg = yaml.safe_load(f)
    taming_dir = dt_cfg["file_path"]["taming_transformer_dir"]
    import model_loading

    cfg = model_loading.load_config(
        taming_dir + "/logs/vqgan_imagenet_f16_1024/configs/model.yaml", display=False
    )
    vqgan = model_loading.load_vqgan(
        cfg,
        ckpt_path=taming_dir + "/logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt",
    ).to(device)
    vqgan.eval()

    loss_fn, target_fn = build_space(args.opt_space, device, no_crop=args.no_crop)

    # shared random RGB-noise sources (same seed -> identical across opt_spaces)
    rng = np.random.default_rng(SEED)
    sources = [
        Image.fromarray(
            rng.integers(0, 256, (args.noise_size, args.noise_size, 3), dtype=np.uint8),
            "RGB",
        )
        for _ in range(args.n_images)
    ]

    out_dir = os.path.join(args.out_root, args.opt_space)
    src_out, rec_out = (
        os.path.join(out_dir, "source"),
        os.path.join(out_dir, "recovered"),
    )
    os.makedirs(src_out, exist_ok=True)
    os.makedirs(rec_out, exist_ok=True)
    init_img = Image.fromarray(np.uint8(np.ones([240, 240, 3]) * 128))

    for k, src in enumerate(sources):
        src.save(os.path.join(src_out, f"{k:02d}.png"))
        target = target_fn(src)
        rec, loss = invert_one(
            vqgan,
            loss_fn,
            target,
            init_img,
            device,
            n_wo=args.n_wo,
            n_lang=args.n_lang,
            optimizer=args.optimizer,
            lr_wo=args.lr_wo,
            clamp=not args.no_clamp,
        )
        rec.save(os.path.join(rec_out, f"{k:02d}.png"))
        print(f"[{args.opt_space}][{k + 1}/{len(sources)}] final corr={-loss:.4f}")
    print(f"MATRIX_INVERT_DONE {args.opt_space}")


if __name__ == "__main__":
    main()
