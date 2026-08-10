"""Repeated recovery-matrix inversion for error bars on the identification figure.

Same procedure as recovery_matrix_invert.py, but the random RGB-noise sources AND
the SGLD noise are driven by --seed, so varying the seed gives INDEPENDENT reps.
Run it ~10 times with different seeds (one process per (opt_space, seed)); each rep
writes its own source/recovered dir, then evaluate each with recovery_check_eval.py
and aggregate with Fig_recon_and_identification_errorbar.py.

The original recovery_matrix_invert.py is imported and left untouched -- this only
reuses its build_space()/invert_one() and constants.

Example (one rep):
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/experiments/recovery_matrix_invert_reps.py \
      --opt_space clip_vitb32 --no_crop --seed 100 \
      --out_root results/recovery_from_rand_images/rep00
"""
import os
import sys
import argparse

import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import recovery_matrix_invert as rmi  # noqa: E402  build_space, invert_one, constants
import yaml  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt_space", required=True,
                    choices=["clip_vitb16", "clip_vitb32", "clip_rn50", "openclip_laion",
                             "clip_rn50_layer1", "clip_rn50_layer2", "clip_rn50_layer3", "clip_rn50_layer4",
                             "alexnet_conv2", "alexnet_conv5", "alexnet_rand_conv2", "alexnet_rand_conv5",
                             "vgg_clip"])
    ap.add_argument("--n_images", type=int, default=25)
    ap.add_argument("--noise_size", type=int, default=224)
    ap.add_argument("--out_root", required=True,
                    help="rep output root; writes <out_root>/<opt_space>/{source,recovered} "
                         "(mirrors recovery_matrix_invert.py so recovery_check_eval.py works unchanged)")
    ap.add_argument("--n_wo", type=int, default=rmi.N_WO)
    ap.add_argument("--n_lang", type=int, default=rmi.N_LANG)
    ap.add_argument("--no_crop", action="store_true", help="CLIP loss without createCrops augmentation")
    ap.add_argument("--optimizer", choices=["Adam", "AdamW"], default="Adam")
    ap.add_argument("--lr_wo", type=float, default=rmi.LR_WO)
    ap.add_argument("--no_clamp", action="store_true", help="disable clamping VQGAN output to [-1,1]")
    ap.add_argument("--seed", type=int, required=True,
                    help="seed for the noise sources + SGLD noise; vary across reps (e.g. 100,101,...). "
                         "AlexNet-random weights stay fixed (rmi.ALEX_RAND_SEED) so invert/eval match.")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dt_cfg = yaml.safe_load(open("./scripts/config/config_recon.yaml", "rb"))
    taming_dir = dt_cfg["file_path"]["taming_transformer_dir"]
    sys.path.insert(0, taming_dir)
    import model_loading
    cfg = model_loading.load_config(taming_dir + "/logs/vqgan_imagenet_f16_1024/configs/model.yaml", display=False)
    vqgan = model_loading.load_vqgan(cfg, ckpt_path=taming_dir + "/logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt").to(device)
    vqgan.eval()

    loss_fn, target_fn = rmi.build_space(args.opt_space, device, no_crop=args.no_crop)

    # Re-seed AFTER build_space so BOTH the noise sources and the SGLD noise depend on --seed.
    # build_space reseeds torch to ALEX_RAND_SEED for the random-weight nets; that only fixes their
    # weights (already instantiated here) and must not fix the SGLD noise -> reseed torch now.
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
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
        rec, loss = rmi.invert_one(vqgan, loss_fn, target, init_img, device, n_wo=args.n_wo, n_lang=args.n_lang,
                                   optimizer=args.optimizer, lr_wo=args.lr_wo, clamp=not args.no_clamp)
        rec.save(os.path.join(rec_out, f"{k:02d}.png"))
        print(f"[{args.opt_space}|seed{args.seed}][{k+1}/{len(sources)}] final corr={-loss:.4f}")
    print(f"MATRIX_INVERT_REPS_DONE {args.opt_space} seed={args.seed} -> {out_dir}")


if __name__ == "__main__":
    main()
