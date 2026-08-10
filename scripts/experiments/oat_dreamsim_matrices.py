"""Full target x reconstruction DreamSim matrices for every OAT sweep condition.

For each condition of the one-at-a-time sampling-parameter sweep, and each
subject, this computes the complete 25x25 DreamSim distance matrix between the
true target images and the reconstructions:

    M[i, j] = DreamSim(target_i, recon_j)

From that both quantities needed to compare conditions fairly fall out:

  * ``matched``  = diag(M)   -- distance to the correct target;
  * ``null``     = off-diag  -- distance to the other 24 targets.

The raw ``matched`` mean alone is misleading: degenerate conditions (e.g. a
near-flat gray output) get a low distance to *everything*, so they look good.
The ``null - matched`` gap is the same DreamSim distance in the same units, but
baselined against what the condition would score by chance.

Results go to one ``.npz`` per condition holding the per-subject matrices, so the
figure script can recompute any summary without touching the images again.

    python scripts/experiments/oat_dreamsim_matrices.py
    python scripts/experiments/oat_dreamsim_matrices.py --root results/lr_a_T_slice
    python scripts/experiments/oat_dreamsim_matrices.py --tags lr_a1_... --overwrite
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load config for mental_img_recon path
with open(str(PROJECT_ROOT / "scripts/config/config_KS_mod.yaml"), "rb") as f:
    dt_cfg = yaml.safe_load(f)
mental_img_recon_dir = dt_cfg["file_path"]["mental_img_recon_dir"]
sys.path.append(mental_img_recon_dir)

from recon_utils import get_target_image  # noqa: E402

DEFAULT_ROOT = PROJECT_ROOT / "results" / "oat_sampling_params"
SUBJECTS = ("S1", "S2", "S3")
N_TARGETS = 25

# With numReps_withLangevin = 0 the pipeline never runs a with-Langevin phase, so
# the wo_lang output *is* that condition's final reconstruction.
WL0_TAG = "lr_a0.00015_lr_b0.15_g0.055_T1e-06_woL1000_wL0_nR1000"


def recon_image_label(target_id: int) -> str:
    """0-based target index -> reconstruction filename stem (Img0016 is absent)."""
    tid = target_id + 1 if target_id > 14 else target_id
    return "Img{:04d}".format(tid + 1)


def recon_dir(root: Path, tag: str, subject: str) -> Path:
    vc = root / tag / subject / "VC"
    return vc / "wo_lang" if tag == WL0_TAG else vc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: <root>/dreamsim_matrices, so pointing --root at "
                             "the lr_a x T slice writes that run's matrices, not the OAT one")
    parser.add_argument("--tags", nargs="*", default=None,
                        help="condition directory names (default: all under --root)")
    parser.add_argument("--subjects", nargs="+", default=list(SUBJECTS),
                        help="subjects present under each condition")
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute conditions whose .npz already exists")
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = args.root / "dreamsim_matrices"
    subjects = tuple(args.subjects)

    from dreamsim import dreamsim

    prm = yaml.safe_load((PROJECT_ROOT / "scripts/config/demo_params.yaml").read_bytes())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, preprocess = dreamsim(pretrained=True, device=device)

    targets = [
        Image.fromarray(get_target_image(i, str(PROJECT_ROOT / prm["dt_targetimages_path"]))[0])
        for i in range(N_TARGETS)
    ]
    target_batch = torch.cat([preprocess(t.convert("RGB")).to(device) for t in targets])

    tags = args.tags or sorted(p.name for p in args.root.iterdir()
                               if p.is_dir() and p.name.startswith("lr_a"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for tag in tags:
        out_path = args.out_dir / f"{tag}.npz"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] exists: {out_path.name}")
            continue

        matrices, pixel_sd = {}, {}
        for subject in subjects:
            rdir = recon_dir(args.root, tag, subject)
            paths = [rdir / f"recon_img_normalized-{recon_image_label(i)}.jpg"
                     for i in range(N_TARGETS)]
            missing = [p for p in paths if not p.exists()]
            if missing:
                print(f"[skip] {tag}/{subject}: {len(missing)} reconstructions missing")
                break
            recons = [Image.open(p).convert("RGB") for p in paths]
            # Track output contrast: a degenerate near-flat image is the usual
            # reason a condition scores a low distance against everything.
            pixel_sd[subject] = np.array(
                [np.asarray(r, dtype=np.float32).std() for r in recons])
            recon_batch = torch.cat([preprocess(r).to(device) for r in recons])
            with torch.no_grad():
                matrices[subject] = np.array([
                    [float(model(target_batch[i:i + 1], recon_batch[j:j + 1]))
                     for j in range(N_TARGETS)]
                    for i in range(N_TARGETS)
                ])
        if len(matrices) != len(subjects):
            continue

        np.savez(
            out_path,
            **{f"M_{s}": matrices[s] for s in subjects},
            **{f"pixel_sd_{s}": pixel_sd[s] for s in subjects},
        )
        matched = np.mean([np.diag(matrices[s]).mean() for s in subjects])
        null = np.mean([matrices[s][~np.eye(N_TARGETS, dtype=bool)].mean() for s in subjects])
        print(f"[done] {tag}  matched={matched:.4f} null={null:.4f} gap={null - matched:+.4f}")

    print(f"matrices in {args.out_dir}")


if __name__ == "__main__":
    main()
