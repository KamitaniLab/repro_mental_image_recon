"""Compare two reconstruction runs made with the same seed.

Usage: python scripts/experiments/compare_recon_runs.py runA_dir runB_dir
Reports, for every matching .pkl, whether the latent vectors are bit-identical
and (if not) how large the discrepancy is relative to the value scale.
"""

import glob
import os
import pickle
import sys

import numpy as np
from PIL import Image

a_dir, b_dir = sys.argv[1], sys.argv[2]

a_pkls = sorted(glob.glob(os.path.join(a_dir, "**", "*.pkl"), recursive=True))
if not a_pkls:
    sys.exit(f"no .pkl found under {a_dir}")

all_identical = True
for a_path in a_pkls:
    rel = os.path.relpath(a_path, a_dir)
    b_path = os.path.join(b_dir, rel)
    if not os.path.exists(b_path):
        print(f"MISSING in B: {rel}")
        all_identical = False
        continue

    with open(a_path, "rb") as f:
        a = pickle.load(f)
    with open(b_path, "rb") as f:
        b = pickle.load(f)

    va, vb = a["latent_vec"], b["latent_vec"]
    print(f"--- {rel}")
    print(f"    seed A={a.get('seed')}  seed B={b.get('seed')}")
    if np.array_equal(va, vb):
        print(f"    latent_vec: IDENTICAL (bit-exact, shape {va.shape})")
    else:
        all_identical = False
        d = np.abs(va - vb)
        scale = np.abs(va).mean()
        print(
            f"    latent_vec: DIFFERS  max|d|={d.max():.3e}  "
            f"mean|d|={d.mean():.3e}  mean|v|={scale:.3e}  "
            f"rel={d.mean() / scale:.3e}"
        )
        print(f"    corr={np.corrcoef(va.ravel(), vb.ravel())[0, 1]:.10f}")

    # loss traces
    for k in ("loss_vgg_withLangevin_list", "loss_clip_withLangevin_list"):
        la, lb = np.array(a[k]), np.array(b[k])
        tag = (
            "IDENTICAL"
            if np.array_equal(la, lb)
            else f"DIFFERS max|d|={np.abs(la - lb).max():.3e}"
        )
        print(f"    {k}: {tag}")

# reconstructed images
a_jpgs = sorted(glob.glob(os.path.join(a_dir, "**", "*.jpg"), recursive=True))
for a_path in a_jpgs:
    rel = os.path.relpath(a_path, a_dir)
    b_path = os.path.join(b_dir, rel)
    if not os.path.exists(b_path):
        continue
    ia = np.asarray(Image.open(a_path), dtype=np.int16)
    ib = np.asarray(Image.open(b_path), dtype=np.int16)
    d = np.abs(ia - ib)
    status = (
        "IDENTICAL" if d.max() == 0 else f"DIFFERS max={d.max()} mean={d.mean():.4f}"
    )
    print(f"--- {rel}\n    image (0-255): {status}")
    if d.max() != 0:
        all_identical = False

print()
print(
    "RESULT:",
    "fully reproducible (bit-exact)"
    if all_identical
    else "NOT bit-exact -- see the numbers above",
)
