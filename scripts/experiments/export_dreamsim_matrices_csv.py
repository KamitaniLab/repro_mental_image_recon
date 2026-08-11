"""Export the per-subject DreamSim distance matrices as labelled CSVs.

``recon_distance_distribution.py`` stores the matrices in ``matrices_<metric>.npz``
(one 25x25 array per subject) but without axis labels. This writes one CSV per
subject with the stimulus names on both axes, taken from the companion
``distances_<metric>.csv``.

Orientation follows the producing script: ``M[i, j] = dist(recon_i, source_j)``.
Rows are reconstructions, columns are target (source) images, and the diagonal is
the matched pair.

Usage (from the repo root):
    python scripts/experiments/export_dreamsim_matrices_csv.py \
        --summary_dir results/<...>/distance_summary_CORRECTED_dreamsim
"""

import argparse
import csv
import glob
import os
import re
import sys

import numpy as np


def read_stim_ids(csv_path):
    """subject -> ordered stimulus ids, from the per-image CSV."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if "subject" not in cols or "stim_id" not in cols:
            sys.exit(f"unexpected columns in {csv_path}: {cols}")
        order = {}
        for row in reader:
            order.setdefault(row["subject"], []).append(int(row["stim_id"]))
    return order


def read_source_names(source_dir):
    """stim id -> stimulus file stem ('imageryExpStim18_anat_goldfish')."""
    out = {}
    for path in glob.glob(os.path.join(source_dir, "*.tiff")):
        m = re.search(r"imageryExpStim(\d+)", os.path.basename(path))
        if m:
            out[int(m.group(1))] = os.path.splitext(os.path.basename(path))[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary_dir",
        required=True,
        help="directory holding matrices_<metric>.npz and distances_<metric>.csv",
    )
    ap.add_argument("--metric", default="dreamsim")
    ap.add_argument(
        "--source_dir",
        default="data/source",
        help="stimulus dir, used to turn stim ids into readable labels",
    )
    ap.add_argument(
        "--out_dir", default=None, help="default: <summary_dir>/matrices_csv"
    )
    args = ap.parse_args()

    npz_path = os.path.join(args.summary_dir, f"matrices_{args.metric}.npz")
    csv_path = os.path.join(args.summary_dir, f"distances_{args.metric}.csv")
    for p in (npz_path, csv_path):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")

    out_dir = args.out_dir or os.path.join(args.summary_dir, "matrices_csv")
    os.makedirs(out_dir, exist_ok=True)

    mats = np.load(npz_path, allow_pickle=True)
    stim_ids = read_stim_ids(csv_path)
    source_names = read_source_names(args.source_dir)

    for subj in sorted(mats.files):
        M = mats[subj]
        ids = stim_ids.get(subj)
        if ids is None:
            sys.exit(f"no rows for {subj} in {csv_path}")
        if len(ids) != M.shape[0] or M.shape[0] != M.shape[1]:
            sys.exit(f"{subj}: matrix {M.shape} does not match {len(ids)} rows")
        # Fall back to the stimulus id when the source images are not available.
        names = [source_names.get(i, f"Img{i:04d}") for i in ids]

        out = os.path.join(out_dir, f"dreamsim_matrix_{subj}.csv")
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            # Corner cell names the orientation so the file is self-describing.
            w.writerow(["recon\\target"] + names)
            for i, r in enumerate(names):
                w.writerow([r] + [f"{v:.6f}" for v in M[i]])

        diag = np.diag(M)
        off = M[~np.eye(len(ids), dtype=bool)]
        print(f"{subj}: {M.shape[0]}x{M.shape[1]} -> {out}")
        print(
            f"    matched(diag) mean={diag.mean():.4f}  "
            f"mismatched(off-diag) mean={off.mean():.4f}"
        )


if __name__ == "__main__":
    main()
