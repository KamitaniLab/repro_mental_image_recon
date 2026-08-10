# Reanalysis of Mental Image Reconstruction in Koide–Majima et al.(2024)

This repository contains the code for the paper:

**Ken Shirakawa, Yoshihiro Nagano, Misato Tanaka, Fan L. Cheng, Yukiyasu Kamitani**  
*"Apparent and actual evidence in brain-to-image reconstruction: a reanalysis of Koide-Majima et al. (Neural Networks, 2024)"*  
Preprint of an earlier version: https://arxiv.org/abs/2511.07960

The repository collects scripts to re-run the imagery reconstruction analyses and reproduce all figures in the revised manuscript. It builds upon the original implementation in [`nkmjm/mental_img_recon`](https://github.com/nkmjm/mental_img_recon).

## Validated environment
- Ubuntu 20.04.2 LTS (kernel 5.15.0)
- Python 3.12.11
- NVIDIA Driver 535.171.04 (driver CUDA 12.2)
- PyTorch 2.9.0 built against CUDA 12.8 (`cu128` wheels)
- GPU: GeForce RTX 3090 (24GB)

## Quick Start

1. **Install `uv`**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone this repository with submodules**
   ```bash
   git clone --recursive https://github.com/KamitaniLab/repro_mental_image_recon.git
   cd repro_mental_image_recon
   git submodule update --init --recursive
   ```

3. **Set up Python 3.12 environment**
   ```bash
   uv python install 3.12
   uv venv --python 3.12
   source .venv/bin/activate
   uv sync --locked
   ```

4. **Download brain features and model weights**
   ```bash
   uv run bash setup_resources.sh
   ```
   This fetches decoded fMRI features and pretrained VQGAN weights from the original repository.

5. **Obtain the imagery target stimuli**

   They are excluded from this repository due to copyright. Contact
   kamitanilab@gmail.com for `imageryExpStim.zip`, then:
   ```bash
   unzip imageryExpStim.zip -d data/
   ```
   This populates `data/source/` with `imageryExpStim01_*.tiff` … `imageryExpStim26_*.tiff`
   (`imageryExpStim16_fixation.tiff` is not a reconstruction target, so the analyses use the
   remaining 25). Set `IMAGERY_SOURCE_DIR` to keep them elsewhere.

Run every command from the repository root: the scripts resolve `scripts/config/*.yaml`
relative to the working directory.

## Reproducing Figures

The work is organized into eight **analyses**. Each is one pipeline: an experiment script
that writes numbers under `results/`, and figure scripts that turn those numbers into
panels under `assets/`.

| # | Analysis | Figures |
|---|---|---|
| 1 | Reproduction of the original Koide-Majima methods | 2C, A1 |
| 2 | Distance distribution analysis | 2D, 2E |
| 3 | Run-to-run variability analysis | 3A |
| 4 | Circular evaluation analysis | 4A, 4B |
| 5 | Ablation & preference analysis | 5B, 5D, 5E, A2–A4 |
| 6 | SGLD sampling effect analysis | 6B–E, A6 |
| 7 | Hyperparameter sweep analysis | A5 |
| 8 | CPU/GPU determinism analysis | A7 |

Analyses 1 and 3 produce the reconstructions that 2, 5 and 6 consume, so run them first.

### Analysis 1 — reproduction of the original Koide-Majima methods (Fig 2C, A1)

```bash
uv run python scripts/experiments/replicate_original_analysis.py original_all
uv run python scripts/create_figure_assets/Fig2C_assets.py
```

Reconstructions land in `results/rep_recon_image_koide-majima/original_all/{S1,S2,S3}/VC/`;
figures in `assets/fig02/`.

### Analysis 2 — distance distribution analysis (Fig 2D, 2E)

```bash
uv run python scripts/experiments/recon_distance_distribution.py \
    --recon_root results/rep_recon_image_koide-majima --method original_all

uv run python scripts/create_figure_assets/Fig_best_pairs.py \
    --csv results/rep_recon_image_koide-majima/original_all/distance_summary/distances_dreamsim.csv \
    --subject S2
```

`--recon_root` is any reconstruction root holding `<method>/<subject>/VC/`. The summary
directory gets `distances_{metric}.csv`, `matrices_{metric}.npz`, the distribution plots and
`summary_*.txt`; figures go to `assets/fig02/`.

`Fig_best_pairs.py` takes the reconstruction directory from the CSV's own location, so it
always draws the run the distances came from — override with `--recon_dir`, and `--true_dir`
if the stimuli are not in `data/source`. Two more scripts read the same summary directory:
`recon_distance_examples.py` (example pairs) and `export_dreamsim_matrices_csv.py`
(`matrices_{metric}.npz` → one labelled CSV per subject).

### Analysis 3 — run-to-run variability analysis (Fig 3A)

```bash
uv run python scripts/experiments/recon_image_koide-majima_methods_multi_times_no_seed.py original_all
uv run python scripts/create_figure_assets/Fig3A_variability_assets.py
```

Ten unseeded repetitions per stimulus, under
`results/rep_recon_image_koide-majima_recon_variability_no_seed/`. The pickles keep the full
500-step SGLD trajectory (~670 MB each), which is what analysis 6 needs. One PDF per stimulus in
`assets/fig03/`.

### Analysis 4 — circular evaluation analysis (Fig 4A, 4B)

```bash
bash scripts/experiments/run_recovery_reps.sh
uv run python scripts/create_figure_assets/Fig_recon_and_identification_errorbar.py \
    --reps_root results/recovery_from_rand_images --err sd
```

No imagery needed: the targets are RGB noise generated from `--seed`. Ten repetitions ×
four optimization spaces, each a full inversion plus evaluation, under
`results/recovery_from_rand_images/rep{00..09}/<opt_space>/`. Figures in `assets/fig04/`.

### Analysis 5 — ablation & preference analysis (Fig 5B, 5D, 5E, A2–A4)

```bash
for m in original_all AdamOnly_all VGGonly_all wo_SGLD_CLIP_all; do
    uv run python scripts/experiments/replicate_original_analysis.py $m
done
bash scripts/experiments/preference_analysis/run_preference_analysis.sh

uv run python scripts/create_figure_assets/Fig5_ablation_assets.py         # 5B, 5D, 5E
uv run python scripts/create_figure_assets/Fig5_ablation_recon_panels.py   # A2-A4
uv run python scripts/experiments/preference_analysis/preference_stats.py  # 5D/5E statistics
```

The four condition trees and `ref_compare_{2,4}_*.pkl` stay under
`results/rep_recon_image_koide-majima/`, the stats table next to them as
`preference_stats.csv`; figures in `assets/fig05/`.

In the manuscript, A2–A4 carry a final reference row from a separate iCNN implementation
(Wang et al., 2025), which is not part of this repository:
<https://github.com/KamitaniLab/InterSiteNeuralCodeConversion> (archived at
<https://doi.org/10.5281/zenodo.14910040>). `Fig5_ablation_recon_panels.py` omits that row
by default and draws the four ablation conditions only. To include it, generate those
reconstructions separately and pass their directory:

```bash
uv run python scripts/create_figure_assets/Fig5_ablation_recon_panels.py \
    --deeprecon-root <dir with {TH,AM,ES}/VC/recon_image-<stimulus>.tiff>
```

### Analysis 6 — SGLD sampling effect analysis (Fig 6B–E, A6)

Reuses the outputs of analyses 1 and 3; no new reconstruction.

```bash
uv run python scripts/create_figure_assets/Fig6_sgld_effect_assets.py             # 6B
uv run python scripts/create_figure_assets/Fig6_sgld_effect_diagnostic_assets.py  # 6C-E

# A6: one subject per run, since each trajectory pickle is ~670 MB
for s in S1 S2 S3; do
    uv run python scripts/experiments/sgld_effect_summary.py $s
done
uv run python scripts/create_figure_assets/FigA6_sgld_systematic_assets.py
```

Fig 6B contrasts `VC/` (after the 500 SGLD steps) with `VC/wo_lang/` (Adam only) from analysis 1.
Fig 6C–E read analysis 3's trajectory pickles directly. The A6 summary lands in
`results/sgld_effect_summary/original_all/{S1,S2,S3}.npz`; figures in `assets/fig06/`.

### Analysis 7 — hyperparameter sweep analysis (Fig A5)

```bash
bash scripts/experiments/run_oat_search_4gpu.sh              # 22 one-at-a-time conditions
MODE=slice bash scripts/experiments/run_oat_search_4gpu.sh   # the 5x5 lr_a x T plane

uv run python scripts/experiments/oat_dreamsim_matrices.py --root results/oat_sampling_params
uv run python scripts/experiments/oat_dreamsim_matrices.py --root results/lr_a_T_slice

uv run python scripts/create_figure_assets/Fig_oat_composite.py
```

The sweep is the expensive one: 22 conditions × 3 subjects × 25 stimuli. Outputs go to
`results/oat_sampling_params/` and `results/lr_a_T_slice/`, each with `dreamsim_matrices/*.npz`;
figures to `assets/figA5/`. `Fig_oat_dreamsim_matrix.py` and `Fig_oat_slice_summary.py` draw
the individual panels the composite is built from.

### Analysis 8 — CPU/GPU determinism analysis (Fig A7)

```bash
uv run python scripts/experiments/check_determinism.py --recon --device cpu
uv run python scripts/experiments/check_determinism.py --recon --device cuda
uv run python scripts/experiments/determinism_sweep.py

uv run python scripts/create_figure_assets/Fig_determinism_cpu_vs_gpu.py
```

The figure needs both devices. CPU reconstruction is slow — shorten it with `--n-sgd` /
`--n-lang`. Dropping `--recon` runs only the forward/backward comparison, which needs no
stimuli. Results under `results/determinism_check/` and `results/determinism_sweep/`;
figure in `assets/figA7/`.

## Seed-Reproducible Reconstruction

The upstream reconstruction does not seed its random operations, so results differ across
runs. A seeded variant is provided:

```bash
uv run python scripts/experiments/recon_image_koide-majima_methods_multi_times_reproducible.py \
    original_all --seed 42 --subjects S01 --targets 18 --iters 10
```

Each reconstruction derives its own seed as `sha256(base_seed | subject | targetID | iter_n)`,
so runs are reproducible while iterations stay independent and shards can run in parallel.

CPU reconstruction is then bit-identical across runs. GPU reconstruction is not: the backward
pass of bilinear interpolation (`grid_sampler_2d_backward_cuda`) accumulates gradients with
`atomicAdd` in non-deterministic order, and the resulting float32 rounding differences amplify
over optimization steps (~36.9 mean absolute pixel difference after 1500 steps). Analysis 8
reproduces this.

**The figures in the paper were generated with the *unseeded* upstream code.** The seeded
variant is a post-hoc diagnostic tool and was not used to produce the published results.

## Dependencies

All core reconstruction functions (VQGAN initialization, feature loading, optimization) come
from the upstream [`mental_img_recon`](https://github.com/nkmjm/mental_img_recon) submodule,
installed as an editable path dependency so the code that runs is always the code at the
checked-out submodule commit. Path resolution is configured in
`scripts/config/config_recon.yaml`, which handles both the `lib/` layout used here and a
root-level layout for local development.

If `lib/mental_img_recon/` or `lib/taming-transformers/` is empty, the submodules were not
fetched:

```bash
git submodule update --init --recursive
```

## License
This project is licensed under the MIT License.

## Contact
For questions or access to imagery target stimuli: **kamitanilab@gmail.com**
