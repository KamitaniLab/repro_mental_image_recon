# Reanalysis of Mental Image Reconstruction in Koide–Majima et al.(2024)

This repository contains the code for the paper:

**Ken Shirakawa, Yoshihiro Nagano, Misato Tanaka, Fan L. Cheng, Yukiyasu Kamitani**  
*"Apparent and actual evidence in brain-to-image reconstruction: a reanalysis of Koide-Majima et al. (Neural Networks, 2024)"*  
Preprint of an earlier version: https://arxiv.org/abs/2511.07960

The repository collects scripts to re-run the imagery reconstruction analyses and reproduce all figures in the revised manuscript. It builds upon the original implementation in
[`nkmjm/mental_img_recon`](https://github.com/nkmjm/mental_img_recon/tree/2eff41d1bcf4814075596238e08a479a2e9f9110)
(the repository carries no tags, so external references here are pinned to commit `2eff41d`).

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
uv run python scripts/create_figure_assets/Fig2C_assets.py   # Fig 2C and Fig A1
```

Reconstructions land in `results/rep_recon_image_koide-majima/original_all/{S1,S2,S3}/VC/`.
One script draws both figures: `assets/fig02/Fig2C_recon_image_random.pdf` (the randomly
selected natural-image targets) and `assets/fig02/FigA1_recon_image_all.pdf` (all 25 targets
× 3 subjects).

### Analysis 2 — distance distribution analysis (Fig 2D, 2E)

```bash
# Fig 2D — matched vs non-target distance distributions, with auc and Mann-Whitney p.
# The analysis script draws this one itself, next to the numbers it computed.
uv run python scripts/experiments/recon_distance_distribution.py \
    --recon_root results/rep_recon_image_koide-majima --method original_all

# Fig 2E — subject 2's target/reconstruction pairs, sorted by matched distance
uv run python scripts/create_figure_assets/Fig2E_best_pairs.py \
    --csv results/rep_recon_image_koide-majima/original_all/distance_summary/distances_dreamsim.csv \
    --subject S2
```

`--recon_root` is any reconstruction root holding `<method>/<subject>/VC/`. Both the numbers
and Fig 2D land in `<run>/original_all/distance_summary/`: `distances_{metric}.csv`,
`matrices_{metric}.npz`, `summary_*.txt`, and `distance_distribution_{tag}.pdf` plus its
per-subject and box-plot variants. Fig 2E goes to `assets/fig02/`.

`Fig2E_best_pairs.py` takes the reconstruction directory from the CSV's own location, so it
always draws the run the distances came from — override with `--recon_dir`, and `--true_dir`
if the stimuli are not in `data/source`.

Two more scripts read the same summary directory. Neither produces a manuscript figure; both
are for inspecting the distances by hand: `recon_distance_examples.py` renders the closest and
farthest matched pairs as a gallery, and `export_dreamsim_matrices_csv.py` turns
`matrices_{metric}.npz` into one labelled CSV per subject.

### Analysis 3 — run-to-run variability analysis (Fig 3A)

```bash
uv run python scripts/experiments/recon_image_koide-majima_methods_multi_times_no_seed.py original_all
uv run python scripts/create_figure_assets/Fig3A_variability_assets.py
```

Ten unseeded repetitions per stimulus, under
`results/rep_recon_image_koide-majima_recon_variability_no_seed/`, and one PDF per stimulus in
`assets/fig03/`.

**Analysis 6 reuses this output**, so do not delete it after drawing Fig 3A. Unlike the
pickles from analysis 1, these keep the full 500-step SGLD trajectory — which is why they run
~670 MB each, and why Fig 6C–E and Fig A6 can only be built from here.

### Analysis 4 — circular evaluation analysis (Fig 4A, 4B)

```bash
bash scripts/experiments/run_recovery_reps.sh
uv run python scripts/create_figure_assets/Fig4_recon_and_identification_errorbar.py \
    --reps_root results/recovery_from_rand_images --err sd   # Fig 4A and Fig 4B
```

No imagery needed: the targets are RGB noise generated from `--seed`. Ten repetitions ×
four optimization spaces, each a full inversion plus evaluation, under
`results/recovery_from_rand_images/rep{00..09}/<opt_space>/`. One script draws both panels of
Figure 4 into `assets/fig04/`: the example noise targets and their feature-matched images
(4A), and the pairwise identification accuracy across evaluation spaces (4B).

### Analysis 5 — ablation & preference analysis (Fig 5B, 5D, 5E, A2–A4)

```bash
for m in original_all AdamOnly_all VGGonly_all wo_SGLD_CLIP_all; do
    uv run python scripts/experiments/replicate_original_analysis.py $m
done
bash scripts/experiments/preference_analysis/run_preference_analysis.sh

uv run python scripts/create_figure_assets/Fig5_ablation_assets.py         # Fig 5B, 5D, 5E
uv run python scripts/create_figure_assets/FigA2A4_ablation_recon_panels.py   # Fig A2, A3, A4
uv run python scripts/experiments/preference_analysis/preference_stats.py  # table, not a figure
```

The four condition trees and `ref_compare_{2,4}_*.pkl` stay under
`results/rep_recon_image_koide-majima/`. `Fig5_ablation_assets.py` writes subject 1's
reconstructions under the four conditions, for five stimuli drawn from all 25 with a seed
(5B), and the four-way and two-way preference results (5D, 5E) to `assets/fig05/`.
`FigA2A4_ablation_recon_panels.py` writes one appendix panel per subject there — A2 for S1, A3
for S2, A4 for S3 — each covering all 25 stimuli. `preference_stats.py` is not a figure: it prints the
binomial tests and confidence intervals behind 5D and 5E, and writes them next to the
pickles as `preference_stats.csv`.

In the manuscript, A2–A4 carry a final reference row from a separate iCNN implementation
(Wang et al., 2025), which is not part of this repository:
<https://github.com/KamitaniLab/InterSiteNeuralCodeConversion/tree/V1.0.0> (archived at
<https://doi.org/10.5281/zenodo.14910040>). `FigA2A4_ablation_recon_panels.py` omits that row
by default and draws the four ablation conditions only. To include it, generate those
reconstructions separately and pass their directory:

```bash
uv run python scripts/create_figure_assets/FigA2A4_ablation_recon_panels.py \
    --deeprecon-root <dir with <subject>/VC/recon_image-<stimulus>.tiff> \
    --deeprecon-subject-dirs <S1 dir> <S2 dir> <S3 dir>
```

`--deeprecon-subject-dirs` is only needed if that tree names its subject directories
differently from `S1`/`S2`/`S3`; the names are matched positionally against `--subjects`.

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

Fig 6B contrasts `VC/` (after the 500 SGLD steps) with `VC/wo_lang/` (Adam only) from
analysis 1, over the same subject and stimuli as Fig 5B, as its caption states. Fig 6C–E read
analysis 3's trajectory pickles directly, and Fig A6 summarizes them over all subjects. The
A6 summary lands in `results/sgld_effect_summary/original_all/{S1,S2,S3}.npz`; all four
figures go to `assets/fig06/`.

### Analysis 7 — hyperparameter sweep analysis (Fig A5)

```bash
uv run python scripts/experiments/oat_search_SGD_SGLD_sampling_params.py --mode oat
uv run python scripts/experiments/oat_search_SGD_SGLD_sampling_params.py --mode slice

uv run python scripts/experiments/oat_dreamsim_matrices.py --root results/oat_sampling_params
uv run python scripts/experiments/oat_dreamsim_matrices.py --root results/lr_a_T_slice

uv run python scripts/create_figure_assets/FigA5_oat_composite.py   # all four panels of Fig A5
```

The sweep is the expensive one. The two modes give 22 + 25 conditions sharing 9, i.e. 38
unique settings, each reconstructing 25 stimuli × 3 subjects (`lr_a` is α in the manuscript).
Outputs go to `results/oat_sampling_params/` and `results/lr_a_T_slice/`, each with
`dreamsim_matrices/*.npz`.

`--dry_run` prints the condition list and writes the manifest without reconstructing anything,
which is the cheap way to see what a mode will do. To spread the work over several GPUs, run
the same command once per device with `--num_shards N --shard_id I` and
`CUDA_VISIBLE_DEVICES` set; `--resume` skips conditions already finished.
`scripts/experiments/run_oat_search_4gpu.sh` is a four-GPU wrapper around exactly that.

`FigA5_oat_composite.py` draws Figure A5 whole into `assets/figA5/`: example reconstructions
per cluster (A), the condition × condition correlation matrix (B), raw matched vs null
distance (C), and the null − matched gap (D). `Fig_oat_dreamsim_matrix.py` and
`Fig_oat_slice_summary.py` are not manuscript figures — they render the sweep and the
lr_a × T plane on their own, in more detail than the composite has room for.

### Analysis 8 — CPU/GPU determinism analysis (Fig A7)

The upstream reconstruction seeds none of its random operations, so no two runs agree. This
analysis runs on `repro_mental_image_recon.recon.func_reproducible`, a seeded variant of
those functions, which is
what makes the question answerable: once the sampling is pinned, whatever difference remains
is arithmetic.

```bash
uv run python scripts/experiments/check_determinism.py --recon --device cpu
uv run python scripts/experiments/check_determinism.py --recon --device cuda
uv run python scripts/experiments/determinism_sweep.py

uv run python scripts/create_figure_assets/FigA7_determinism_cpu_vs_gpu.py
```

The figure needs both devices. CPU reconstruction is slow — shorten it with `--n-sgd` /
`--n-lang`. Dropping `--recon` runs only the forward/backward comparison, which needs no
stimuli. Results under `results/determinism_check/` and `results/determinism_sweep/`;
figure in `assets/figA7/`.

The finding: on CPU two seeded runs are bit-identical. On GPU they are not, because the
backward pass of bilinear interpolation (`grid_sampler_2d_backward_cuda`) accumulates
gradients with `atomicAdd` in non-deterministic order, and the resulting float32 rounding
differences amplify over optimization steps (~36.9 mean absolute pixel difference after 1500
steps).

The seeded variant also has a full reconstruction entry point of its own:

```bash
uv run python scripts/experiments/recon_image_koide-majima_methods_multi_times_reproducible.py \
    original_all --seed 42 --subjects S01 --targets 18 --iters 10
```

Each reconstruction derives its own seed as `sha256(base_seed | subject | targetID | iter_n)`,
so runs are reproducible while iterations stay independent and shards can run in parallel.

**The figures in the paper were generated with the *unseeded* upstream code.** The seeded
variant is a post-hoc diagnostic tool and was not used to produce the published results.

## Dependencies

All core reconstruction functions (VQGAN initialization, feature loading, optimization) come
from the upstream
[`mental_img_recon`](https://github.com/nkmjm/mental_img_recon/tree/2eff41d1bcf4814075596238e08a479a2e9f9110)
submodule, installed as an editable path dependency so the code that runs is always the code
at the checked-out submodule commit. The submodule itself is pinned by commit in `.gitmodules`
and the index, so `git clone --recursive` always fetches the same code. Path resolution is
configured in
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
