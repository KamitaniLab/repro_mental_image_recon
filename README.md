# Reanalysis of Mental Image Reconstruction in Koide–Majima et al.(2024)

This repository contains the code for the paper:

**Ken Shirakawa, Yoshihiro Nagano, Misato Tanaka, Fan L. Cheng, Yukiyasu Kamitani**  
*"Advancing credibility and transparency in brain-to-image reconstruction research: Reanalysis of Koide-Majima, Nishimoto, and Majima (Neural Networks, 2024)"*  
Preprint: https://arxiv.org/abs/2511.07960

The repository collects scripts to re-run the imagery reconstruction analyses and reproduce all figures in the revised manuscript. It builds upon the original implementation in [`nkmjm/mental_img_recon`](https://github.com/nkmjm/mental_img_recon).

## Validated environment
- Ubuntu 20.04.6 LTS
- Python 3.12.4
- NVIDIA Driver 535.183.01
- CUDA 12.8
- GPU: GeForce RTX 4090 (24GB)

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
   **Note:** Imagery target stimuli are excluded due to copyright. For access, contact shirakawaken0118@gmail.com.

## Reproducing Figures

All analyses are organized by **analysis unit** (pipeline), each producing one or more figures in the manuscript.

| # | Analysis Unit | Figures | Requires imagery | Entry Point(s) | Figure Scripts |
|---|---|---|---|---|---|
| 1 | Reconstruction generation & representativeness | 2C, A1 | ✅ Yes | `replicate_original_analysis.py original_all` | `Fig2C_assets.py` |
| 2 | DreamSim distance distribution & pairs | 2D, 2E | ✅ Yes | `recon_distance_distribution.py --recon_root <run>` | `Fig_best_pairs.py`, `recon_distance_examples.py`, `export_dreamsim_matrices_csv.py` |
| 3 | Run-to-run variability | 3A | ✅ Yes | `recon_image_koide-majima_methods_multi_times_no_seed.py original_all` | `Fig3A_variability_assets.py` |
| 5 | Circular evaluation / recovery matrix | 4A, 4B | ❌ No | `run_recovery_reps.sh` (drives `recovery_matrix_invert_reps.py` + `recovery_check_eval.py`) | `Fig_recon_and_identification_errorbar.py` |
| 6 | SGLD/CLIP ablation | 5B, 5D, 5E, A2–A4 | ✅ Yes | `replicate_original_analysis.py` ×4 conditions → `preference_analysis/run_preference_analysis.sh` | `Fig5_ablation_assets.py`, `Fig5_ablation_recon_panels.py`, `preference_analysis/preference_stats.py` |
| 7 | SGLD sampling effect | 6B–E, A6 | ✅ Yes | Units 1 and 3 outputs → `sgld_effect_summary.py <subject>` | `Fig6_sgld_effect_assets.py`, `Fig6_sgld_effect_diagnostic_assets.py`, `FigA6_sgld_systematic_assets.py` |
| 8 | SGLD hyperparameter sweep | A5 | ✅ Yes | `run_oat_search_4gpu.sh` (`MODE=oat`, then `MODE=slice`) + `oat_dreamsim_matrices.py` for each | `Fig_oat_composite.py`, `Fig_oat_dreamsim_matrix.py`, `Fig_oat_slice_summary.py` |
| 9 | CPU/GPU determinism | A7 | ⚠️ For `--recon` | `check_determinism.py --recon` (once per device) + `determinism_sweep.py` | `Fig_determinism_cpu_vs_gpu.py` |

### Imagery stimuli setup

To run analyses that require imagery (Units 1, 2, 3, 6, 7, 8), you must provide the imagery target stimuli:

1. Contact the authors: **shirakawaken0118@gmail.com** to obtain `imageryExpStim.zip`
2. Extract into the `data/` directory:
   ```bash
   unzip imageryExpStim.zip -d data/
   ```
   This populates `data/source/` with `imageryExpStim01_*.tiff` … `imageryExpStim26_*.tiff`
   (26 files; `imageryExpStim16_fixation.tiff` is not a reconstruction target, so the
   analyses use the remaining 25). Set `IMAGERY_SOURCE_DIR` to override this location.

**Analyses that do NOT require imagery** (self-contained, can run immediately):
- Unit 5 (Circular evaluation): uses noise-generated targets
- Unit 9, the `forward` / `backward` determinism checks: use a fixed synthetic image.
  The `--recon` check and `determinism_sweep.py` do need the decoded features and stimuli.

### Example workflow

**For users WITHOUT imagery** (Unit 5, and the cheap determinism checks):
```bash
# Circular evaluation with noise targets (Unit 5)
# 10 repetitions x 4 optimization spaces; each is a full inversion + evaluation.
bash scripts/experiments/run_recovery_reps.sh
uv run python scripts/create_figure_assets/Fig_recon_and_identification_errorbar.py \
    --reps_root results/recovery_from_rand_images --err sd

# CPU/GPU determinism check (Unit 9)
uv run python scripts/experiments/check_determinism.py --device cpu
uv run python scripts/experiments/check_determinism.py --device cuda
```

**For users WITH imagery** (all units):
```bash
# Generate base reconstructions (Units 1, 6, 7)
uv run python scripts/experiments/replicate_original_analysis.py original_all

# Run variability analysis (Unit 3)
uv run python scripts/experiments/recon_image_koide-majima_methods_multi_times_no_seed.py original_all

# Generate all figure assets
uv run python scripts/create_figure_assets/Fig2C_assets.py
uv run python scripts/create_figure_assets/Fig3A_variability_assets.py
uv run python scripts/create_figure_assets/Fig5_ablation_assets.py
# ... and so on for other figure scripts
```

### Reproducing specific figures

Each figure in the manuscript can be regenerated from the code. Below are the minimal commands for each key figure (assumes imagery stimuli are available where required):

| Figure | Requirements | Generate analysis | Results location | Generate figure | Output |
|--------|---|---|---|---|---|
| **2D, 2E** | Imagery | `uv run python scripts/experiments/recon_distance_distribution.py --recon_root <run> --method original_all` | `<run>/original_all/distance_summary/` (`distances_{metric}.csv`, `matrices_{metric}.npz`, distribution plots, `summary_*.txt`) | `uv run python scripts/create_figure_assets/Fig_best_pairs.py --csv <run>/original_all/distance_summary/distances_dreamsim.csv --subject S2` | `assets/fig02/` |
| **3A** | Imagery | `uv run python scripts/experiments/recon_image_koide-majima_methods_multi_times_no_seed.py original_all` | `results/rep_recon_image_koide-majima_recon_variability_no_seed/` | `uv run python scripts/create_figure_assets/Fig3A_variability_assets.py` | `assets/fig03/` (one PDF per stimulus) |
| **4A, 4B** | (None) | `bash scripts/experiments/run_recovery_reps.sh` | `results/recovery_from_rand_images/rep{00..09}/<opt_space>/` (`source/`, `recovered/`, `recovery_check_identification.pkl`) | `uv run python scripts/create_figure_assets/Fig_recon_and_identification_errorbar.py --reps_root results/recovery_from_rand_images --err sd` | `assets/fig04/` |
| **5B, 5D, 5E** | Imagery | `for m in original_all AdamOnly_all VGGonly_all wo_SGLD_CLIP_all; do uv run python scripts/experiments/replicate_original_analysis.py $m; done && bash scripts/experiments/preference_analysis/run_preference_analysis.sh` | `results/rep_recon_image_koide-majima/` (four condition trees + `ref_compare_{2,4}_*.pkl`) | `uv run python scripts/create_figure_assets/Fig5_ablation_assets.py` and `uv run python scripts/experiments/preference_analysis/preference_stats.py` | `assets/fig05/`; stats table to `results/rep_recon_image_koide-majima/preference_stats.csv` |
| **A2–A4** | Imagery | (same four conditions as above) | `results/rep_recon_image_koide-majima/` | `uv run python scripts/create_figure_assets/Fig5_ablation_recon_panels.py` | `assets/fig05/` |
| **A5** | Imagery | `bash scripts/experiments/run_oat_search_4gpu.sh` then `MODE=slice bash scripts/experiments/run_oat_search_4gpu.sh`, then `oat_dreamsim_matrices.py` once per root | `results/oat_sampling_params/`, `results/lr_a_T_slice/` (each with `dreamsim_matrices/*.npz`) | `uv run python scripts/create_figure_assets/Fig_oat_composite.py` | `assets/figA5/` |
| **6B** | Imagery | (Unit 1 output; `VC/` is post-SGLD and `VC/wo_lang/` pre-SGLD) | `results/rep_recon_image_koide-majima/original_all/` | `uv run python scripts/create_figure_assets/Fig6_sgld_effect_assets.py` | `assets/fig06/` |
| **6C–E** | Imagery | (Unit 3 output; the pickles hold the 500-step SGLD trajectory) | `results/rep_recon_image_koide-majima_recon_variability_no_seed/` | `uv run python scripts/create_figure_assets/Fig6_sgld_effect_diagnostic_assets.py` | `assets/fig06/` |
| **A6** | Imagery | `for s in S1 S2 S3; do uv run python scripts/experiments/sgld_effect_summary.py $s; done` (one subject per run — the trajectory pickles are ~670 MB each) | `results/sgld_effect_summary/original_all/{S1,S2,S3}.npz` | `uv run python scripts/create_figure_assets/FigA6_sgld_systematic_assets.py` | `assets/fig06/`; summary table to `results/sgld_effect_summary/original_all/` |
| **A7** | Imagery | `uv run python scripts/experiments/check_determinism.py --recon --device cpu && uv run python scripts/experiments/check_determinism.py --recon --device cuda && uv run python scripts/experiments/determinism_sweep.py` (the figure needs both devices; the CPU reconstruction is slow — shorten it with `--n-sgd` / `--n-lang`) | `results/determinism_check/`, `results/determinism_sweep/` | `uv run python scripts/create_figure_assets/Fig_determinism_cpu_vs_gpu.py` | `assets/figA7/` |

`<run>` is the reconstruction output root whose `<method>/<subject>/VC/` reconstructions
are being scored — e.g. `results/rep_recon_image_koide-majima`. Numbers (CSV, `.npz`,
text summaries) stay under `results/`; only figures are written to `assets/`.

`Fig_best_pairs.py` takes the reconstruction directory from the CSV's own location, so
the reconstructions it draws are always the ones the distances were computed from. Pass
`--recon_dir` to override, and `--true_dir` (or `IMAGERY_SOURCE_DIR`) to point at the
stimuli if they are not in `data/source`. `export_dreamsim_matrices_csv.py` turns
`matrices_{metric}.npz` into one labelled CSV per subject.

### Figure A2–A4 note

In the manuscript, panels A2–A4 carry a final reference row from a separate iCNN
implementation (Wang et al., 2025), which is not part of this repository:
<https://github.com/KamitaniLab/InterSiteNeuralCodeConversion> (archived at
<https://doi.org/10.5281/zenodo.14910040>). `Fig5_ablation_recon_panels.py` therefore
omits that row by default and draws the four ablation conditions only, which are fully
reproducible here. To include it, generate those reconstructions separately and pass
their directory:

```bash
uv run python scripts/create_figure_assets/Fig5_ablation_recon_panels.py \
    --deeprecon-root <dir with {TH,AM,ES}/VC/recon_image-<stimulus>.tiff>
```

## Seed-Reproducible Reconstruction

The upstream reconstruction does not seed random operations, so results differ across runs. A seeded variant is provided:

```bash
uv run python scripts/experiments/recon_image_koide-majima_methods_multi_times_reproducible.py \
    original_all --seed 42 --subjects S01 --targets 18 --iters 10
```

Each reconstruction gets its own seed derived as `sha256(base_seed | subject | targetID | iter_n)`, ensuring reproducibility while allowing independent variations across iterations and shardable parallel runs.

### CPU vs GPU determinism

CPU reconstruction is fully bit-identical across runs with seeding. GPU reconstruction, however, exhibits non-determinism in the backward pass of bilinear interpolation (`grid_sampler_2d_backward_cuda`), which accumulates gradients with `atomicAdd` in non-deterministic order. This results in float32 rounding differences that amplify over optimization steps (~36.9 mean absolute pixel difference after 1500 steps).

Check both yourself:
```bash
uv run python scripts/experiments/check_determinism.py --device cpu --recon
uv run python scripts/experiments/check_determinism.py --recon  # GPU
```

**Important:** The figures reported in the paper were generated with the *unseeded* upstream code (every run produced different values). The seeded variant above is a post-hoc diagnostic tool and was **not** used to produce the published results.

## Dependency Management

All core reconstruction functions (VQGAN initialization, feature loading, optimization) are imported from the upstream [`mental_img_recon`](https://github.com/nkmjm/mental_img_recon) submodule.  
Path resolution is configured via `scripts/config/config_KS_mod.yaml` and automatically resolves both `lib/` (public clone) and root-level (local development) layouts.

### Submodule setup

After cloning with `--recursive`, ensure submodules are initialized:
```bash
git submodule update --init --recursive
```

This populates `lib/mental_img_recon/` and `lib/taming-transformers/` with the required dependencies.

## License
This project is licensed under the MIT License.

## Contact
For questions or access to imagery target stimuli: **shirakawaken0118@gmail.com**
