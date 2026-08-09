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
| 2 | DreamSim distance distribution & pairs | 2D, 2E | ✅ Yes | `recon_distance_distribution.py` | `Fig_best_pairs.py`, `recon_distance_examples.py` |
| 3 | Run-to-run variability | 3A | ✅ Yes | `recon_image_koide-majima_methods_multi_times_no_seed.py` | `Fig3A_variability_assets.py` |
| 4 | Published example comparison | 3B | ✅ Yes | (none — CC BY 4.0 adoption from original paper) | (none) |
| 5 | Circular evaluation / recovery matrix | 4A, 4B | ❌ No | `recovery_matrix_invert_reps.py` + `recovery_check_eval.py` | `Fig_recon_and_identification_errorbar.py` |
| 6 | SGLD/CLIP ablation | 5A–E, A2–A4 | ✅ Yes | `compare_SGD_SGLD_recon_for_eval_sampling_variance.py` + `run_preference_analysis.py` | `Fig5_ablation_assets.py`, `Fig5_ablation_recon_panels.py`, `scripts/experiments/preference_analysis/preference_stats.py` |
| 7 | SGLD sampling effect | 6A–E, A6 | ✅ Yes | `replicate_original_analysis.py original_all` → `sgld_effect_summary.py` | `Fig6_sgld_effect_assets.py`, `Fig6_sgld_effect_diagnostic_assets.py`, `Fig6A_sgld_systematic_assets.py` |
| 8 | SGLD hyperparameter sweep (OAT) | A5 | ❌ No | `oat_search_SGD_SGLD_sampling_params.py` + `oat_dreamsim_matrices.py` | `Fig_oat_composite.py`, `Fig_oat_dreamsim_matrix.py`, `Fig_oat_slice_summary.py` |
| 9 | CPU/GPU determinism | A7 | ❌ No | `check_determinism.py` + `determinism_sweep.py` | `Fig_determinism_cpu_vs_gpu.py` |

### Imagery stimuli setup

To run analyses that require imagery (Units 1–4, 6–7), you must provide the imagery target stimuli:

1. Contact the authors: **shirakawaken0118@gmail.com** to obtain `imageryExpStim.zip`
2. Extract into the `data/` directory:
   ```bash
   unzip imageryExpStim.zip -d data/
   ```
   This populates `data/ImageryDeeprecon/source/` with the 25 target images.

**Analyses that do NOT require imagery** (self-contained, can run immediately):
- Unit 5 (Circular evaluation): uses noise-generated targets
- Unit 8 (OAT hyperparameter sweep): uses noise-generated targets  
- Unit 9 (CPU/GPU determinism): uses noise-generated targets

### Example workflow

**For users WITHOUT imagery** (Units 5, 8, 9 only):
```bash
# Circular evaluation with noise targets (Unit 5)
uv run python scripts/experiments/recovery_matrix_invert_reps.py
uv run python scripts/experiments/recovery_check_eval.py
uv run python scripts/create_figure_assets/Fig_recon_and_identification_errorbar.py

# OAT hyperparameter sweep (Unit 8)
uv run python scripts/experiments/oat_search_SGD_SGLD_sampling_params.py
uv run python scripts/experiments/oat_dreamsim_matrices.py

# CPU/GPU determinism check (Unit 9)
uv run python scripts/experiments/check_determinism.py --device cpu
uv run python scripts/experiments/check_determinism.py --device cuda
```

**For users WITH imagery** (all units):
```bash
# Generate base reconstructions (Units 1, 6, 7)
uv run python scripts/experiments/replicate_original_analysis.py original_all

# Run variability analysis (Unit 3)
uv run python scripts/experiments/recon_image_koide-majima_methods_multi_times_no_seed.py

# Generate all figure assets
uv run python scripts/create_figure_assets/Fig2C_assets.py
uv run python scripts/create_figure_assets/Fig3A_variability_assets.py
uv run python scripts/create_figure_assets/Fig5_ablation_assets.py
# ... and so on for other figure scripts
```

### Figure A2–A4 note

Panels A2–A4 include a reference row showing results from an unpublished "revised iCNN" implementation (Tanaka et al., 2024; Nagano et al., in preparation). This variant is **not** included in this repository — those reference rows are external image assets. The ablation results for the main conditions (SGLD+CLIP, No SGLD+CLIP, SGLD+No CLIP, No SGLD+No CLIP) are fully reproducible via Unit 6.

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
