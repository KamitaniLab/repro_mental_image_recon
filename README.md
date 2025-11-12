# Reanalysis of Mental Image Reconstruction in Koide–Majima et al.(2024).

This repository collects the scripts we use to re-run the imagery reconstruction analyses introduced by Koide-Majima et al. (2024). This repository contains the code for the paper:
Ken Shirakawa, Yoshihiro Nagano, Misato Tanaka, Fan L. Cheng, Yukiyasu Kamitani,
"Advancing credibility and transparency in brain-to-image reconstruction research: Reanalysis of Koide-Majima, Nishimoto, and Majima (Neural Networks, 2024)" 

The code wraps the original implementation provided in [`nkmjm/mental_img_recon`](https://github.com/nkmjm/mental_img_recon) and adds convenience utilities for batch experiments, figure generation, and exploratory comparisons.

## Quick Start
Follow the steps below to set up the environment and download the required assets.

1. **Install `uv`**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the Koide–Majima reproduction repository and enter it**
   ```bash
   git clone https://github.com/KamitaniLab/repro_mental_image_recon.git
   cd repro_mental_image_recon
   git clone https://github.com/nkmjm/mental_img_recon.git && git -C mental_img_recon checkout 2eff41d
   ```

3. **Prepare Python 3.12 and create a local virtual environment**
   ```bash
   uv python install 3.12
   uv venv --python 3.12
   source .venv/bin/activate
   ```
   Keep the environment activated for the remaining steps.

4. **Install the remaining dependencies with `uv`**
   ```bash
   uv sync
   ```
   This command reads `pyproject.toml` / `uv.lock` and installs everything else into the active `.venv`.

5. **Fetch the brain features and model weights**
   ```bash
   uv run bash setup_resources.sh
   ```
   The helper script orchestrates `download_brain_features.py`, `download_vqgan_model.sh`. and the extraction of imagery stimuli archives. If you want to run the evaluation scripts, make sure the required archives are placed under `data/` before running the script. Note that the imagery stimuli themselves are not included in this repository due to copyright restrictions; if you need access to them for evaluation, please contact us directly.

## Running the Experiments
The main entry points live under `scripts/experiments/`:
- `replicate_original_analysis.py` — mirrors the original Koide–Majima reconstruction pipeline and supports method presets such as `original_all`, `CLIPonly_all`, and `wo_SGLD_CLIP_all`. This script is related to Figrure 2C, 3A, 4B, and 5.
- `recon_image_koide-majima_methods_multi_times_no_seed.py` — convenience wrapper for running multiple reconstructions with different configurations. This script is related to Figure 2D.
- `compare_SGD_SGLD_recon_for_eval_sampling_variance.py` and the `preference_analysis_*` scripts — supplementary analyses exploring reconstruction variability and quality metrics.　This script is related to Figure 4DE.

Run the scripts inside the managed environment with `uv run`, for example:
```bash
uv run python scripts/experiments/replicate_original_analysis.py original_all
```
Outputs are written to subdirectories of `results/` (see the `save_base_dir` logic inside each script). Adjust the configuration files in `scripts/config/` if your data lives in a different location.

## Relationship to `mental_img_recon`
All core reconstruction functions (e.g. VQGAN initialisation, feature loading, and optimisation routines) are imported directly from the upstream [`mental_img_recon`](https://github.com/nkmjm/mental_img_recon) repository. Keeping this dependency intact ensures that the behaviour here matches the original release, but it also inherits the lack of deterministic seeding mentioned above. If you need reproducible results, you will have to modify the upstream utilities to accept explicit random seeds for PyTorch, NumPy, and Python’s `random` module before running these scripts.

## Troubleshooting
- **Different outputs across runs:** this is expected due to the upstream non-deterministic optimisation. If you want to quantify variability, run the reconstruction scripts multiple times and compare the saved outputs under `results/`.
- **Missing data errors:** double-check that the decoded features, mean features, and reference images are placed under the paths referenced in `scripts/config/config_KS_mod.yaml`.
- **Model checkpoint issues:** ensure the VQGAN checkpoints from the original release exist under `external/taming-transformers/logs/...` as referenced in the configuration file.

Feel free to adapt the scripts for your own experiments, but keep the reproducibility warning in mind whenever you interpret the results.
