# Reanalysis of Mental Image Reconstruction in Koide–Majima et al.(2024).

This repository collects the scripts we use to re-run the imagery reconstruction analyses introduced by Koide-Majima et al. (2024). 

This repository contains the code for the paper:
Ken Shirakawa, Yoshihiro Nagano, Misato Tanaka, Fan L. Cheng, Yukiyasu Kamitani,
"Advancing credibility and transparency in brain-to-image reconstruction research: Reanalysis of Koide-Majima, Nishimoto, and Majima (Neural Networks, 2024)" 
Preprint: https://arxiv.org/abs/2511.07960

This repository builds upon the original implementation provided in [`nkmjm/mental_img_recon`](https://github.com/nkmjm/mental_img_recon) and includes additional scripts for a systematic reanalysis that verifies and quantifies their reported findings.

## Validated environment
- Ubuntu 20.04.6 LTS
- Python 3.12.4
- NVIDIA Driver 535.183.01
- CUDA 12.8
- GPU: GeForce RTX 4090 (24GB)


## Quick Start
Follow the steps below to set up the environment and download the required data.

1. **Install `uv`**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone this repository and enter it**
   ```bash
   git clone --recursive https://github.com/KamitaniLab/repro_mental_image_recon.git
   cd repro_mental_image_recon
   ```
   Be sure to include the `--recursive` option so that submodules are properly cloned.

3. **Prepare Python 3.12 and create a local virtual environment**
   ```bash
   uv python install 3.12
   uv venv --python 3.12
   source .venv/bin/activate
   ```
   Keep the environment activated for the remaining steps.

4. **Install the remaining dependencies with `uv`**
   ```bash
   uv sync --locked
   ```
   This command reads `pyproject.toml` / `uv.lock` and installs everything else into the active `.venv`.

5. **Fetch the brain features and model weights**
   ```bash
   uv run bash setup_resources.sh
   ```
   This helper script runs download_brain_features.py, download_vqgan_model.sh, and also extracts imagery stimulus archives if available. These data follow the original repository (https://github.com/nkmjm/mental_img_recon) and their Colab demo (https://colab.research.google.com/drive/1gaMoae0ntiT94-rQUMymkZboNc-imTzl?usp=drive_link), including decoded features and pretrained VQGAN weights. 
   
   Note that the imagery target stimuli themselves are not included in this repository due to copyright restrictions; if you need access to them for evaluation, please contact us directly. 

## Running the Experiments
The main entry points live under `scripts/experiments/`:
- `replicate_original_analysis.py` — mirrors the original Koide–Majima reconstruction pipeline and supports condition presets such as `original_all`, `CLIPonly_all`, and `wo_SGLD_CLIP_all`. This script is related to Figure 2C, 3A, 4B, and 5.

Usage:
```bash
# at repro_mental_image_recon dir 
uv run python scripts/experiments/replicate_original_analysis.py original_all
```

- `recon_image_koide-majima_methods_multi_times_no_seed.py` — running multiple reconstructions with different configurations. This script is related to Figure 2D.
- `compare_SGD_SGLD_recon_for_eval_sampling_variance.py` and the `run_preference_analysis.sh` scripts — supplementary analyses exploring reconstruction variability and quality metrics.　These scripts are related to Figures 4D and 4E.
- The replicating figures can be obtained from scripts in `create_figure_assets` directory.


## Notes on Dependency and Reproducibility
All core reconstruction functions (e.g. VQGAN initialization, feature loading, and optimization routines) are imported directly from the upstream [`mental_img_recon`](https://github.com/nkmjm/mental_img_recon) repository. Keeping this dependency intact ensures compatibility with the original behavior of the original release, but it also inherits the lack of deterministic seeding mentioned above. Different outputs across runs are expected due to the upstream non-deterministic optimisation.

## Contact
If you would like to use imagery target stimuli or have any questions, please contact us:
shirakawaken0118@gmail.com

## License
This project is licensed under the MIT License.
