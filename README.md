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

### Seeded reconstruction

The upstream reconstruction draws every random number from the *global* torch / NumPy
RNGs, which the public demo never seeds, so the result changes on every run.
`scripts/experiments/recon_func_reproducible.py` is a seeded variant that leaves the
upstream `recon_func.py` untouched: every draw goes through one explicit
`torch.Generator` instead of the global RNG.

| Randomness in upstream `recon_func.py` | In the seeded variant |
|---|---|
| crop size `torch.normal`, crop offsets `torch.randint` | `generator=` |
| per-crop noise `torch.randn_like`, its scale `torch.rand` | `generator=` |
| Langevin noise `np.random.normal` (NumPy!) | `torch.randn(generator=)` |
| `RandomHorizontalFlip` / `RandomAffine` | sub-seed derived from the generator; the global RNG state is saved and restored around it |

Use the dedicated script (`..._no_seed.py` is deliberately left as the unseeded
variability experiment):

```bash
uv run python scripts/experiments/recon_image_koide-majima_methods_multi_times_reproducible.py \
    original_all --seed 42 --subjects S01 --targets 18 --iters 10
```

Each reconstruction gets its own seed, derived as
`sha256(base_seed | subject | targetID | iter_n)`. Because it depends on the indices
rather than on the loop order, the repeats still differ from one another while each one
stays reproducible on its own — and a run can be sharded across GPUs or restarted
partway and still produce identical results. The seed used is stored in the output
`.pkl`.

### What `--seed` does and does not guarantee

Verify it yourself — this runs each check twice in two separate processes and compares:

```bash
uv run python scripts/experiments/check_determinism.py --device cpu --recon
uv run python scripts/experiments/check_determinism.py --recon           # GPU
```

Measured on one RTX 3090 (seed 42 → `2956797496`, S01/target 18, Adam 1000 + SGLD 500):

| | CPU | GPU (CUDA) |
|---|---|---|
| `createCrops` forward | identical (sha256) | identical (sha256) |
| `createCrops` backward | identical, `max\|d\|=0` | **differs**, `max\|d\|=3.8e-06` |
| final image, two runs | **bit-identical** | differs, `mean\|d\|=36.9` |
| final latent, two runs | corr = 1.0 | corr = 0.324 |

So the *sampling* is fully fixed on both devices, but bit-identical *output* holds only
on CPU. On CUDA, `grid_sampler_2d_backward_cuda` (and the bilinear `interpolate`
backward) accumulate gradients with `atomicAdd`, whose order is not fixed, and PyTorch
has no deterministic implementation for them — `cudnn.deterministic` covers only
convolutions. The resulting float32 rounding difference is amplified by the update rule
(`T=1e-06` multiplies the gradient by ~166) and the VQGAN/CLIP non-linearities: the two
runs fall to corr 0.50 by step 200 and saturate at 0.33 by step 1000.

`scripts/create_figure_assets/Fig_determinism_cpu_vs_gpu.py` renders this comparison.

> **Note on the results reported in the paper.** The analyses in the paper reuse the
> existing upstream code, which has no seed handling, so every run produced different
> values. The seeded variant described here was written afterwards and is *not* what
> produced the published figures. Even with the seed fixed, we have confirmed that
> running on a GPU still makes the values drift between runs; on CPU the two runs match
> exactly.

## Contact
If you would like to use imagery target stimuli or have any questions, please contact us:
shirakawaken0118@gmail.com

## License
This project is licensed under the MIT License.
