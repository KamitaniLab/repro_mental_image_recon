"""Unified preference analysis runner.

This script replaces the ad-hoc variants under ``preference_analysis_dirty`` by
exposing their functionality through a single command-line interface.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np
import torch
import yaml
from PIL import Image

from bdpy.dl.torch.domain import ComposedDomain, image_domain
from bdpy.recon.torch.modules import build_encoder
from bdpy.recon.torch.modules.critic import LayerWiseAverageCritic, MSE
from bdpy.recon.torch.modules.encoder import SimpleEncoder

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "mental_img_recon"))

from recon_utils import get_target_image  # noqa: E402  pylint: disable=wrong-import-position

COMPARISON_CONFIGS = {
    "cand2": {
        "prefix": "compare_2",
        "recon_methods": ["original_all", "wo_SGLD_CLIP_all"],
    },
    "cand4": {
        "prefix": "compare_4",
        "recon_methods": [
            "original_all",
            "AdamOnly_all",
            "VGGonly_all",
            "wo_SGLD_CLIP_all",
        ],
    },
}

SUBJECTS: tuple[str, ...] = ("S1", "S2", "S3")
TARGET_IDS: tuple[int, ...] = tuple(range(25))
SEED = 42


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reconstruction preference analysis with configurable setups.",
    )
    parser.add_argument(
        "--comparison",
        choices=COMPARISON_CONFIGS.keys(),
        default="cand4",
        help="Which reconstruction comparison to evaluate (two or four methods).",
    )
    parser.add_argument(
        "--metric",
        choices=("feature", "dreamsim", "lpips"),
        default="feature",
        help="Similarity metric used to score reconstructions.",
    )
    parser.add_argument(
        "--eval-model",
        nargs="+",
        default=None,
        help="One or more backbone names for the feature metric (ignored otherwise).",
    )
    parser.add_argument(
        "--loss",
        choices=("correlation", "MSE", "SE"),
        default="correlation",
        help="Loss definition for the feature-based metric.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/rep_recon_image_koide-majima"),
        help="Directory that stores reconstruction outputs.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Computation device identifier (defaults to CUDA when available).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


class CorrelationLoss(LayerWiseAverageCritic):
    """Layer-wise average correlation converted to a loss."""

    def compare_layer(
        self,
        feature: torch.Tensor,
        target_feature: torch.Tensor,
        layer_name: str,
    ) -> torch.Tensor:
        cosine = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        feature = feature.reshape(feature.shape[0], -1)
        target_feature = target_feature.reshape(target_feature.shape[0], -1)
        return -cosine(
            feature - feature.mean(dim=1, keepdim=True),
            target_feature - target_feature.mean(dim=1, keepdim=True),
        )


class CLIPEncoder(SimpleEncoder):
    """SimpleEncoder wrapper for CLIP models, with optional output layer."""

    def __init__(self, feature_network, layer_names, domain):
        include_output = "output" in layer_names
        filtered_names = [name for name in layer_names if name != "output"]
        super().__init__(feature_network.visual, filtered_names, domain)
        self._feature_network = feature_network
        self._include_output = include_output

    def encode(self, images: torch.Tensor):
        images = self._domain.receive(images)
        features = self._feature_extractor(images)
        if self._include_output:
            features["output"] = self._feature_network.encode_image(images)
        return features


def load_stimuli(target_image_path: str) -> List[Image.Image]:
    images: List[Image.Image] = []
    for _subject in SUBJECTS:
        for target_id in TARGET_IDS:
            array, _ = get_target_image(target_id, target_image_path)
            images.append(Image.fromarray(array).convert("RGB").resize((224, 224)))
    return images


def _stimulus_label(target_id: int) -> str:
    tid = target_id + 1 if target_id > 14 else target_id
    return f"Img{tid + 1:04d}"


def load_recon_images(result_dir: Path, recon_method: str) -> List[Image.Image]:
    images: List[Image.Image] = []
    for subject in SUBJECTS:
        for target_id in TARGET_IDS:
            image_path = (
                result_dir
                / recon_method
                / subject
                / "VC"
                / f"recon_img_normalized-{_stimulus_label(target_id)}.jpg"
            )
            with Image.open(image_path) as handle:
                images.append(handle.convert("RGB").resize((224, 224)))
    return images


def collect_recon_sets(result_dir: Path, recon_methods: Iterable[str]) -> Dict[str, List[Image.Image]]:
    return {method: load_recon_images(result_dir, method) for method in recon_methods}


def prepare_feature_backend(model_name: str, device: torch.device):
    dtype = torch.float32

    if model_name == "RN50":
        import clip

        model, preprocess = clip.load(model_name, device=device)
        layer_tuples = [
            (name, module)
            for name, module in model.visual.named_modules()
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
        ]
        layer_names = [name for name, _ in layer_tuples if not name.startswith("attnpool")]
        layer_names = [name for name in layer_names if "downsample" not in name] + ["output"]
        image_mean = np.array(preprocess.transforms[-1].mean)
        image_std = np.array(preprocess.transforms[-1].std)
        encoder_domain = ComposedDomain(
            [
                image_domain.StandardizedDomain(
                    center=image_mean,
                    scale=image_std,
                    device=device,
                    dtype=dtype,
                ),
                image_domain.FixedResolutionDomain((224, 224)),
            ]
        )
        model.eval()
        return model, encoder_domain, layer_names

    weights = torch.hub.load("pytorch/vision", "get_model_weights", name=model_name)
    model = torch.hub.load("pytorch/vision", model_name, weights=weights)
    model.to(device).eval()
    preprocess = weights.DEFAULT.transforms()
    image_mean = np.array(preprocess.mean)
    image_std = np.array(preprocess.std)
    encoder_domain = ComposedDomain(
        [
            image_domain.StandardizedDomain(
                center=image_mean,
                scale=image_std,
                device=device,
                dtype=dtype,
            ),
            image_domain.FixedResolutionDomain((224, 224)),
        ]
    )
    layer_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
    ]
    return model, encoder_domain, layer_names


def select_critic(loss_name: str) -> LayerWiseAverageCritic:
    if loss_name == "correlation":
        return CorrelationLoss()
    if loss_name == "SE":
        return MSE(reduction="sum")
    return MSE()


def stack_images(images: List[Image.Image], device: torch.device) -> torch.Tensor:
    batch = torch.cat(
        [
            torch.from_numpy(np.asarray(image, dtype=np.float32))[None]
            for image in images
        ],
        dim=0,
    )
    return batch.to(device)


def evaluate_feature_metric(
    recon_sets: Mapping[str, List[Image.Image]],
    target_images: List[Image.Image],
    model_name: str,
    loss_name: str,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model, encoder_domain, layer_names = prepare_feature_backend(model_name, device)
    pil_domain = image_domain.PILDomainWithExplicitCrop()
    critic = select_critic(loss_name)

    target_tensor = stack_images(target_images, device)
    target_domain = pil_domain.send(target_tensor)

    recon_domains = {
        method: pil_domain.send(stack_images(images, device))
        for method, images in recon_sets.items()
    }

    similarity_matrices: Dict[str, List[np.ndarray]] = {layer: [] for layer in layer_names}

    for layer in layer_names:
        if model_name == "RN50":
            encoder = CLIPEncoder(model, [layer], domain=encoder_domain)
        else:
            encoder = build_encoder(model, [layer], domain=encoder_domain)
        with torch.no_grad():
            target_features = encoder(target_domain)
            for method, recon_domain in recon_domains.items():
                recon_features = encoder(recon_domain)
                loss = critic(recon_features, target_features)
                similarity = -loss.detach().cpu().numpy().reshape(-1)
                similarity_matrices[layer].append(similarity)

    return {
        layer: np.stack(similarities, axis=1)
        for layer, similarities in similarity_matrices.items()
    }


def evaluate_dreamsim_metric(
    recon_sets: Mapping[str, List[Image.Image]],
    target_images: List[Image.Image],
    device: torch.device,
    use_lpips: bool = False,
) -> Dict[str, np.ndarray]:
    from dreamsim import dreamsim

    model, preprocess = dreamsim(pretrained=True, device=device)

    if use_lpips:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)

    def _prep(img: Image.Image) -> torch.Tensor:
        tensor = preprocess(img)
        if isinstance(tensor, dict):
            tensor = tensor["pixel_values"]
        tensor = torch.as_tensor(tensor, dtype=torch.float32)
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        return tensor

    target_tensor = torch.stack([_prep(image) for image in target_images]).to(device)

    similarity_stacks: List[np.ndarray] = []
    for method, images in recon_sets.items():
        recon_tensor = torch.stack([_prep(image) for image in images]).to(device)
        with torch.no_grad():
            if use_lpips:
                distances = torch.stack(
                    [lpips(target_tensor[i][None], recon_tensor[i][None]) for i in range(len(target_tensor))]
                )
            else:
                distances = model(target_tensor, recon_tensor)
        similarity = (1 - distances).detach().cpu().numpy().reshape(-1)
        similarity_stacks.append(similarity)

    return {"output": np.stack(similarity_stacks, axis=1)}


def preference_from_similarity(matrices: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {layer: np.argmax(matrix, axis=1) for layer, matrix in matrices.items()}


def save_results(
    matrices: Mapping[str, np.ndarray],
    prefix: str,
    stem: str,
    results_dir: Path,
) -> None:
    pref_path = results_dir / f"ref_{prefix}_{stem}.pkl"
    sim_path = results_dir / f"ref_{prefix}_{stem}_sim_matrix.pkl"

    with pref_path.open("wb") as handle:
        pickle.dump(preference_from_similarity(matrices), handle)
    with sim_path.open("wb") as handle:
        pickle.dump(matrices, handle)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    config_dir = REPO_ROOT / "scripts" / "config"
    with (config_dir / "demo_params.yaml").open("rb") as handle:
        demo_params = yaml.safe_load(handle)

    target_images = load_stimuli(demo_params["dt_targetimages_path"])
    comparison = COMPARISON_CONFIGS[args.comparison]
    recon_sets = collect_recon_sets(args.results_dir, comparison["recon_methods"])

    if args.metric == "feature":
        eval_models = args.eval_model or ["alexnet", "RN50"]
        for model_name in eval_models:
            matrices = evaluate_feature_metric(
                recon_sets,
                target_images,
                model_name,
                args.loss,
                device,
            )
            stem = f"preference_analysis_results_{model_name}_{args.loss}"
            save_results(matrices, comparison["prefix"], stem, args.results_dir)
        return

    if args.metric == "dreamsim":
        matrices = evaluate_dreamsim_metric(
            recon_sets,
            target_images,
            device,
            use_lpips=False,
        )
        stem = f"preference_analysis_results_dreamsim_{args.loss}"
    else:
        matrices = evaluate_dreamsim_metric(
            recon_sets,
            target_images,
            device,
            use_lpips=True,
        )
        stem = f"preference_analysis_results_lpips_{args.loss}"

    save_results(matrices, comparison["prefix"], stem, args.results_dir)


if __name__ == "__main__":
    main()
