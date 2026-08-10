"""Condition taxonomy of the sampling-parameter sweep, and its DreamSim matrices.

The sweep writes one ``.npz`` per condition, named after the parameter setting
that produced it (``lr_a0.00015_lr_b0.15_g0.055_T1e-06_woL1000_wL500_nR1000``).
This module turns those names back into parameters, labels them the way the
manuscript does, loads the matrices into matched/null profiles, and clusters
conditions by how they behave -- everything the sweep figures share.

``scripts/create_figure_assets/Fig_oat_dreamsim_matrix.py`` and
``FigA5_oat_composite.py`` both build on it.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist

# The setting used by Koide-Majima et al.; every other condition is described by
# how it differs from this one.
REFERENCE_TAG = "lr_a0.00015_lr_b0.15_g0.055_T1e-06_woL1000_wL500_nR1000"
SUBJECTS = ("S1", "S2", "S3")
N = 25

# Categorical hues, fixed order -- assigned to clusters, never cycled. Chosen to
# stay clear of the red/blue RdBu correlation matrix they sit beside (no pure red
# or blue), and readable as label text on white.
CLUSTER_COLORS = ["#5e35b1", "#e07b39", "#2e9e6f", "#8d6e63", "#d81b60", "#546e7a"]

# --- parameter-type ordering (for the OAT sweep, which varies one knob at a time) ---
_FIELD_RE = re.compile(
    r"lr_a(?P<lr_a>[\d.e+-]+)_lr_b(?P<lr_b>[\d.e+-]+)_g(?P<g>[\d.e+-]+)"
    r"_T(?P<T>[\d.e+-]+)_woL(?P<woL>\d+)_wL(?P<wL>\d+)_nR"
)
REF_FIELDS = {
    "lr_a": 0.00015,
    "lr_b": 0.15,
    "g": 0.055,
    "T": 1e-06,
    "woL": 1000,
    "wL": 500,
}


def parse_fields(tag: str) -> dict[str, float] | None:
    m = _FIELD_RE.search(tag)
    return {k: float(v) for k, v in m.groupdict().items()} if m else None


def classify_condition(tag: str) -> tuple[str, float]:
    """Return (varied-parameter key, its value) relative to the reference setting."""
    f = parse_fields(tag)
    if f is None:
        return "multi", 0.0
    diffs = [
        k
        for k, ref in REF_FIELDS.items()
        if abs(f[k] - ref) > 1e-12 * max(1.0, abs(ref))
    ]
    if not diffs:
        return "reference", 0.0
    if len(diffs) > 1:
        return "multi", 0.0
    return diffs[0], f[diffs[0]]


def short_label(tag: str, reference: str) -> str:
    """Label a condition by the single field in which it differs from the reference."""
    parts, ref_parts = tag.split("_"), reference.split("_")
    if len(parts) != len(ref_parts):
        return tag
    diffs = [p for p, r in zip(parts, ref_parts) if p != r]
    return ", ".join(diffs) if diffs else "reference"


# Display form of the tag fields: the symbols the paper uses. wL / woL are the
# Langevin and the Adam phase respectively (recon.func_mod.withoutLangevin
# optimises with Adam), so they read as step counts, not weights.
TOKEN_DISPLAY = {
    "a": r"$\alpha$",
    "b": "b",
    "g": r"$\gamma$",
    "T": "T",
    "wL": r"$N_\mathrm{SGLD}$",
    "woL": r"$N_\mathrm{Adam}$",
}
_TOKEN_RE = re.compile(r"^(woL|wL|a|b|g|T)([\d.e+-]+)$")
REFERENCE_DISPLAY = "KM"


def _format_value(text: str) -> str:
    """3 significant digits (0.115533 -> 0.116), but whole numbers stay plain so
    the step counts read as 1500 rather than 1.5e+03."""
    value = float(text)
    return f"{value:.0f}" if value == int(value) else f"{value:.3g}"


def pretty_label(tag: str, reference: str) -> str:
    """short_label's diff tokens rendered with the paper's symbols.

    ``a0.01`` -> ``alpha = 0.01``, ``b0.115533`` -> ``b = 0.116``,
    ``woL0`` -> ``N_Adam = 0``; the reference condition becomes ``KM``.
    """
    raw = short_label(tag, reference)
    if raw == "reference":
        return REFERENCE_DISPLAY
    out = []
    for token in raw.split(", "):
        m = _TOKEN_RE.match(token)
        out.append(f"{TOKEN_DISPLAY[m[1]]} = {_format_value(m[2])}" if m else token)
    return ", ".join(out)


def load_conditions(matrix_dir: Path) -> dict[str, dict]:
    """Load every condition .npz into matched / null / pixel-SD summaries."""
    off_mask = ~np.eye(N, dtype=bool)
    out = {}
    for path in sorted(matrix_dir.glob("*.npz")):
        with np.load(path) as handle:
            mats = {s: handle[f"M_{s}"] for s in SUBJECTS}
            sds = np.concatenate([handle[f"pixel_sd_{s}"] for s in SUBJECTS])
        out[path.stem] = {
            # 75-value profile: the matched distance of every reconstruction.
            "matched": np.concatenate([np.diag(mats[s]) for s in SUBJECTS]),
            "null": np.concatenate(
                [mats[s][off_mask].reshape(N, N - 1).mean(axis=1) for s in SUBJECTS]
            ),
            "pixel_sd": sds,
        }
    if not out:
        raise FileNotFoundError(
            f"no .npz in {matrix_dir}\n"
            "Run: python scripts/experiments/oat_dreamsim_matrices.py"
        )
    return out


def cluster_order(profiles: np.ndarray, n_clusters: int):
    """Cluster conditions by the *pattern* of their profile, not its offset."""
    link = linkage(pdist(profiles, metric="correlation"), method="average")
    order = dendrogram(link, no_plot=True)["leaves"]
    return order, fcluster(link, t=n_clusters, criterion="maxclust")
