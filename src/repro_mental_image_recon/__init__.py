"""Reusable analysis code behind the reanalysis of Koide-Majima et al. (2024).

The executable entry points live under ``scripts/``; everything importable by
more than one of them lives here:

``repro_mental_image_recon.recon``
    The reconstruction procedure itself (Adam + SGLD on the VQGAN latent) and
    the feature-inversion machinery used for the circular-evaluation analysis.
``repro_mental_image_recon.figures``
    Path resolution, stimulus/reconstruction loading, panel drawing, and the
    condition taxonomy of the sampling-parameter sweep.
"""

__all__ = ["figures", "recon"]
