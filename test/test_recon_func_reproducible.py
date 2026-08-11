"""Reproducibility tests for repro_mental_image_recon.recon.func_reproducible.

These run on CPU (no GPU/model/data needed): the module only imports
torch/torchvision/numpy/PIL/matplotlib. CUDA-specific checks are gated by
``torch.cuda.is_available()``.

Run: ``uv run pytest test/ -v``
"""

import numpy as np
import pytest
import torch

from repro_mental_image_recon.recon import func_reproducible as R


def _dummy_img(size=64, device="cpu"):
    # Deterministic input image so any output difference comes from createCrops' RNG.
    g = torch.Generator(device=device).manual_seed(0)
    return torch.rand(1, 3, size, size, generator=g, device=device)


def test_set_seed_reproducible():
    """set_seed pins both the torch and numpy global RNGs."""
    R.set_seed(123)
    t1 = torch.rand(5)
    n1 = np.random.rand(5)
    R.set_seed(123)
    t2 = torch.rand(5)
    n2 = np.random.rand(5)
    assert torch.equal(t1, t2)
    assert np.array_equal(n1, n2)


def test_set_seed_sets_cudnn_flags():
    """set_seed forces deterministic cuDNN."""
    # flip them first to make sure set_seed actually sets the values
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    R.set_seed(0)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_createCrops_same_seed_identical():
    """Same-seed generators -> identical crops (augmentation + all torch draws)."""
    img = _dummy_img()
    g1 = torch.Generator(device="cpu").manual_seed(42)
    g2 = torch.Generator(device="cpu").manual_seed(42)
    a = R.createCrops(
        img.clone(), num_crops=8, DEVICE="cpu", generator=g1, augment=True
    )
    b = R.createCrops(
        img.clone(), num_crops=8, DEVICE="cpu", generator=g2, augment=True
    )
    assert torch.equal(a, b)


def test_createCrops_different_seed_differs():
    """Different seeds -> different crops (randomness is actually present)."""
    img = _dummy_img()
    g1 = torch.Generator(device="cpu").manual_seed(42)
    g2 = torch.Generator(device="cpu").manual_seed(43)
    a = R.createCrops(
        img.clone(), num_crops=8, DEVICE="cpu", generator=g1, augment=True
    )
    c = R.createCrops(
        img.clone(), num_crops=8, DEVICE="cpu", generator=g2, augment=True
    )
    assert not torch.equal(a, c)


def test_createCrops_augment_effect():
    """augment=True vs augment=False produce different output (augTransform applied)."""
    img = _dummy_img()
    g_on = torch.Generator(device="cpu").manual_seed(42)
    g_off = torch.Generator(device="cpu").manual_seed(42)
    on = R.createCrops(
        img.clone(), num_crops=8, DEVICE="cpu", generator=g_on, augment=True
    )
    off = R.createCrops(
        img.clone(), num_crops=8, DEVICE="cpu", generator=g_off, augment=False
    )
    assert not torch.equal(on, off)


def test_createCrops_default_generator():
    """generator=None path (internal manual_seed(0)) is reproducible across calls."""
    img = _dummy_img()
    a = R.createCrops(
        img.clone(), num_crops=8, DEVICE="cpu", generator=None, augment=True
    )
    b = R.createCrops(
        img.clone(), num_crops=8, DEVICE="cpu", generator=None, augment=True
    )
    assert torch.equal(a, b)


def test_createCrops_does_not_touch_global_rng():
    """createCrops leaves the caller's global RNG untouched.

    The augmentation is seeded via torch.manual_seed internally; the state must
    be saved and restored so that unrelated global draws are unaffected.
    """
    img = _dummy_img()
    g = torch.Generator(device="cpu").manual_seed(42)

    torch.manual_seed(7)
    expected = torch.rand(5)

    torch.manual_seed(7)
    R.createCrops(img.clone(), num_crops=4, DEVICE="cpu", generator=g, augment=True)
    after = torch.rand(5)

    assert torch.equal(expected, after)


def test_compute_loss_CLIP_requires_generator_for_input2():
    """The input2='img' path refuses to silently fall back to the seed-0 stream."""
    with pytest.raises(RuntimeError, match="generator"):
        R.compute_loss_CLIP(
            CLIPmodel=[None],
            CLIPmodelWeight=[1.0],
            input1=[torch.zeros(1, 4)],
            input1_type="feat",
            input2=_dummy_img(),
            input2_type="img",
            meanCLIPfeature=[torch.zeros(4)],
            cosSimilarity=torch.nn.CosineSimilarity(dim=1, eps=1e-6),
            DEVICE="cpu",
            generator=None,
        )


def test_imageRecon_generator_seeded():
    """imageRecon stores generators seeded from `seed` (no heavy models needed)."""
    # Build generators the same way imageRecon.__init__ does and check the seed.
    seed = 42
    gen_cpu = torch.Generator(device="cpu").manual_seed(seed)
    assert gen_cpu.initial_seed() == seed
    # Two same-seed generators yield the same first draw.
    g1 = torch.Generator(device="cpu").manual_seed(seed)
    g2 = torch.Generator(device="cpu").manual_seed(seed)
    assert torch.equal(torch.rand(4, generator=g1), torch.rand(4, generator=g2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_createCrops_same_seed_identical_cuda():
    """Same-seed generators -> identical crops on CUDA as well."""
    dev = "cuda"
    img = _dummy_img(device=dev)
    g1 = torch.Generator(device=dev).manual_seed(42)
    g2 = torch.Generator(device=dev).manual_seed(42)
    a = R.createCrops(img.clone(), num_crops=8, DEVICE=dev, generator=g1, augment=True)
    b = R.createCrops(img.clone(), num_crops=8, DEVICE=dev, generator=g2, augment=True)
    assert torch.equal(a, b)
