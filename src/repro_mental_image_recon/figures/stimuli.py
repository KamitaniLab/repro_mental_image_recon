"""The stimulus subset that Fig 5B and Fig 6B share.

Both panels show the same subject and the same five stimuli -- Fig 6B contrasts
the SGLD and Adam-only outputs of the condition Fig 5B leads with -- so the draw
lives here rather than in either figure script, where the two could drift apart.
"""

from __future__ import annotations

import numpy as np

from repro_mental_image_recon.figures.assets import SOURCE_IMAGE_NAMES

SUBJECT_ID = "S1"

# The panels draw RANDOM_COUNT stimuli from the full set of 25, as the captions
# state. The pool must stay larger than the count: drawing n of n is a
# permutation, and the result is sorted, so the seed would have no effect on
# which stimuli appear.
RANDOM_POOL = SOURCE_IMAGE_NAMES
RANDOM_COUNT = 5
RANDOM_SEED = 42


def select_random_stimuli() -> tuple[str, ...]:
    """The seeded draw of RANDOM_COUNT stimulus names, in canonical order."""
    rng = np.random.default_rng(RANDOM_SEED)
    selection = rng.choice(RANDOM_POOL, size=RANDOM_COUNT, replace=False)
    return tuple(sorted(selection))
