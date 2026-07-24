import numpy as np

from podcast_dub.models import VerificationResult
from podcast_dub.stages.verify import evaluate_masks


def test_evaluate_masks_returns_typed_failure_with_longest_gap() -> None:
    original = np.array([True, True, True, True])
    dubbed = np.array([True, False, False, False])

    result = evaluate_masks(original, dubbed, tolerance_frames=0, frames_per_second=1.0, max_dead_s=1.0)

    assert result == VerificationResult(coverage=0.25, longest_dead_air_s=3.0, passed=False)
