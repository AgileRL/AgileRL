"""Build the Minari test fixture used by tests/test_utils/test_minari_utils.py.

Downloads ``D4RL/door/human-v2`` from the Farama/HuggingFace registry once and
saves it under ``tests/assets/minari_cache/`` so subsequent test runs can load
the dataset offline (``HF_HUB_OFFLINE=1`` is set globally in conftest.py).

The download writes ``.cache/huggingface/`` metadata alongside the dataset; we
prune it because Minari's offline loader doesn't need it and it inflates the
fixture size.

Run when the upstream dataset is updated, when fixture files are missing, or
when Minari format changes. The output directory should be committed to the
repo (a few MB).

Usage::

    uv run python tests/assets/build_minari_fixture.py
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

DATASET_ID = "D4RL/door/human-v2"
FIXTURE_DIR = Path(__file__).resolve().parent / "minari_cache"
SIZE_BUDGET_MB = 10


def main() -> None:
    if FIXTURE_DIR.exists():
        shutil.rmtree(FIXTURE_DIR)
    FIXTURE_DIR.mkdir(parents=True)

    os.environ["MINARI_DATASETS_PATH"] = str(FIXTURE_DIR)
    # Allow network for this build only; HF_HUB_OFFLINE is on globally for tests.
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    import minari

    print(f"Downloading {DATASET_ID} to {FIXTURE_DIR}...")
    minari.download_dataset(DATASET_ID)

    # Prune HF download metadata — not needed for offline loading.
    for cache_dir in FIXTURE_DIR.rglob(".cache"):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)

    total_bytes = sum(p.stat().st_size for p in FIXTURE_DIR.rglob("*") if p.is_file())
    total_mb = total_bytes / (1024 * 1024)
    print(f"Fixture size: {total_mb:.2f} MB")

    if total_mb > SIZE_BUDGET_MB:
        msg = (
            f"Fixture exceeds {SIZE_BUDGET_MB} MB budget ({total_mb:.2f} MB). "
            "Aborting — investigate before committing."
        )
        raise RuntimeError(msg)

    print("Done. Commit the fixture directory:")
    print(f"  git add {FIXTURE_DIR.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
