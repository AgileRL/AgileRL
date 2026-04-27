"""Build the tiny LLM test fixture used by unit tests.

Downloads ``trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`` from HuggingFace
Hub once and saves it under ``tests/assets/tiny_llm/`` so subsequent test
runs can load the model offline via ``from_pretrained(<local_path>)``.

Run when the upstream model is updated, when fixture files are missing, or
when transformers/safetensors compatibility shifts. The output directory
should be committed to the repo (a few MB).

Usage::

    uv run python tests/assets/build_tiny_llm_fixture.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

UPSTREAM_MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
FIXTURE_DIR = Path(__file__).resolve().parent / "tiny_llm"
SIZE_BUDGET_MB = 25


def main() -> None:
    if FIXTURE_DIR.exists():
        shutil.rmtree(FIXTURE_DIR)
    FIXTURE_DIR.mkdir(parents=True)

    print(f"Downloading {UPSTREAM_MODEL_ID} from HF Hub...")
    model = AutoModelForCausalLM.from_pretrained(UPSTREAM_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(UPSTREAM_MODEL_ID)

    print(f"Saving fixture to {FIXTURE_DIR}...")
    model.save_pretrained(FIXTURE_DIR, safe_serialization=True)
    tokenizer.save_pretrained(FIXTURE_DIR)
    # Drop redundant slow-tokenizer files; the fast tokenizer.json is sufficient.
    for legacy in ("vocab.json", "merges.txt"):
        (FIXTURE_DIR / legacy).unlink(missing_ok=True)

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
