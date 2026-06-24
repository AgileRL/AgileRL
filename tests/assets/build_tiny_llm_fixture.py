"""Build the tiny LLM test fixture used by unit tests.

Constructs a randomly-initialised Qwen2 model with the upstream
``trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`` tokenizer and saves it
under ``tests/assets/tiny_llm/`` so subsequent test runs can load the model
offline via ``from_pretrained(<local_path>)``.

Why we don't just download the upstream model: it ships with
``hidden_size=8`` / ``num_attention_heads=4``, giving ``head_dim=2``. vLLM
imposes a per-backend lower bound on ``head_dim`` that the upstream model
violates:

- **GPU FlexAttention backend** (used on GPUs without FA2, i.e. compute
  capability < 8) requires ``head_dim >= 16``.
- **CPU backend** (`_PagedAttention` in ``vllm/v1/attention/backends/cpu_attn.py``,
  used on macOS / Linux-without-CUDA) only accepts head sizes from a fixed
  whitelist: ``{32, 64, 80, 96, 112, 128, 192, 256}``. A ``head_dim`` outside
  this set fails at runtime with ``RuntimeError: Unsupported head size: N``
  during ``paged_attention_v1``.

We construct our own Qwen2 with ``head_dim=32`` (the smallest value satisfying
both backends) so the same fixture is usable across CI matrices and local
macOS development.

Run when transformers/safetensors compatibility shifts, or when the model
config needs adjusting. The output directory should be committed to the repo
(<25 MB).

Usage::

    uv run python tests/assets/build_tiny_llm_fixture.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

UPSTREAM_TOKENIZER_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
FIXTURE_DIR = Path(__file__).resolve().parent / "tiny_llm"
SIZE_BUDGET_MB = 25
# transformers >=4.57.3 flags any local non-mistral tokenizer with vocab >100k
# as a "broken Mistral regex" unless the saved transformers_version is <=4.57.2.
# Pin the saved version so AutoTokenizer doesn't emit the false-positive warning.
PINNED_TRANSFORMERS_VERSION = "4.57.2"

# head_dim = hidden_size // num_attention_heads must satisfy BOTH backends:
#   - vLLM GPU FlexAttention (no-FA2 GPUs):   head_dim >= 16
#   - vLLM CPU PagedAttention (macOS, etc.):  head_dim in
#     {32, 64, 80, 96, 112, 128, 192, 256}
# 32 is the smallest head_dim satisfying both. To stay under the 25 MB fixture
# budget (tokenizer alone is ~11 MB, embeddings dominate model weights at this
# scale because tie_word_embeddings shares them with lm_head), we keep
# hidden_size=32 and use a single head so head_dim = 32 / 1 = 32. The single-
# head config still exercises the full Qwen2 forward/backward path; head count
# is an implementation detail that the LLM-level tests (DPO/GRPO/SFT/REINFORCE)
# don't probe.
HIDDEN_SIZE = 32
NUM_ATTENTION_HEADS = 1  # head_dim = 32
NUM_KEY_VALUE_HEADS = 1
NUM_HIDDEN_LAYERS = 2
INTERMEDIATE_SIZE = 64
SEED = 0


def main() -> None:
    if FIXTURE_DIR.exists():
        shutil.rmtree(FIXTURE_DIR)
    FIXTURE_DIR.mkdir(parents=True)

    print(f"Loading tokenizer from {UPSTREAM_TOKENIZER_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(UPSTREAM_TOKENIZER_ID)

    print("Constructing tiny Qwen2 model from scratch...")
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_key_value_heads=NUM_KEY_VALUE_HEADS,
        max_position_embeddings=32768,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        torch_dtype="float32",
        bos_token_id=151643,
        eos_token_id=151645,
    )
    torch.manual_seed(SEED)
    model = Qwen2ForCausalLM(config)
    # Store weights in float16 to keep the fixture under the size budget.
    # Tests run under bf16/fp16 anyway (DeepSpeed bf16, vLLM fp16 downcast)
    # so this matches the working dtype and avoids any precision surprise.
    model = model.to(torch.float16)

    print(f"Saving fixture to {FIXTURE_DIR}...")
    model.save_pretrained(FIXTURE_DIR, safe_serialization=True)
    tokenizer.save_pretrained(FIXTURE_DIR)
    # Drop redundant slow-tokenizer files; the fast tokenizer.json is sufficient.
    for legacy in ("vocab.json", "merges.txt"):
        (FIXTURE_DIR / legacy).unlink(missing_ok=True)

    config_path = FIXTURE_DIR / "config.json"
    config_dict = json.loads(config_path.read_text())
    config_dict["transformers_version"] = PINNED_TRANSFORMERS_VERSION
    config_path.write_text(json.dumps(config_dict, indent=2) + "\n")

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
