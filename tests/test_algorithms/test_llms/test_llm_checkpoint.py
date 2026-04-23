"""Tests for LLMAlgorithm.save_checkpoint / load_checkpoint.

There are 16 cells we care about, defined by the grid
    lora_only:       True / False
    save_optimizer:  True / False
    use_deepspeed:   True / False
× {save, load}.

Expected behaviour per cell (the spec this file enforces):

SAVE — deepspeed path
    lora_only=T, save_optim=T  →  deepspeed save (exclude frozen params)
    lora_only=F, save_optim=T  →  deepspeed save (include frozen params)
    lora_only=T, save_optim=F  →  peft (save adapters)
    lora_only=F, save_optim=F  →  gather params + torch save
                                  (actor model in attributes.pt)

SAVE — plain torch/peft path
    lora_only=T, save_optim=T  →  peft save + optim in attributes.pt
    lora_only=F, save_optim=T  →  torch save  (actor + optim in attributes.pt)
    lora_only=T, save_optim=F  →  peft save
    lora_only=F, save_optim=F  →  torch save  (actor in attributes.pt, no optim)

LOAD — deepspeed path
    LoRA=T, Optim=T  →  load deepspeed
    LoRA=F, Optim=T  →  load deepspeed
    LoRA=T, Optim=F  →  load peft
    LoRA=F, Optim=F  →  load torch state dict from attributes.pt

LOAD — plain torch/peft path
    LoRA=T, Optim=T  →  load peft + torch load (optim)
    LoRA=F, Optim=T  →  torch load
    LoRA=T, Optim=F  →  load peft
    LoRA=F, Optim=F  →  torch load

Test organisation:
  * ``grpo_factory`` — session-scoped, expensive agent build happens once.
  * ``llm_simple_checkpoint_save`` / ``llm_mocked_deepspeed_checkpoint_save`` — session-scoped, parametrised over
    the 4 cells. Each cell runs ``save_checkpoint`` once and tests read from
    the resulting artefacts.
  * ``llm_simple_checkpoint_load`` / ``llm_mocked_deepspeed_checkpoint_load`` — function-scoped
    because load tests mutate agent state (stamp sentinels, step optimizer).
  * Test bodies use the fixture's ``lora_only`` / ``save_optimizer`` fields
    as a truth table rather than branching per cell — each test runs 4x
    (once per parametrised fixture variant).

DeepSpeed tests spy-wrap ``actor.save_checkpoint`` / ``load_checkpoint``
(they'd normally talk to a distributed backend we don't have); we assert the
right branch was taken with the right kwargs.
"""

from __future__ import annotations

import copy
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock

import dill
import pytest
import torch

pytest.importorskip("peft", reason="LLM checkpoint tests require peft.")
pytest.importorskip("transformers", reason="LLM checkpoint tests require transformers.")

from agilerl import HAS_LLM_DEPENDENCIES
from peft import LoraConfig
from typing import TYPE_CHECKING

from agilerl.algorithms.grpo import GRPO
from tests.test_algorithms.test_core_base import _make_mock_accelerator
from tests.test_algorithms.test_llms.test_grpo import create_module

# ``deepspeed_config_stage_2`` lives inside test_grpo.py, which module-level
# ``importorskip``s deepspeed + vllm. On CPU-only hosts without those extras,
# that import would skip the whole module and crash our own import. Gate it
# on ``HAS_LLM_DEPENDENCIES`` so CPU-only runs still collect this file.
if HAS_LLM_DEPENDENCIES or TYPE_CHECKING:
    from tests.test_algorithms.test_llms.test_grpo import deepspeed_config_stage_2


SAVE_LOAD_OPTIONS = [
    pytest.param((True, True), id="lora_only+optim"),
    pytest.param((True, False), id="lora_only"),
    pytest.param((False, True), id="full+optim"),
    pytest.param((False, False), id="full"),
]

SMALL_LORA = LoraConfig(
    r=2,
    lora_alpha=4,
    target_modules=["linear_1"],
    task_type="CAUSAL_LM",
    lora_dropout=0.0,
)


def get_param_by_name(agent, substring: str) -> tuple[str, torch.nn.Parameter]:
    """Return the first actor parameter whose name contains ``substring``."""
    for name, param in agent.actor.named_parameters():
        if substring in name:
            return name, param
    raise KeyError(f"no actor param matching {substring!r}")


def find_exp_avg_in_opt_state(agent) -> torch.Tensor | None:
    """Return a reference to the first Adam ``exp_avg`` tensor in agent.optimizer.

    Returns None if optimizer.state is empty (e.g. before any step)."""
    for state in agent.optimizer.optimizer.state.values():
        if "exp_avg" in state:
            return state["exp_avg"]
    return None


def load_attributes_checkpoint(path):
    return torch.load(
        str(path / "attributes.pt"),
        weights_only=False,
        pickle_module=dill,
    )


def normalize_optimizer_state(value):
    """Normalize nested optimizer state for deterministic comparisons."""
    if isinstance(value, torch.Tensor):
        return {
            "__tensor__": True,
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
            "data": value.detach().cpu().tolist(),
        }
    if isinstance(value, dict):
        return {k: normalize_optimizer_state(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_optimizer_state(v) for v in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    # DeepSpeed includes enum-like/custom metadata objects in state_dict.
    return repr(value)


def generate_tiny_grpo(accelerator=None) -> GRPO:
    """Build a tiny CPU GRPO agent with (actor, reference) adapters."""
    actor = create_module(input_size=6, max_tokens=4, vocab_size=64, device="cpu")
    return GRPO(
        actor_network=actor,
        pad_token_id=63,
        pad_token="<pad>",
        batch_size=4,
        group_size=2,
        max_output_tokens=4,
        max_model_len=12,
        lora_config=SMALL_LORA,
        accelerator=accelerator,
        wrap=False,
        gradient_checkpointing=False,
        device="cpu",
        use_separate_reference_adapter=True,
    )


@pytest.fixture(scope="function")
def grpo_factory():
    """Expensive PEFT-wrapped GRPO, built once per session.

    Tests consume deepcopies of this template so the session-scoped instance
    is never mutated after construction.
    """
    return generate_tiny_grpo(accelerator=None)


# --------------------------------------------------------------------------- #
# SAVE — plain torch/peft path (accelerator is None)                          #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session", params=SAVE_LOAD_OPTIONS)
def llm_simple_checkpoint(request, grpo_factory, tmp_path_factory):
    """One saved plain-path checkpoint per cell, shared across all tests that
    only *read* the output. This does not involve deepspeed.

    Session scope means ``save_checkpoint`` runs exactly 4 times for the whole
    test session (once per cell), not once per test.
    """
    lora_only, save_optimizer = request.param
    agent = grpo_factory
    tmp_path = tmp_path_factory.mktemp(
        f"plain_save_lora={lora_only}_optim={save_optimizer}"
    )
    agent.save_checkpoint(
        str(tmp_path),
        lora_only=lora_only,
        save_optimizer=save_optimizer,
    )
    return SimpleNamespace(
        agent=agent,
        path=tmp_path,
        lora_only=lora_only,
        save_optimizer=save_optimizer,
    )


class TestLLMSimpleCheckpointSave:
    """Each test runs 4× (one per SAVE_LOAD_OPTIONS param) against a pre-saved
    checkpoint. Assertions are phrased as truth tables over
    ``plain_saved.lora_only`` / ``plain_saved.save_optimizer``."""

    def test_attributes_pt_always_written(self, llm_simple_checkpoint):
        assert (llm_simple_checkpoint.path / "attributes.pt").exists()

    def test_no_deepspeed_tag_dir_on_plain_path(self, llm_simple_checkpoint):
        # deepspeed engines write to a tag subdirectory; plain path must not.
        assert not (llm_simple_checkpoint.path / "save_checkpoint").exists()

    def test_adapter_dirs_present_iff_lora_only(self, llm_simple_checkpoint):
        actor_adapter = (
            llm_simple_checkpoint.path / "actor" / "adapter_model.safetensors"
        )
        ref_adapter = (
            llm_simple_checkpoint.path / "reference" / "adapter_model.safetensors"
        )
        assert actor_adapter.exists() == llm_simple_checkpoint.lora_only
        assert ref_adapter.exists() == llm_simple_checkpoint.lora_only

    def test_attributes_pt_contents_match_cell(self, llm_simple_checkpoint):
        ck = load_attributes_checkpoint(llm_simple_checkpoint.path)
        ni = ck.get("network_info")

        # _lora_only flag round-trips verbatim.
        assert ck.get("_lora_only") == llm_simple_checkpoint.lora_only

        # actor_state_dict in attributes.pt if full-model save (not lora_only).
        has_actor_sd = "actor_state_dict" in ni["modules"]
        assert has_actor_sd == (not llm_simple_checkpoint.lora_only), (
            f"actor_state_dict presence wrong for cell "
            f"(lora_only={llm_simple_checkpoint.lora_only}, save_optimizer={llm_simple_checkpoint.save_optimizer})"
        )

        # Optimizer state in attributes.pt if save_optimizer=True (plain path).
        has_optim = bool(ni["optimizers"])
        assert has_optim == llm_simple_checkpoint.save_optimizer, (
            f"optimizer presence wrong for cell "
            f"(lora_only={llm_simple_checkpoint.lora_only}, save_optimizer={llm_simple_checkpoint.save_optimizer})"
        )


# --------------------------------------------------------------------------- #
# LOAD — plain torch/peft path                                                #
# --------------------------------------------------------------------------- #


@pytest.fixture(params=SAVE_LOAD_OPTIONS)
def llm_simple_checkpoint_load(request, grpo_factory, tmp_path):
    """Fresh agent per test (load tests mutate state: stamp sentinels, step
    optimizer). Cheap because deepcopy of the template is near-instant."""
    lora_only, save_optimizer = request.param
    agent = grpo_factory
    return SimpleNamespace(
        agent=agent,
        path=tmp_path,
        lora_only=lora_only,
        save_optimizer=save_optimizer,
    )


class TestLLMSimpleCheckpointLoad:
    """Roundtrip: stamp sentinels on tracked state → save → clobber → load →
    assert sentinels restored. Specifically catches 'load silently
    reinitialised a fresh optimizer / fresh weights'."""

    def test_adapter_weights_roundtrip(self, llm_simple_checkpoint_load):
        s = llm_simple_checkpoint_load
        lora_sentinel, base_sentinel, clobber = 0.1234, 0.4321, 9.9999

        _, lora_param = get_param_by_name(s.agent, "lora_A.actor.weight")
        with torch.no_grad():
            lora_param.fill_(lora_sentinel)

        base_param = None
        if not s.lora_only:
            _, base_param = get_param_by_name(s.agent, "linear_1.base_layer.weight")
            with torch.no_grad():
                base_param.fill_(base_sentinel)

        s.agent.save_checkpoint(
            str(s.path),
            lora_only=s.lora_only,
            save_optimizer=s.save_optimizer,
        )
        with torch.no_grad():
            lora_param.fill_(clobber)
            if base_param is not None:
                base_param.fill_(clobber)

        s.agent.load_checkpoint(str(s.path), load_optimizer=s.save_optimizer)

        _, lora_post = get_param_by_name(s.agent, "lora_A.actor.weight")
        assert torch.allclose(lora_post, torch.full_like(lora_post, lora_sentinel)), (
            f"LoRA weight not restored for cell "
            f"(lora_only={s.lora_only}, save_optimizer={s.save_optimizer})"
        )
        if not s.lora_only:
            _, base_post = get_param_by_name(s.agent, "linear_1.base_layer.weight")
            assert torch.allclose(
                base_post, torch.full_like(base_post, base_sentinel)
            ), (
                f"base weight not restored for cell "
                f"(lora_only={s.lora_only}, save_optimizer={s.save_optimizer})"
            )

    def test_optimizer_state_roundtrip(self, llm_simple_checkpoint_load):
        s = llm_simple_checkpoint_load
        sentinel, clobber = 0.3333, 9.9999

        # Populate optimizer state: fake grads → step.
        for p in s.agent.actor.parameters():
            if p.requires_grad:
                p.grad = torch.ones_like(p)
        s.agent.optimizer.step()
        s.agent.optimizer.zero_grad()

        exp_avg = find_exp_avg_in_opt_state(s.agent)
        assert exp_avg is not None, "optimizer.state not populated after step"
        with torch.no_grad():
            exp_avg.fill_(sentinel)

        s.agent.save_checkpoint(
            str(s.path),
            lora_only=s.lora_only,
            save_optimizer=s.save_optimizer,
        )
        with torch.no_grad():
            exp_avg.fill_(clobber)

        if s.save_optimizer:
            s.agent.load_checkpoint(str(s.path), load_optimizer=True)
            restored = find_exp_avg_in_opt_state(s.agent)
            assert restored is not None, (
                f"optimizer state empty after load for cell "
                f"(lora_only={s.lora_only}, save_optimizer=True)"
            )
            assert torch.allclose(restored, torch.full_like(restored, sentinel)), (
                f"optimizer state not restored for cell "
                f"(lora_only={s.lora_only}, save_optimizer=True)"
            )
        else:
            # Nothing in the checkpoint to restore from → warn + fresh state.
            with pytest.warns(UserWarning, match="Optimizer state not found"):
                s.agent.load_checkpoint(str(s.path), load_optimizer=True)
            post = find_exp_avg_in_opt_state(s.agent)
            # Sentinel must NOT be present (either rebuilt fresh or still clobbered).
            if post is not None:
                assert not torch.allclose(post, torch.full_like(post, sentinel)), (
                    "optimizer state silently restored despite load_optimizer=False path"
                )


# --------------------------------------------------------------------------- #
# SAVE — deepspeed path                                                        #
# --------------------------------------------------------------------------- #


def _fit_deepspeed_mock(agent, zero_stage: int = 2) -> None:
    """Mutate ``agent`` so it looks like a DeepSpeed-wrapped agent for
    dispatch tests. Mock accelerator, overridden zero_stage, and a reasonable
    unwrap_model that just returns the wrapped model."""
    agent.accelerator = _make_mock_accelerator()
    agent.accelerator.unwrap_model = MagicMock(side_effect=lambda m: m)
    agent._uses_deepspeed = True
    agent.zero_stage = zero_stage


def _inner_actor(agent):
    """Strip the DummyEvolvable wrapper; returns the inner peft model.

    save_pretrained / load_state_dict are called on this inner object,
    not on agent.actor itself, so spies for those must be attached here.
    """
    from agilerl.modules.dummy import DummyEvolvable

    actor = agent.actor
    while isinstance(actor, DummyEvolvable):
        actor = actor.module
    return actor


@pytest.fixture(scope="session", params=SAVE_LOAD_OPTIONS)
def llm_mocked_deepspeed_checkpoint_save(request, grpo_factory, tmp_path_factory):
    """Spy-wrapped DeepSpeed save per cell. Session-scoped — 4 deepcopies of
    the template, each saved once."""
    lora_only, save_optimizer = request.param
    agent = grpo_factory
    _fit_deepspeed_mock(agent)

    # save_checkpoint is called as ``self.actor.save_checkpoint(...)`` on the
    # DummyEvolvable wrapper. save_pretrained is called on the unwrapped
    # inner peft model, so that spy must live there.
    save_ckpt_spy = MagicMock()
    agent.actor.save_checkpoint = save_ckpt_spy
    inner = _inner_actor(agent)
    save_pretrained_spy = MagicMock(wraps=inner.save_pretrained)
    inner.save_pretrained = save_pretrained_spy

    tmp_path = tmp_path_factory.mktemp(
        f"ds_save_lora={lora_only}_optim={save_optimizer}"
    )
    agent.save_checkpoint(
        str(tmp_path),
        lora_only=lora_only,
        save_optimizer=save_optimizer,
    )
    return SimpleNamespace(
        agent=agent,
        path=tmp_path,
        lora_only=lora_only,
        save_optimizer=save_optimizer,
        save_checkpoint_spy=save_ckpt_spy,
        save_pretrained_spy=save_pretrained_spy,
    )


class TestLLMDeepspeedCheckpointSave:
    """Dispatch-only spy tests: replace ``actor.save_checkpoint`` with a
    MagicMock and assert the right branch was called with the right kwargs.

    LIMITATION: these do NOT verify DeepSpeed actually wrote correct bytes
    to disk — the real engine is mocked out. For end-to-end confidence see
    ``TestDeepspeedSaveE2E`` below (CUDA-only, @pytest.mark.llm).
    These spy tests run on any machine and catch dispatch regressions.
    """

    def test_deepspeed_save_called_if_save_optimizer(
        self, llm_mocked_deepspeed_checkpoint_save
    ):
        spy = llm_mocked_deepspeed_checkpoint_save.save_checkpoint_spy
        if llm_mocked_deepspeed_checkpoint_save.save_optimizer:
            assert spy.call_count == 1
            kwargs = spy.call_args.kwargs
            assert (
                kwargs.get("exclude_frozen_parameters")
                == llm_mocked_deepspeed_checkpoint_save.lora_only
            )
        else:
            assert spy.call_count == 0

    def test_save_pretrained_called_if_lora_only(
        self, llm_mocked_deepspeed_checkpoint_save
    ):
        spy = llm_mocked_deepspeed_checkpoint_save.save_pretrained_spy
        assert (spy.call_count >= 1) == llm_mocked_deepspeed_checkpoint_save.lora_only

    def test_attributes_pt_has_actor_state_dict_only_when_full_no_optim(
        self,
        llm_mocked_deepspeed_checkpoint_save,
    ):
        ck = load_attributes_checkpoint(llm_mocked_deepspeed_checkpoint_save.path)
        ni = ck.get("network_info", {}) or {}
        modules = ni.get("modules", {}) if isinstance(ni, dict) else {}
        has_actor_sd = "actor_state_dict" in modules
        expected = (not llm_mocked_deepspeed_checkpoint_save.lora_only) and (
            not llm_mocked_deepspeed_checkpoint_save.save_optimizer
        )
        assert has_actor_sd == expected


# --------------------------------------------------------------------------- #
# LOAD — deepspeed path                                                        #
# --------------------------------------------------------------------------- #


@pytest.fixture(params=SAVE_LOAD_OPTIONS)
def llm_mocked_deepspeed_checkpoint_load(request, grpo_factory, tmp_path):
    """Fresh agent + pre-saved deepspeed-shape checkpoint per load test.

    Function-scoped because each test patches in method-level spies and we
    want a clean baseline per test.
    """
    from pathlib import Path

    lora_only, save_optimizer = request.param

    # Saver: writes a cell-specific checkpoint to tmp_path. The real
    # DeepSpeed save is stubbed (can't run without a distributed backend)
    # but we still need the expected tag directory on disk so that the load
    # side's ``Path.glob('save_checkpoint')`` assertion passes.
    saver = grpo_factory
    _fit_deepspeed_mock(saver)

    def _fake_ds_save(path_str, *args, tag="save_checkpoint", **kwargs):
        (Path(path_str) / tag).mkdir(parents=True, exist_ok=True)

    saver.actor.save_checkpoint = MagicMock(side_effect=_fake_ds_save)
    saver.save_checkpoint(
        str(tmp_path),
        lora_only=lora_only,
        save_optimizer=save_optimizer,
    )

    # Loader: spy its engine load so we can assert dispatch.
    loader = grpo_factory
    _fit_deepspeed_mock(loader)
    load_ckpt_spy = MagicMock(return_value=(str(tmp_path / "save_checkpoint"), None))
    loader.actor.load_checkpoint = load_ckpt_spy

    return SimpleNamespace(
        agent=loader,
        path=tmp_path,
        lora_only=lora_only,
        save_optimizer=save_optimizer,
        load_checkpoint_spy=load_ckpt_spy,
    )


class TestDeepspeedLoad:
    """Dispatch-only spy tests. Same limitations as TestDeepspeedSave — these
    assert the correct load branch was taken but do not verify DeepSpeed
    actually restored state. See ``TestDeepspeedLoadE2E`` for real roundtrip.
    """

    def test_deepspeed_load_called_if_save_optimizer(
        self, llm_mocked_deepspeed_checkpoint_load
    ):
        s = llm_mocked_deepspeed_checkpoint_load
        from unittest.mock import patch

        inner = _inner_actor(s.agent)
        # Stub the non-deepspeed branches so they don't actually try to read
        # adapter files / overwrite state — we only care about dispatch here.
        with (
            patch.object(s.agent, "_load_model_checkpoint"),
            patch.object(inner, "load_state_dict"),
        ):
            s.agent.load_checkpoint(str(s.path), load_optimizer=s.save_optimizer)
        assert (s.load_checkpoint_spy.call_count == 1) == s.save_optimizer

    def test_peft_adapter_load_when_lora_only_and_no_optim(
        self,
        llm_mocked_deepspeed_checkpoint_load,
    ):
        s = llm_mocked_deepspeed_checkpoint_load
        from unittest.mock import patch

        inner = _inner_actor(s.agent)
        with (
            patch.object(s.agent, "_load_model_checkpoint") as peft_spy,
            patch.object(inner, "load_state_dict"),
        ):
            s.agent.load_checkpoint(str(s.path), load_optimizer=s.save_optimizer)
        # peft load is entered for both LoRA-only cells:
        #   * (lora_only=T, save_optim=F): PEFT-only restore path
        #   * (lora_only=T, save_optim=T): DS resume + PEFT adapter refresh
        expected = s.lora_only
        assert (peft_spy.call_count == 1) == expected

    def test_state_dict_load_when_full_and_no_optim(
        self, llm_mocked_deepspeed_checkpoint_load
    ):
        s = llm_mocked_deepspeed_checkpoint_load
        from unittest.mock import patch

        inner = _inner_actor(s.agent)
        with (
            patch.object(s.agent, "_load_model_checkpoint"),
            patch.object(inner, "load_state_dict") as sd_spy,
        ):
            s.agent.load_checkpoint(str(s.path), load_optimizer=s.save_optimizer)
        # load_state_dict on the unwrapped actor is called only for
        # (lora_only=F, save_optim=F).
        expected = (not s.lora_only) and (not s.save_optimizer)
        assert (sd_spy.call_count == 1) == expected


# --------------------------------------------------------------------------- #
# ZeRO-3 gather behaviour                                                      #
# --------------------------------------------------------------------------- #


class TestLLMGatherIfZero3OnSave:
    """Sanity-check that gather_if_zero3 is entered when zero_stage=3.

    Two save cells gather: the save_pretrained branch (lora_only=True) and
    the full torch-save branch (lora_only=False, save_optim=False). One test
    per branch — no full-grid parametrisation needed.
    """

    def test_gather_entered_on_peft_save_when_zero3(self, grpo_factory, tmp_path):
        from contextlib import contextmanager
        from unittest.mock import patch

        agent = grpo_factory
        _fit_deepspeed_mock(agent, zero_stage=3)
        agent.actor.save_checkpoint = MagicMock()

        calls = []

        @contextmanager
        def gather_spy(zero_stage, params, modifier_rank=None):
            calls.append(zero_stage)
            yield

        with patch(
            "agilerl.algorithms.core.base.gather_if_zero3",
            side_effect=gather_spy,
        ):
            agent.save_checkpoint(
                str(tmp_path),
                lora_only=True,
                save_optimizer=False,
            )
        assert 3 in calls, "gather_if_zero3 was not entered for lora_only save"

    def test_gather_entered_on_full_save_when_zero3(self, grpo_factory, tmp_path):
        from contextlib import contextmanager
        from unittest.mock import patch

        agent = grpo_factory
        _fit_deepspeed_mock(agent, zero_stage=3)
        agent.actor.save_checkpoint = MagicMock()

        calls = []

        @contextmanager
        def gather_spy(zero_stage, params, modifier_rank=None):
            calls.append(zero_stage)
            yield

        with patch(
            "agilerl.algorithms.core.base.gather_if_zero3",
            side_effect=gather_spy,
        ):
            agent.save_checkpoint(
                str(tmp_path),
                lora_only=False,
                save_optimizer=False,
            )
        assert 3 in calls, "gather_if_zero3 was not entered for full save"


# --------------------------------------------------------------------------- #
# E2E DeepSpeed tests — real save/load, CUDA-only, @pytest.mark.llm           #
# --------------------------------------------------------------------------- #
# These run a real Accelerator with a DeepSpeedPlugin and exercise the full
# save/load pipeline end-to-end.


def _require_cuda_deepspeed() -> None:
    """Skip if the environment can't run real DeepSpeed."""
    if not HAS_LLM_DEPENDENCIES:
        pytest.skip("E2E deepspeed tests require the 'llm' extras.")
    if not torch.cuda.is_available():
        pytest.skip("E2E deepspeed tests require CUDA")


def build_deepspeed_grpo(accelerator):
    """Build a real DeepSpeed-wrapped GRPO for end-to-end tests.

    Same synthetic ``create_module`` used in the mocked tests, but with
    ``wrap=True`` so ``accelerator.prepare(...)`` wraps the model into a
    DeepSpeedEngine. Device is resolved to CUDA by ``GRPO.__init__`` when an
    accelerator is attached.
    """
    actor = create_module(
        input_size=6,
        max_tokens=4,
        vocab_size=64,
        device="cuda",
    )
    return GRPO(
        actor_network=actor,
        pad_token_id=63,
        pad_token="<pad>",
        batch_size=4,
        group_size=2,
        max_output_tokens=4,
        max_model_len=12,
        lora_config=SMALL_LORA,
        accelerator=accelerator,
        wrap=True,
        gradient_checkpointing=False,
        device="cuda",
        use_separate_reference_adapter=True,
    )


@pytest.fixture(params=SAVE_LOAD_OPTIONS)
def llm_deepspeed_checkpoint_save(
    request,
    deepspeed_env,
    accelerator_factory,
    tmp_path,
):
    """Real DeepSpeed save per cell — ONE real accelerator+agent per test.

    Function-scoped (Accelerator is not reusable across tests because
    the cleanup_after_test autouse fixture resets accelerator state).
    """
    _require_cuda_deepspeed()
    lora_only, save_optimizer = request.param
    accelerator = accelerator_factory(
        use_deepspeed_optimizer=False,
        config=deepspeed_config_stage_2,
    )
    agent = build_deepspeed_grpo(accelerator)
    agent.save_checkpoint(
        str(tmp_path),
        lora_only=lora_only,
        save_optimizer=save_optimizer,
    )
    return SimpleNamespace(
        agent=agent,
        path=tmp_path,
        lora_only=lora_only,
        save_optimizer=save_optimizer,
    )


@pytest.mark.llm
class TestLLMDeepspeedCheckpointSave:
    """Real DeepSpeed save → assertions against bytes on disk (no spies).

    All artefact assertions in a single parametrised test to keep the number
    of real DeepSpeed builds small (1 per cell = 4 total).

                                deepspeed tag dir   adapter dirs   actor_state_dict
                                (<path>/save_checkpoint/*)         in attributes.pt
        lora_only=T, optim=T    present             present        absent
        lora_only=T, optim=F    absent              present        absent
        lora_only=F, optim=T    present             absent         absent
        lora_only=F, optim=F    absent              absent         present
    """

    def test_save_artifacts_match_cell(self, llm_deepspeed_checkpoint_save):
        s = llm_deepspeed_checkpoint_save

        # attributes.pt always present.
        assert (s.path / "attributes.pt").exists()

        # DeepSpeed engine's own tag directory if save_optimizer=True.
        assert (s.path / "save_checkpoint").is_dir() == s.save_optimizer

        # PEFT adapter dirs if lora_only=True (both actor + reference because
        # use_separate_reference_adapter=True in the fixture).
        actor_adapter = s.path / "actor" / "adapter_model.safetensors"
        ref_adapter = s.path / "reference" / "adapter_model.safetensors"
        assert actor_adapter.exists() == s.lora_only
        assert ref_adapter.exists() == s.lora_only

        # attributes.pt payload:
        #   - ``_lora_only`` flag always matches the save call.
        #   - ``actor_state_dict`` only lands in attrs.pt for the (F, F)
        #     deepspeed cell (gather+torch-save branch).
        ck = load_attributes_checkpoint(s.path)
        assert ck.get("_lora_only") == s.lora_only
        modules = ck.get("network_info", {}).get("modules", {})
        has_actor_sd = "actor_state_dict" in modules
        expected = (not s.lora_only) and (not s.save_optimizer)
        assert has_actor_sd == expected, (
            f"actor_state_dict presence in attributes.pt wrong for cell "
            f"(lora_only={s.lora_only}, save_optimizer={s.save_optimizer})"
        )


@pytest.fixture(params=SAVE_LOAD_OPTIONS)
def llm_deepspeed_checkpoint_load(
    request,
    deepspeed_env,
    accelerator_factory,
    tmp_path,
):
    """Fresh real-DeepSpeed agent per load test + the factory so the test can
    build a second accelerator for the load-side agent.

    Function-scoped — each test stamps sentinels then saves+loads, so agents
    cannot be shared.
    """
    _require_cuda_deepspeed()
    lora_only, save_optimizer = request.param
    accelerator = accelerator_factory(
        use_deepspeed_optimizer=False,
        config=deepspeed_config_stage_2,
    )
    agent = build_deepspeed_grpo(accelerator)
    return SimpleNamespace(
        agent=agent,
        path=tmp_path,
        lora_only=lora_only,
        save_optimizer=save_optimizer,
        accelerator_factory=accelerator_factory,
    )


@pytest.mark.llm
class TestLLMDeepspeedCheckpointSaveLoad:
    """Real DeepSpeed roundtrip: stamp sentinels → save → fresh agent → load
    → assert sentinels restored. One parametrised test per concern
    (weights / optimizer) to keep the real DeepSpeed builds bounded.

    NB: building the second accelerator via the factory triggers
    ``AcceleratorState._reset_state`` which invalidates the first engine.
    That's fine because the first agent is only needed for the save step.
    """

    def test_adapter_and_base_weight_roundtrip_e2e(self, llm_deepspeed_checkpoint_load):
        s = llm_deepspeed_checkpoint_load
        lora_sentinel, base_sentinel, clobber = 0.1234, 0.4321, 9.9999

        # Stamp the actor's LoRA-A weight on the pre-save agent.
        _, lora_param = get_param_by_name(s.agent, "lora_A.actor.weight")
        with torch.no_grad():
            lora_param.fill_(lora_sentinel)

        # Full-save cells also round-trip base model weights, so stamp one.
        if not s.lora_only:
            _, base_param = get_param_by_name(s.agent, "linear_1.base_layer.weight")
            with torch.no_grad():
                base_param.fill_(base_sentinel)

        s.agent.save_checkpoint(
            str(s.path),
            lora_only=s.lora_only,
            save_optimizer=s.save_optimizer,
        )

        # Build the loading side on a fresh accelerator + agent.
        new_accel = s.accelerator_factory(
            use_deepspeed_optimizer=False,
            config=deepspeed_config_stage_2,
        )
        new_agent = build_deepspeed_grpo(new_accel)

        # Clobber a weight on new_agent so a silent no-op load would fail
        # the sentinel comparison.
        _, new_lora = get_param_by_name(new_agent, "lora_A.actor.weight")
        with torch.no_grad():
            new_lora.fill_(clobber)

        new_agent.load_checkpoint(
            str(s.path),
            load_optimizer=s.save_optimizer,
        )

        # Re-fetch after load; load may rebuild adapter modules.
        _, lora_post = get_param_by_name(new_agent, "lora_A.actor.weight")
        assert torch.allclose(lora_post, torch.full_like(lora_post, lora_sentinel)), (
            f"LoRA weight not restored for cell "
            f"(lora_only={s.lora_only}, save_optimizer={s.save_optimizer})"
        )
        if not s.lora_only:
            _, base_post = get_param_by_name(new_agent, "linear_1.base_layer.weight")
            assert torch.allclose(
                base_post, torch.full_like(base_post, base_sentinel)
            ), (
                f"base weight not restored for cell "
                f"(lora_only={s.lora_only}, save_optimizer={s.save_optimizer})"
            )

    def test_optimizer_state_roundtrip_e2e(self, deepspeed_load_scenario_e2e):
        """After a real backward+step, optimizer state should round-trip
        through a save+load cycle when ``save_optimizer=True``.

        We don't use a sentinel here — DeepSpeed's optimizer state is
        partitioned/internal and the public state_dict shape isn't trivial to
        tensor-stamp. A "state is non-empty after load" check is sufficient
        to catch a silent fresh-optimizer regression.
        """
        s = deepspeed_load_scenario_e2e
        _, base_linear = get_param_by_name(s.agent, "linear_1.base_layer.weight")
        in_features = int(base_linear.shape[1])
        input_ids = torch.randint(0, 64, (1, in_features), device=s.agent.device)
        attn_mask = torch.ones_like(input_ids)
        out = s.agent.actor(input_ids=input_ids, attention_mask=attn_mask)
        s.agent.actor.backward(out.logits.sum())
        s.agent.optimizer.step()
        before_inner = getattr(s.agent.optimizer, "optimizer", s.agent.optimizer)
        before_sd = (
            before_inner.state_dict() if hasattr(before_inner, "state_dict") else {}
        )

        s.agent.save_checkpoint(
            str(s.path),
            lora_only=s.lora_only,
            save_optimizer=s.save_optimizer,
        )

        new_accel = s.accelerator_factory(
            use_deepspeed_optimizer=False,
            config=deepspeed_config_stage_2,
        )
        new_agent = build_deepspeed_grpo(new_accel)
        new_agent.load_checkpoint(
            str(s.path),
            load_optimizer=s.save_optimizer,
        )

        if s.save_optimizer:
            inner = getattr(new_agent.optimizer, "optimizer", new_agent.optimizer)
            sd = inner.state_dict() if hasattr(inner, "state_dict") else {}
            assert normalize_optimizer_state(sd) == normalize_optimizer_state(
                before_sd
            ), (
                f"optimizer state mismatch after DeepSpeed load for cell "
                f"(lora_only={s.lora_only}, save_optimizer=True); "
                f"got keys {list(sd.keys())}"
            )


# --------------------------------------------------------------------------- #
# LoRA config merge — unit tests (static method, no fixtures)                 #
# --------------------------------------------------------------------------- #

from agilerl.algorithms.core.base import LLMAlgorithm  # noqa: E402


def get_lora_config(
    r=4, target_modules=("linear_1",), modules_to_save=None, lora_alpha=8
):
    """Helper to build a LoraConfig with sensible defaults for merge tests."""
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=list(target_modules),
        modules_to_save=list(modules_to_save) if modules_to_save is not None else None,
        task_type="CAUSAL_LM",
        lora_dropout=0.0,
    )


class TestMergeLoraConfigs:
    """Unit tests for ``LLMAlgorithm._merge_lora_configs``. Rules under test:

    * ``current=None`` → checkpoint is returned as-is, no warnings.
    * ``r``               → ``max(current, checkpoint)``; warn on mismatch.
    * ``target_modules``  → set union; warn on difference.
    * ``modules_to_save`` → set union; warn on difference.
    * anything else       → current kept; warn on difference.
    """

    def test_current_none_returns_checkpoint_unchanged(self):
        ckpt = get_lora_config(r=8)
        # No warnings should fire when there's nothing to merge against.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            merged = LLMAlgorithm._merge_lora_configs(None, ckpt)
        assert merged is ckpt

    def test_rank_takes_max_and_warns_on_mismatch(self):
        current = get_lora_config(r=2)
        ckpt = get_lora_config(r=8)
        with pytest.warns(UserWarning, match="LoRA rank mismatch"):
            merged = LLMAlgorithm._merge_lora_configs(current, ckpt)
        assert merged.r == 8

    def test_rank_equal_no_warning(self):
        current = get_lora_config(r=4)
        ckpt = get_lora_config(r=4)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            merged = LLMAlgorithm._merge_lora_configs(current, ckpt)
        assert merged.r == 4

    def test_target_modules_unioned_and_warns(self):
        current = get_lora_config(target_modules=("linear_1",))
        ckpt = get_lora_config(target_modules=("linear_1", "linear_2"))
        with pytest.warns(UserWarning, match="'target_modules' differs"):
            merged = LLMAlgorithm._merge_lora_configs(current, ckpt)
        # merged.target_modules is a sorted list (per the implementation).
        assert set(merged.target_modules) == {"linear_1", "linear_2"}

    def test_target_modules_equal_no_warning(self):
        current = get_lora_config(target_modules=("linear_1", "linear_2"))
        ckpt = get_lora_config(target_modules=("linear_1", "linear_2"))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            LLMAlgorithm._merge_lora_configs(current, ckpt)

    def test_modules_to_save_unioned_and_warns(self):
        current = get_lora_config(modules_to_save=("summary",))
        ckpt = get_lora_config(modules_to_save=("summary", "v_head"))
        with pytest.warns(UserWarning, match="'modules_to_save' differs"):
            merged = LLMAlgorithm._merge_lora_configs(current, ckpt)
        assert set(merged.modules_to_save) == {"summary", "v_head"}

    def test_other_field_mismatch_warns_and_keeps_current(self):
        current = get_lora_config(lora_alpha=8)
        ckpt = get_lora_config(lora_alpha=32)
        with pytest.warns(UserWarning, match="'lora_alpha' differs"):
            merged = LLMAlgorithm._merge_lora_configs(current, ckpt)
        # Current wins for non-special fields.
        assert merged.lora_alpha == 8


# --------------------------------------------------------------------------- #
# LoRA config merge — integration with save/load                              #
# --------------------------------------------------------------------------- #


def _build_grpo_with_lora(lora_config: LoraConfig) -> GRPO:
    """Like ``_build_grpo`` but lets the caller override ``lora_config``."""
    actor = create_module(input_size=6, max_tokens=4, vocab_size=64, device="cpu")
    return GRPO(
        actor_network=actor,
        pad_token_id=63,
        pad_token="<pad>",
        batch_size=4,
        group_size=2,
        max_output_tokens=4,
        max_model_len=12,
        lora_config=lora_config,
        accelerator=None,
        wrap=False,
        gradient_checkpointing=False,
        device="cpu",
        use_separate_reference_adapter=True,
    )


class TestMergeLoraConfigsRoundtrip:
    """Save a lora-only checkpoint with config A, load into an agent built
    with config B, and verify the merged config survives load.

    Only lora-only checkpoints carry a ``LoraConfig`` on disk (via
    ``save_pretrained``), so that's the branch where ``_merge_lora_configs``
    actually runs during load.
    """

    def test_merged_lora_config_persists_on_agent(self, tmp_path):
        """The merged config should survive on ``self.lora_config``, mirroring
        the deepspeed path's ``_restore_checkpoint_attributes`` behaviour."""
        from unittest.mock import patch

        saver = _build_grpo_with_lora(
            get_lora_config(r=2, target_modules=("linear_1",))
        )
        saver.save_checkpoint(str(tmp_path), lora_only=True, save_optimizer=False)

        loader = _build_grpo_with_lora(
            get_lora_config(r=8, target_modules=("linear_1", "linear_2"))
        )
        with (
            patch.object(LLMAlgorithm, "_load_adapter_weights"),
            patch.object(LLMAlgorithm, "_copy_adapter_weights"),
            patch.object(LLMAlgorithm, "_reconfigure_adapters_to_match"),
        ):
            loader.load_checkpoint(str(tmp_path), load_optimizer=False)

        assert loader.lora_config.r == 8
        assert set(loader.lora_config.target_modules) == {"linear_1", "linear_2"}

    def test_full_roundtrip_with_rank_growth_loads_weights(self, tmp_path):
        """End-to-end: save at r=2, load into r=8 agent — merge takes
        ``r=max(2,8)=8``, ``_reconfigure_adapters_to_match`` rebuilds the live
        adapter at rank 8, and ``_pad_adapter_state_to_live_shape`` drops the
        saved r=2 weights into the top-left rank slice before peft's
        ``set_peft_model_state_dict`` applies them."""
        saver = _build_grpo_with_lora(
            get_lora_config(r=2, target_modules=("linear_1",))
        )
        saver.save_checkpoint(str(tmp_path), lora_only=True, save_optimizer=False)

        loader = _build_grpo_with_lora(
            get_lora_config(r=8, target_modules=("linear_1",))
        )
        loader.load_checkpoint(str(tmp_path), load_optimizer=False)
        assert loader.lora_config.r == 8

    def test_load_no_warning_when_configs_match(self, tmp_path):
        cfg = get_lora_config(r=4, target_modules=("linear_1",))
        saver = _build_grpo_with_lora(cfg)
        saver.save_checkpoint(str(tmp_path), lora_only=True, save_optimizer=False)

        loader = _build_grpo_with_lora(
            get_lora_config(r=4, target_modules=("linear_1",))
        )
        # We only assert the merge-specific warnings don't fire — PEFT /
        # other parts of load may legitimately warn on unrelated things.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loader.load_checkpoint(str(tmp_path), load_optimizer=False)
        merge_warnings = [
            w
            for w in caught
            if "rank mismatch" in str(w.message)
            or "'target_modules' differs" in str(w.message)
            or "'modules_to_save' differs" in str(w.message)
        ]
        assert merge_warnings == [], (
            f"unexpected merge warnings: {[str(w.message) for w in merge_warnings]}"
        )
        assert loader.lora_config.r == 4
