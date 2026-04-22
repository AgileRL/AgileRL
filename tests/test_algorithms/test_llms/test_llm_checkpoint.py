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
  * ``grpo_template`` — session-scoped, expensive agent build happens once.
  * ``plain_saved`` / ``deepspeed_saved`` — session-scoped, parametrised over
    the 4 cells. Each cell runs ``save_checkpoint`` once and tests read from
    the resulting artefacts.
  * ``plain_load_scenario`` / ``deepspeed_load_scenario`` — function-scoped
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
from types import SimpleNamespace
from unittest.mock import MagicMock

import dill
import pytest
import torch

pytest.importorskip("peft", reason="LLM checkpoint tests require peft.")
pytest.importorskip("transformers", reason="LLM checkpoint tests require transformers.")

from peft import LoraConfig

from agilerl.algorithms.grpo import GRPO
from tests.test_algorithms.test_core_base import _make_mock_accelerator
from tests.test_algorithms.test_llms.test_grpo import create_module


def _find_param(agent, substring: str) -> tuple[str, torch.nn.Parameter]:
    """Return the first actor parameter whose name contains ``substring``."""
    for name, param in agent.actor.named_parameters():
        if substring in name:
            return name, param
    raise KeyError(f"no actor param matching {substring!r}")


def _find_exp_avg(agent) -> torch.Tensor | None:
    """Return a reference to the first Adam ``exp_avg`` tensor in agent.optimizer.

    Returns None if optimizer.state is empty (e.g. before any step)."""
    for state in agent.optimizer.optimizer.state.values():
        if "exp_avg" in state:
            return state["exp_avg"]
    return None


def _load_attributes_pt(path):
    return torch.load(
        str(path / "attributes.pt"),
        weights_only=False,
        pickle_module=dill,
    )


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


def _build_grpo(accelerator=None) -> GRPO:
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


@pytest.fixture(scope="session")
def grpo_template():
    """Expensive PEFT-wrapped GRPO, built once per session.

    Tests consume deepcopies of this template so the session-scoped instance
    is never mutated after construction.
    """
    return _build_grpo(accelerator=None)


# --------------------------------------------------------------------------- #
# SAVE — plain torch/peft path (accelerator is None)                          #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session", params=SAVE_LOAD_OPTIONS)
def plain_saved(request, grpo_template, tmp_path_factory):
    """One saved plain-path checkpoint per cell, shared across all tests that
    only *read* the output.

    Session scope means ``save_checkpoint`` runs exactly 4 times for the whole
    test session (once per cell), not once per test.
    """
    lora_only, save_optimizer = request.param
    agent = copy.deepcopy(grpo_template)
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


class TestPlainSave:
    """Each test runs 4× (one per SAVE_LOAD_OPTIONS param) against a pre-saved
    checkpoint. Assertions are phrased as truth tables over
    ``plain_saved.lora_only`` / ``plain_saved.save_optimizer``."""

    def test_attributes_pt_always_written(self, plain_saved):
        assert (plain_saved.path / "attributes.pt").exists()

    def test_no_deepspeed_tag_dir_on_plain_path(self, plain_saved):
        # deepspeed engines write to a tag subdirectory; plain path must not.
        assert not (plain_saved.path / "save_checkpoint").exists()

    def test_adapter_dirs_present_iff_lora_only(self, plain_saved):
        actor_adapter = plain_saved.path / "actor" / "adapter_model.safetensors"
        ref_adapter = plain_saved.path / "reference" / "adapter_model.safetensors"
        assert actor_adapter.exists() == plain_saved.lora_only
        assert ref_adapter.exists() == plain_saved.lora_only

    def test_attributes_pt_contents_match_cell(self, plain_saved):
        ck = _load_attributes_pt(plain_saved.path)
        ni = ck.get("network_info")

        # _lora_only flag round-trips verbatim.
        assert ck.get("_lora_only") == plain_saved.lora_only

        # actor_state_dict in attributes.pt iff full-model save (not lora_only).
        has_actor_sd = "actor_state_dict" in ni["modules"]
        assert has_actor_sd == (not plain_saved.lora_only), (
            f"actor_state_dict presence wrong for cell "
            f"(lora_only={plain_saved.lora_only}, save_optimizer={plain_saved.save_optimizer})"
        )

        # Optimizer state in attributes.pt iff save_optimizer=True (plain path).
        has_optim = bool(ni["optimizers"])
        assert has_optim == plain_saved.save_optimizer, (
            f"optimizer presence wrong for cell "
            f"(lora_only={plain_saved.lora_only}, save_optimizer={plain_saved.save_optimizer})"
        )


# --------------------------------------------------------------------------- #
# LOAD — plain torch/peft path                                                #
# --------------------------------------------------------------------------- #


@pytest.fixture(params=SAVE_LOAD_OPTIONS)
def plain_load_scenario(request, grpo_template, tmp_path):
    """Fresh agent per test (load tests mutate state: stamp sentinels, step
    optimizer). Cheap because deepcopy of the template is near-instant."""
    lora_only, save_optimizer = request.param
    agent = copy.deepcopy(grpo_template)
    return SimpleNamespace(
        agent=agent,
        path=tmp_path,
        lora_only=lora_only,
        save_optimizer=save_optimizer,
    )


class TestPlainLoad:
    """Roundtrip: stamp sentinels on tracked state → save → clobber → load →
    assert sentinels restored. Specifically catches 'load silently
    reinitialised a fresh optimizer / fresh weights'."""

    def test_adapter_weights_roundtrip(self, plain_load_scenario):
        s = plain_load_scenario
        lora_sentinel, base_sentinel, clobber = 0.1234, 0.4321, 9.9999

        _, lora_param = _find_param(s.agent, "lora_A.actor.weight")
        with torch.no_grad():
            lora_param.fill_(lora_sentinel)

        base_param = None
        if not s.lora_only:
            _, base_param = _find_param(s.agent, "linear_1.base_layer.weight")
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

        _, lora_post = _find_param(s.agent, "lora_A.actor.weight")
        assert torch.allclose(lora_post, torch.full_like(lora_post, lora_sentinel)), (
            f"LoRA weight not restored for cell "
            f"(lora_only={s.lora_only}, save_optimizer={s.save_optimizer})"
        )
        if not s.lora_only:
            _, base_post = _find_param(s.agent, "linear_1.base_layer.weight")
            assert torch.allclose(
                base_post, torch.full_like(base_post, base_sentinel)
            ), (
                f"base weight not restored for cell "
                f"(lora_only={s.lora_only}, save_optimizer={s.save_optimizer})"
            )

    def test_optimizer_state_roundtrip(self, plain_load_scenario):
        s = plain_load_scenario
        sentinel, clobber = 0.3333, 9.9999

        # Populate optimizer state: fake grads → step.
        for p in s.agent.actor.parameters():
            if p.requires_grad:
                p.grad = torch.ones_like(p)
        s.agent.optimizer.step()
        s.agent.optimizer.zero_grad()

        exp_avg = _find_exp_avg(s.agent)
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
            restored = _find_exp_avg(s.agent)
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
            post = _find_exp_avg(s.agent)
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
def deepspeed_saved(request, grpo_template, tmp_path_factory):
    """Spy-wrapped DeepSpeed save per cell. Session-scoped — 4 deepcopies of
    the template, each saved once."""
    lora_only, save_optimizer = request.param
    agent = copy.deepcopy(grpo_template)
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


class TestDeepspeedSave:
    """Dispatch-only spy tests: replace ``actor.save_checkpoint`` with a
    MagicMock and assert the right branch was called with the right kwargs.

    LIMITATION: these do NOT verify DeepSpeed actually wrote correct bytes
    to disk — the real engine is mocked out. For end-to-end confidence see
    ``TestDeepspeedSaveE2E`` below (CUDA-only, @pytest.mark.llm).
    These spy tests run on any machine and catch dispatch regressions.
    """

    def test_deepspeed_save_called_iff_save_optimizer(self, deepspeed_saved):
        spy = deepspeed_saved.save_checkpoint_spy
        if deepspeed_saved.save_optimizer:
            assert spy.call_count == 1
            kwargs = spy.call_args.kwargs
            assert kwargs.get("exclude_frozen_parameters") == deepspeed_saved.lora_only
        else:
            assert spy.call_count == 0

    def test_save_pretrained_called_iff_lora_only(self, deepspeed_saved):
        spy = deepspeed_saved.save_pretrained_spy
        assert (spy.call_count >= 1) == deepspeed_saved.lora_only

    def test_attributes_pt_has_actor_state_dict_only_when_full_no_optim(
        self,
        deepspeed_saved,
    ):
        ck = _load_attributes_pt(deepspeed_saved.path)
        ni = ck.get("network_info", {}) or {}
        modules = ni.get("modules", {}) if isinstance(ni, dict) else {}
        has_actor_sd = "actor_state_dict" in modules
        expected = (not deepspeed_saved.lora_only) and (
            not deepspeed_saved.save_optimizer
        )
        assert has_actor_sd == expected


# --------------------------------------------------------------------------- #
# LOAD — deepspeed path                                                        #
# --------------------------------------------------------------------------- #


@pytest.fixture(params=SAVE_LOAD_OPTIONS)
def deepspeed_load_scenario(request, grpo_template, tmp_path):
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
    saver = copy.deepcopy(grpo_template)
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
    loader = copy.deepcopy(grpo_template)
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

    def test_deepspeed_load_called_iff_save_optimizer(self, deepspeed_load_scenario):
        s = deepspeed_load_scenario
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
        deepspeed_load_scenario,
    ):
        s = deepspeed_load_scenario
        from unittest.mock import patch

        inner = _inner_actor(s.agent)
        with (
            patch.object(s.agent, "_load_model_checkpoint") as peft_spy,
            patch.object(inner, "load_state_dict"),
        ):
            s.agent.load_checkpoint(str(s.path), load_optimizer=s.save_optimizer)
        # peft load is entered only for (lora_only=T, save_optim=F).
        expected = s.lora_only and not s.save_optimizer
        assert (peft_spy.call_count == 1) == expected

    def test_state_dict_load_when_full_and_no_optim(self, deepspeed_load_scenario):
        s = deepspeed_load_scenario
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


class TestGatherIfZero3:
    """Sanity-check that gather_if_zero3 is entered when zero_stage=3.

    Two save cells gather: the save_pretrained branch (lora_only=True) and
    the full torch-save branch (lora_only=False, save_optim=False). One test
    per branch — no full-grid parametrisation needed.
    """

    def test_gather_entered_on_peft_save_when_zero3(self, grpo_template, tmp_path):
        from contextlib import contextmanager
        from unittest.mock import patch

        agent = copy.deepcopy(grpo_template)
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

    def test_gather_entered_on_full_save_when_zero3(self, grpo_template, tmp_path):
        from contextlib import contextmanager
        from unittest.mock import patch

        agent = copy.deepcopy(grpo_template)
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
# save/load pipeline end-to-end: files on disk, real engine, real state
# restored. They complement the spy tests above — spies catch dispatch
# regressions on any machine; E2E catches "did DeepSpeed actually do it"
# bugs on CUDA-capable machines.
#
# Pulls fixtures/config from:
#   - tests/conftest.py                   → deepspeed_env (env vars)
#   - tests/test_algorithms/test_llms/conftest.py → accelerator_factory
#   - tests/test_algorithms/test_llms/test_grpo.py → deepspeed_config_stage_2

# Import here (not top-of-file) so the spy-based tests can still run on
# machines without vllm/deepspeed installed. Guarded so import failures
# cause the relevant tests to skip cleanly.
try:
    from tests.test_algorithms.test_llms.test_grpo import deepspeed_config_stage_2

    _E2E_IMPORT_ERR: Exception | None = None
except Exception as _e:  # pragma: no cover
    deepspeed_config_stage_2 = None  # type: ignore[assignment]
    _E2E_IMPORT_ERR = _e


def _require_cuda_deepspeed() -> None:
    """Skip if the environment can't run real DeepSpeed."""
    import torch

    if _E2E_IMPORT_ERR is not None:
        pytest.skip(f"E2E deepspeed imports unavailable: {_E2E_IMPORT_ERR!r}")
    if not torch.cuda.is_available():
        pytest.skip("E2E deepspeed tests require CUDA")
    pytest.importorskip("deepspeed", reason="E2E deepspeed tests require deepspeed.")


# E2E fixtures are FUNCTION-scoped: the autouse ``cleanup_after_test`` in
# test_llms/conftest.py clears the Accelerator singleton after every test, so
# a session-scoped agent would be handed out with a wrecked Accelerator.


@pytest.fixture(params=SAVE_LOAD_OPTIONS)
def deepspeed_saved_e2e(
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
    # TODO: accelerator = accelerator_factory(use_deepspeed_optimizer=False,
    #                                          config=deepspeed_config_stage_2)
    #   agent = _build_grpo(accelerator=accelerator)  # wrap=True implicitly
    #   agent.save_checkpoint(str(tmp_path), lora_only=lora_only,
    #                         save_optimizer=save_optimizer)
    #   Return SimpleNamespace(agent, path=tmp_path, lora_only, save_optimizer).
    pytest.skip("TODO: implement E2E deepspeed save fixture")


@pytest.mark.llm
class TestDeepspeedSaveE2E:
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

    def test_save_artifacts_match_cell(self, deepspeed_saved_e2e):
        # TODO: truth-table over deepspeed_saved_e2e.lora_only / .save_optimizer:
        #   1) attributes.pt always exists.
        #   2) (<path>/save_checkpoint/ dir with mp_rank_* files) exists iff save_optimizer.
        #   3) (<path>/actor/adapter_model.safetensors) exists iff lora_only
        #      (and same for reference/).
        #   4) attributes.pt contents: actor_state_dict present iff (not lora_only
        #      and not save_optimizer); _lora_only flag matches.
        pytest.skip("TODO")


@pytest.fixture(params=SAVE_LOAD_OPTIONS)
def deepspeed_load_scenario_e2e(
    request,
    deepspeed_env,
    accelerator_factory,
    tmp_path,
):
    """Fresh real-DeepSpeed agent per load test.

    Function-scoped — each test stamps sentinels then saves+loads, so they
    cannot share state.
    """
    _require_cuda_deepspeed()
    lora_only, save_optimizer = request.param
    # TODO: accelerator = accelerator_factory(...); agent = _build_grpo(accelerator)
    #   Return SimpleNamespace(agent, path=tmp_path, lora_only, save_optimizer).
    pytest.skip("TODO: implement E2E deepspeed load scenario fixture")


@pytest.mark.llm
class TestDeepspeedLoadE2E:
    """Real DeepSpeed roundtrip: stamp sentinels → save → fresh agent → load
    → assert sentinels restored. One parametrised test per concern
    (weights / optimizer) to keep the real DeepSpeed builds bounded."""

    def test_adapter_and_base_weight_roundtrip_e2e(self, deepspeed_load_scenario_e2e):
        # TODO: stamp LoRA weight with sentinel A. When not lora_only, also
        #   stamp base-model linear_1.weight with sentinel B. Real save.
        #   Build a fresh accelerator + agent (factory call; Accelerator
        #   singleton reset handled by cleanup). Real load. Assert sentinels.
        #   NB: for ZeRO ≥ 2 the live weights are sharded — gather via
        #   gather_if_zero3 (or the plain stage-2 analogue) before comparing.
        pytest.skip("TODO")

    def test_optimizer_state_roundtrip_e2e(self, deepspeed_load_scenario_e2e):
        # TODO: step optimizer once to populate state; stamp exp_avg of one
        #   tracked param with a sentinel. Save. Fresh agent. Load with
        #   load_optimizer=deepspeed_load_scenario_e2e.save_optimizer.
        #     save_optimizer=True  → sentinel restored in optimizer state
        #     save_optimizer=False → optimizer state is fresh (empty);
        #                            a UserWarning may be emitted.
        #   DeepSpeed optimizer state lives inside the engine — access via
        #   engine.optimizer.state_dict() or the unwrapped optimizer.
        pytest.skip("TODO")
