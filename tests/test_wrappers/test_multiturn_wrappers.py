"""Tests for TokenObservationWrapper sliding-window prompt fields."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from agilerl.llm_envs import (
    FormatRewardWrapper,
    SearchTool,
    SyncMultiTurnVecEnv,
    TokenObservationWrapper,
    Trajectory,
    TrajectoryBuffer,
)


class _StubTokenizer:
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "x" * len(ids)


def _bare_wrapper() -> TokenObservationWrapper:
    return TokenObservationWrapper.__new__(TokenObservationWrapper)


def test_build_model_prompt_fields_no_truncation() -> None:
    w = _bare_wrapper()
    w.tokenizer = _StubTokenizer()
    w._initial_prompt_len = 3
    w.turn_boundaries = []
    w.full_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    out = w.build_model_prompt_fields(max_prompt_tokens=100)
    assert torch.equal(out["trajectory_input_ids"], w.full_ids)
    assert out["stitch_prefix_ids"].shape[1] == 0
    assert out["initial_prompt_len"] == 3


def test_sliding_window_reconstructs_full_sequence() -> None:
    """After dropping oldest post-initial turns, stitch + trunc[:I] + trunc[I:] equals full."""
    w = _bare_wrapper()
    w.tokenizer = _StubTokenizer()
    w._initial_prompt_len = 2
    # [init0,init1 | gen0a,gen0b | fb0a,fb0b | gen1a,gen1b | tail0,tail1]
    w.full_ids = torch.tensor(
        [[0, 1, 10, 11, 20, 21, 40, 41, 50, 51]], dtype=torch.long
    )
    w.turn_boundaries = [
        (2, 4, 0),
        (6, 8, 1),
    ]
    out = w.build_model_prompt_fields(max_prompt_tokens=6)
    trunc = out["trajectory_input_ids"]
    stitch = out["stitch_prefix_ids"]
    il = out["initial_prompt_len"]
    assert il == 2
    merged = torch.cat([trunc[:, :il], stitch, trunc[:, il:]], dim=1)
    assert torch.equal(merged, w.full_ids)


def test_max_prompt_tokens_for_sliding_window() -> None:
    from agilerl.utils.llm_utils import max_prompt_tokens_for_sliding_window

    assert max_prompt_tokens_for_sliding_window(8192, 1024) == 8192 - 1024
    assert max_prompt_tokens_for_sliding_window(100, 50) == 50
    assert max_prompt_tokens_for_sliding_window(100, None) == 99


def test_middle_stitch_tensor_layout_matches_vllm_colocate() -> None:
    """Same cat layout as LLMAlgorithm._generate_with_vllm_colocate post-process."""
    il = 2
    stitch = torch.tensor([[30, 31]], dtype=torch.long)
    block = torch.tensor([[0, 1, 10, 11, 100, 101]], dtype=torch.long)
    merged = torch.cat([block[:, :il], stitch, block[:, il:]], dim=1)
    expected = torch.tensor([[0, 1, 30, 31, 10, 11, 100, 101]], dtype=torch.long)
    assert torch.equal(merged, expected)


class _ChrTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"

    def __call__(self, texts, **kwargs):
        ids = [[ord(c) for c in texts[0]]]
        t = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}

    def encode(self, s: str) -> list[int]:
        return [ord(c) for c in s]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(int(x)) for x in ids)


class _RecordingGemEnv:
    def __init__(self) -> None:
        self.last_gen: str | None = None

    def reset(self):
        return "hello", {}

    def step(self, gen_text: str):
        self.last_gen = gen_text
        return "", 1.0, True, False, {}


def test_reset_returns_tuple_with_text_and_sets_prompt_len() -> None:
    inner = _RecordingGemEnv()
    w = TokenObservationWrapper(
        inner,
        _ChrTokenizer(),
        max_turns=3,
        pad_id=None,
        apply_chat_template=False,
    )
    obs, info = w.reset()
    assert isinstance(info, dict)
    assert set(obs.keys()) >= {"input_ids", "attention_mask", "text"}
    assert obs["text"] == "hello"
    assert w._last_full_prompt_token_len == obs["input_ids"].shape[1]


def test_step_from_full_completion_slices_generation() -> None:
    inner = _RecordingGemEnv()
    w = TokenObservationWrapper(
        inner,
        _ChrTokenizer(),
        max_turns=3,
        pad_id=None,
        apply_chat_template=False,
    )
    obs, _ = w.reset()
    pl = obs["input_ids"].shape[1]
    gen_ids = torch.tensor([[ord("x"), ord("y")]], dtype=torch.long)
    full = torch.cat([obs["input_ids"], gen_ids], dim=1)
    _pd, _r, term, _trunc, _i = w.step(full)
    assert inner.last_gen == "xy"
    assert term is True
    assert _pd == {}


def test_policy_observation_merges_sliding_window_when_max_model_len_set() -> None:
    w = _bare_wrapper()
    w.tokenizer = _StubTokenizer()
    w._sw_max_model_len = 100
    w._sw_max_output_tokens = 10
    w._initial_prompt_len = 2
    w.full_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    w.turn_boundaries = []
    obs = w._policy_observation_from_state()
    assert "trajectory_input_ids" in obs
    assert "trajectory_text" in obs
    assert obs["input_ids"].shape[1] == 4


class _SyncStubEnv:
    def __init__(self, sw_max_model_len: int | None = None) -> None:
        self._sw_max_model_len = sw_max_model_len
        self.turn_boundaries: list[int] = []
        self.reset_calls: list[int | None] = []
        self.close_calls = 0

    def reset(self, seed: int | None = None):
        self.reset_calls.append(seed)
        return (
            {
                "input_ids": torch.ones(1, 3, dtype=torch.long),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
            },
            {},
        )

    def step(self, full_completion: torch.Tensor):
        del full_completion
        self.turn_boundaries.append(1)
        return ({}, 1.0, True, False, {})

    def close(self) -> None:
        self.close_calls += 1

    def get_episode_data(self):
        return (
            torch.ones(1, 4, dtype=torch.long),
            torch.ones(1, 3, dtype=torch.bool),
            torch.zeros(1, 3, dtype=torch.long),
            torch.ones(2, dtype=torch.float32),
        )


def test_sync_gem_vec_env_reset_seeds_per_batch_group() -> None:
    vec_env = SyncMultiTurnVecEnv(
        env_factory=lambda: _SyncStubEnv(),
        batch_size=2,
        group_size=2,
    )
    _ = vec_env.reset(seed=10)
    seen = [traj.env.reset_calls[-1] for traj in vec_env.trajectories]
    assert seen == [10, 10, 11, 11]


def test_sync_gem_vec_env_reset_with_none_seed() -> None:
    vec_env = SyncMultiTurnVecEnv(
        env_factory=lambda: _SyncStubEnv(),
        batch_size=2,
        group_size=2,
    )
    _ = vec_env.reset(seed=None)
    seen = [traj.env.reset_calls[-1] for traj in vec_env.trajectories]
    assert seen == [None, None, None, None]


def test_sync_gem_vec_env_step_raises_when_completion_count_mismatches_active() -> None:
    vec_env = SyncMultiTurnVecEnv(
        env_factory=lambda: _SyncStubEnv(),
        batch_size=1,
        group_size=2,
    )
    _ = vec_env.reset(seed=0)
    with pytest.raises(
        RuntimeError,
        match="Number of completions does not match number of active trajectories",
    ):
        vec_env.step([torch.ones(1, 5, dtype=torch.long)])


def test_sync_vec_env_close_calls_underlying_env_close_once() -> None:
    vec_env = SyncMultiTurnVecEnv(
        env_factory=lambda: _SyncStubEnv(sw_max_model_len=1024),
        batch_size=2,
        group_size=2,
    )
    _ = vec_env.reset(seed=0)
    vec_env.close()
    close_counts = [traj.env.close_calls for traj in vec_env.trajectories]
    assert close_counts == [1, 1, 1, 1]


def test_sync_vec_env_constructor_rejects_non_positive_sizes() -> None:
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        _ = SyncMultiTurnVecEnv(
            env_factory=lambda: _SyncStubEnv(),
            batch_size=0,
            group_size=1,
        )
    with pytest.raises(ValueError, match="group_size must be > 0"):
        _ = SyncMultiTurnVecEnv(
            env_factory=lambda: _SyncStubEnv(),
            batch_size=1,
            group_size=0,
        )


def test_trajectory_buffer_reset_trajectory_out_of_bounds() -> None:
    buf = TrajectoryBuffer(batch_size=1, group_size=1)
    with pytest.raises(IndexError, match="env_idx out of bounds"):
        buf.reset_trajectory(seed=0, env_idx=0)


class _ChatTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        del tokenize, add_generation_prompt
        content = messages[0]["content"]
        return [100] + [ord(c) for c in content]

    def __call__(self, texts, **kwargs):
        del kwargs
        ids = [[ord(c) for c in texts[0]]]
        t = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}

    def encode(self, s: str) -> list[int]:
        return [ord(c) for c in s]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "|".join(str(int(x)) for x in ids)


class _SeedlessResetEnv:
    def reset(self):
        return "obs", {}

    def step(self, gen_text: str):
        del gen_text
        return "done", 0.0, True, False, {}


class _NonTerminalEnv:
    def __init__(self) -> None:
        self.calls = 0
        self.last_gen: str | None = None

    def reset(self, seed: int | None = None):
        del seed
        return "hello", {"prefix": "P:", "suffix": "S"}

    def step(self, gen_text: str):
        self.calls += 1
        self.last_gen = gen_text
        if self.calls == 1:
            return "feedback", 0.5, False, False, {"prefix": "F:", "suffix": "T"}
        return "done", 1.0, True, False, {}


def test_format_obs_prefix_suffix_and_empty_info() -> None:
    assert TokenObservationWrapper._format_obs("x", None) == "x"
    assert (
        TokenObservationWrapper._format_obs("x", {"prefix": "A:", "suffix": "B"})
        == "A:x\nB"
    )


def test_policy_observation_raises_without_full_ids() -> None:
    w = _bare_wrapper()
    w.full_ids = None
    with pytest.raises(RuntimeError, match="No prompt"):
        w._policy_observation_from_state()


def test_step_raises_without_prior_policy_observation() -> None:
    w = _bare_wrapper()
    w._last_full_prompt_token_len = None
    with pytest.raises(RuntimeError, match="requires a prior reset"):
        w.step(torch.ones(1, 2, dtype=torch.long))


def test_reset_seed_fallback_for_seedless_env() -> None:
    w = TokenObservationWrapper(
        _SeedlessResetEnv(),
        _ChrTokenizer(),
        max_turns=1,
        pad_id=None,
        apply_chat_template=False,
    )
    obs, _ = w.reset(seed=123)
    assert "input_ids" in obs


def test_chat_template_paths_and_nonterminal_step_feedback_append() -> None:
    env = _NonTerminalEnv()
    w = TokenObservationWrapper(
        env,
        _ChatTokenizer(),
        max_turns=2,
        pad_id=None,
        apply_chat_template=True,
        max_model_len=32,
        max_output_tokens=4,
    )
    obs, _ = w.reset()
    assert obs["input_ids"].dtype == torch.long
    assert "trajectory_input_ids" in obs

    completion = torch.cat(
        [obs["input_ids"], torch.tensor([[7, 8]], dtype=torch.long)],
        dim=1,
    )
    next_obs, reward, terminated, truncated, _ = w.step(completion)
    assert reward == 0.5
    assert not terminated and not truncated
    assert env.last_gen == "7|8"
    assert next_obs["input_ids"].shape[1] > completion.shape[1]
    assert w._feedback_texts[-1] == "F:feedback\nT"


def test_build_model_prompt_fields_errors() -> None:
    w = _bare_wrapper()
    w.tokenizer = _StubTokenizer()
    w.full_ids = None
    with pytest.raises(RuntimeError, match="No prompt"):
        w.build_model_prompt_fields(max_prompt_tokens=8)

    w.full_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    w._initial_prompt_len = 3
    w.turn_boundaries = []
    with pytest.raises(RuntimeError, match="Initial prompt"):
        w.build_model_prompt_fields(max_prompt_tokens=2)

    w.full_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    w._initial_prompt_len = 2
    w.turn_boundaries = []
    with pytest.raises(RuntimeError, match="Could not fit prompt"):
        w.build_model_prompt_fields(max_prompt_tokens=5)


def test_build_model_prompt_fields_drop_from_ge_seq_len_branch() -> None:
    w = _bare_wrapper()
    w.tokenizer = _StubTokenizer()
    w._initial_prompt_len = 2
    w.full_ids = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.long)
    w.turn_boundaries = [(5, 5, 0)]
    out = w.build_model_prompt_fields(max_prompt_tokens=4)
    assert torch.equal(
        out["trajectory_input_ids"],
        torch.tensor([[10, 11]], dtype=torch.long),
    )


def test_get_episode_data_padding_and_pad_mask() -> None:
    w = _bare_wrapper()
    w.pad_id = 0
    w.max_turns = 3
    w.full_ids = torch.tensor([[9, 5, 0, 7, 8]], dtype=torch.long)
    w.turn_boundaries = [(1, 3, 0), (3, 5, 1)]
    w.turn_rewards = [1.5]
    full_ids, action_mask, turn_ids, rewards = w.get_episode_data()
    assert torch.equal(full_ids, w.full_ids)
    assert action_mask.dtype == torch.bool
    assert turn_ids.dtype == torch.long
    assert rewards.tolist() == [1.5, 0.0, 0.0]
    assert action_mask[0, 1].item() is False
    assert turn_ids[0, 1].item() == -1


def test_get_episode_data_raises_without_reset() -> None:
    w = _bare_wrapper()
    w.full_ids = None
    with pytest.raises(RuntimeError, match="No episode data"):
        w.get_episode_data()


def test_get_debug_info_paths() -> None:
    w = _bare_wrapper()
    w.full_ids = None
    assert w.get_debug_info() == {"error": "No episode data"}

    w = _bare_wrapper()
    w.tokenizer = _ChrTokenizer()
    w.pad_id = None
    w.max_turns = 2
    w.full_ids = torch.tensor([[65, 66, 67, 68]], dtype=torch.long)
    w.turn_boundaries = [(2, 4, 0)]
    w.turn_rewards = [0.5]
    w._gen_texts = ["AB"]
    w._feedback_texts = ["fb"]
    w._prompt_text = "prompt"
    info = w.get_debug_info()
    assert info["n_turns"] == 1
    assert info["turn_rewards_padded"] == [0.5, 0.0]
    assert info["feedback_texts"] == ["fb"]
    assert info["turn_details"][0]["gen_len"] == 2


def test_search_tool_parse_action_and_instruction() -> None:
    tool = SearchTool(search_url="http://x")
    query, parsed_action, valid = tool._parse_action("a<search> q </search>z")
    assert (query, parsed_action, valid) == ("q", "a<search> q </search>", True)
    assert tool._parse_action("no tags") == ("", "", False)
    assert "<answer>" in tool.instruction_string()


def test_search_tool_search_success_and_failure_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = SearchTool(search_url="http://x", topk=1, timeout=1)

    class _Resp:
        def json(self):
            return {"results": [{"content": "first"}, {"content": "second"}]}

    def _ok_get(url, params, timeout):
        del url, params, timeout
        return _Resp()

    monkeypatch.setattr("agilerl.llm_envs.requests.get", _ok_get)
    out = tool._search("hello")
    assert "first" in out and "second" not in out

    def _fail_get(url, params, timeout):
        del url, params, timeout
        raise RuntimeError("boom")

    monkeypatch.setattr("agilerl.llm_envs.requests.get", _fail_get)
    assert "[SearchTool Error:" in tool._search("hello")

    no_url = SearchTool(search_url=None)
    monkeypatch.delenv("SEARCH_URL", raising=False)
    with pytest.raises(ValueError, match="search_url must be provided"):
        no_url._search("x")


def test_search_tool_passages_to_string_and_execute_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = SearchTool(search_url="http://x")
    passages = [
        {"document": {"contents": "Title A\nBody A"}},
        {"document": {"contents": "Title B\nBody B"}},
    ]
    formatted = tool._passages2string(passages)
    assert "Doc 1(Title: Title A) Body A" in formatted

    valid, has_error, observation, parsed_action = tool.execute_action("nope")
    assert (valid, has_error, observation, parsed_action) == (False, True, "", "")

    monkeypatch.setattr(tool, "_search", lambda q: f"res:{q}")
    valid, has_error, observation, parsed_action = tool.execute_action(
        "<search>cats</search> trailing",
    )
    assert valid is True and has_error is False
    assert "<information>res:cats</information>" in observation
    assert parsed_action == "<search>cats</search>"


class _FormatEnv:
    def __init__(self):
        self.state = "ok"

    def reset(self, **kwargs):
        return ("obs", kwargs)

    def step(self, action: str, **kwargs):
        del kwargs
        return ("next", 1.0, True, False, {"correct": False, "action": action})


def test_format_reward_wrapper_branches_and_passthrough() -> None:
    env = _FormatEnv()
    wrapped = FormatRewardWrapper(env, format_bonus=0.3)
    assert wrapped.format_bonus == 0.3
    assert wrapped.state == "ok"
    obs, rew, term, trunc, info = wrapped.step("<answer>bad</answer>")
    assert (obs, term, trunc) == ("next", True, False)
    assert info["correct"] is False
    assert rew == 1.3
    assert wrapped.reset(seed=7) == ("obs", {"seed": 7})

    class _NoBonusEnv(_FormatEnv):
        def step(self, action: str, **kwargs):
            del action, kwargs
            return ("next", 2.0, True, False, {"correct": True})

    no_bonus = FormatRewardWrapper(_NoBonusEnv(), format_bonus=0.5)
    assert no_bonus.step("<answer>good</answer>")[1] == 2.0


def test_trajectory_buffer_invariants_and_helpers() -> None:
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        _ = TrajectoryBuffer(batch_size=0, group_size=1)
    with pytest.raises(ValueError, match="group_size must be > 0"):
        _ = TrajectoryBuffer(batch_size=1, group_size=0)

    env = _SyncStubEnv()
    t1 = Trajectory(
        env=env,
        batch_idx=1,
        group_idx=0,
        prompt={
            "input_ids": torch.ones(1, 1, dtype=torch.long),
            "attention_mask": torch.ones(1, 1, dtype=torch.long),
        },
        done=False,
    )
    t2 = Trajectory(
        env=env,
        batch_idx=0,
        group_idx=1,
        prompt={
            "input_ids": torch.ones(1, 1, dtype=torch.long),
            "attention_mask": torch.ones(1, 1, dtype=torch.long),
        },
        done=True,
    )
    buf = TrajectoryBuffer(batch_size=1, group_size=2)
    buf.add_trajectory(t1)
    buf.add_trajectory(t2)
    assert buf.is_initialized is True
    assert buf.has_active() is True
    assert len(buf) == 2
    assert list(iter(buf))[0] is t1
    assert buf[1] is t2
    buf.sort(key=lambda t: (t.batch_idx, t.group_idx))
    assert [t.batch_idx for t in buf] == [0, 1]
    buf.clear()
    assert len(buf) == 0 and buf.has_active() is False


def test_trajectory_buffer_get_active_trajectories_sorting() -> None:
    env = _SyncStubEnv()
    t0 = Trajectory(
        env=env,
        batch_idx=1,
        group_idx=0,
        prompt={
            "input_ids": torch.ones(1, 1, dtype=torch.long),
            "attention_mask": torch.ones(1, 1, dtype=torch.long),
        },
        done=False,
    )
    t1 = Trajectory(
        env=env,
        batch_idx=0,
        group_idx=1,
        prompt={
            "input_ids": torch.ones(1, 1, dtype=torch.long),
            "attention_mask": torch.ones(1, 1, dtype=torch.long),
        },
        done=False,
    )
    t2 = Trajectory(
        env=env,
        batch_idx=0,
        group_idx=0,
        prompt={
            "input_ids": torch.ones(1, 1, dtype=torch.long),
            "attention_mask": torch.ones(1, 1, dtype=torch.long),
        },
        done=True,
    )
    buf = TrajectoryBuffer(batch_size=2, group_size=2)
    buf.add_trajectory(t0)
    buf.add_trajectory(t1)
    buf.add_trajectory(t2)

    unsorted_active = buf.get_active_trajectories(sorted_by_index=False)
    assert unsorted_active == [t0, t1]
    sorted_active = buf.get_active_trajectories(sorted_by_index=True)
    assert sorted_active == [t1, t0]


def test_trajectory_buffer_get_prompts_returns_none_when_no_active() -> None:
    env = _SyncStubEnv()
    done_traj = Trajectory(
        env=env,
        batch_idx=0,
        group_idx=0,
        prompt={
            "input_ids": torch.ones(1, 2, dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        },
        done=True,
    )
    active_traj = Trajectory(
        env=env,
        batch_idx=0,
        group_idx=1,
        prompt={
            "input_ids": torch.ones(1, 3, dtype=torch.long),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        },
        done=False,
    )
    buf = TrajectoryBuffer(batch_size=1, group_size=2)
    buf.add_trajectory(done_traj)
    buf.add_trajectory(active_traj)

    prompts = buf.get_prompts()
    assert prompts is not None
    assert isinstance(prompts, list)
    assert len(prompts) == 1
    assert prompts[0]["input_ids"].shape == (1, 3)
    assert prompts[0]["attention_mask"].shape == (1, 3)

    active_traj.done = True
    assert buf.get_prompts() is None


def test_trajectory_buffer_stack_prompt_validation() -> None:
    env = _SyncStubEnv()
    a = Trajectory(
        env=env,
        batch_idx=0,
        group_idx=0,
        prompt={
            "input_ids": torch.ones(1, 2, dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "trajectory_input_ids": torch.ones(1, 2, dtype=torch.long),
            "trajectory_attention_mask": torch.ones(1, 2, dtype=torch.long),
            "stitch_prefix_ids": torch.ones(1, 1, dtype=torch.long),
            "initial_prompt_len": 2,
        },
        done=False,
    )
    b = Trajectory(
        env=env,
        batch_idx=0,
        group_idx=1,
        prompt={
            "input_ids": torch.ones(1, 3, dtype=torch.long),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "trajectory_input_ids": torch.ones(1, 3, dtype=torch.long),
            "trajectory_attention_mask": torch.ones(1, 3, dtype=torch.long),
            "stitch_prefix_ids": torch.ones(1, 2, dtype=torch.long),
            "initial_prompt_len": 3,
        },
        done=False,
    )
    buf = TrajectoryBuffer(batch_size=1, group_size=2)
    buf.add_trajectory(a)
    buf.add_trajectory(b)
    prompts = buf.get_prompts()
    assert prompts is not None
    assert len(prompts) == 2
    assert [int(p["initial_prompt_len"]) for p in prompts] == [2, 3]
    assert prompts[0]["input_ids"].shape == (1, 2)
    assert prompts[1]["input_ids"].shape == (1, 3)


class _StepVariantEnv:
    def __init__(self, done_after_step: bool, include_turn_boundaries: bool = True):
        self.done_after_step = done_after_step
        self.include_turn_boundaries = include_turn_boundaries
        self.step_shapes: list[tuple[int, ...]] = []
        if include_turn_boundaries:
            self.turn_boundaries: list[int] = []

    def reset(self, seed: int | None = None):
        del seed
        return (
            {
                "input_ids": torch.ones(1, 3, dtype=torch.long),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
            },
            {},
        )

    def step(self, full_completion: torch.Tensor):
        self.step_shapes.append(tuple(full_completion.shape))
        if self.include_turn_boundaries:
            self.turn_boundaries.append(1)
        if self.done_after_step:
            return ({}, 1.0, True, False, {})
        return (
            {
                "input_ids": torch.ones(1, 4, dtype=torch.long),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
            },
            0.5,
            False,
            False,
            {},
        )

    def close(self) -> None:
        pass

    def get_episode_data(self):
        return (
            torch.ones(1, 5, dtype=torch.long),
            torch.ones(1, 4, dtype=torch.bool),
            torch.zeros(1, 4, dtype=torch.long),
            torch.ones(2, dtype=torch.float32),
        )


def test_sync_vec_env_step_happy_path_1d_and_2d_and_active_filtering() -> None:
    created = [
        _StepVariantEnv(done_after_step=False),
        _StepVariantEnv(done_after_step=True),
    ]
    idx = {"i": 0}

    def _factory():
        env = created[idx["i"]]
        idx["i"] += 1
        return env

    vec = SyncMultiTurnVecEnv(env_factory=_factory, batch_size=1, group_size=2)
    _ = vec.reset(seed=0)
    prompts = vec.step(
        [
            torch.tensor([1, 2, 3], dtype=torch.long),
            torch.tensor([[1, 2, 3]], dtype=torch.long),
        ]
    )
    assert prompts is not None
    assert isinstance(prompts, list)
    assert len(prompts) == 1
    assert prompts[0]["input_ids"].shape == (1, 4)
    assert created[0].step_shapes == [(1, 3)]
    assert created[1].step_shapes == [(1, 3)]


def test_sync_vec_env_get_trajectories_counts_steps_with_and_without_turn_boundaries() -> (
    None
):
    created = [
        _StepVariantEnv(done_after_step=True, include_turn_boundaries=True),
        _StepVariantEnv(done_after_step=True, include_turn_boundaries=False),
    ]
    idx = {"i": 0}

    def _factory():
        env = created[idx["i"]]
        idx["i"] += 1
        return env

    vec = SyncMultiTurnVecEnv(env_factory=_factory, batch_size=1, group_size=2)
    _ = vec.reset(seed=0)
    _ = vec.step(
        [
            torch.tensor([[1, 2, 3]], dtype=torch.long),
            torch.tensor([[1, 2, 3]], dtype=torch.long),
        ]
    )
    *_parts, batch_steps = vec.get_trajectories()
    assert batch_steps == 1


def test_sync_vec_env_close_dedupes_same_env_instance() -> None:
    shared = _SyncStubEnv()
    vec = SyncMultiTurnVecEnv(env_factory=lambda: shared, batch_size=2, group_size=2)
    _ = vec.reset(seed=0)
    vec.close()
    assert shared.close_calls == 1


def test_non_chat_feedback_tokenization_path() -> None:
    env = _NonTerminalEnv()
    w = TokenObservationWrapper(
        env,
        _ChrTokenizer(),
        max_turns=2,
        pad_id=None,
        apply_chat_template=False,
    )
    obs, _ = w.reset()
    completion = torch.cat(
        [obs["input_ids"], torch.tensor([[120, 121]], dtype=torch.long)],
        dim=1,
    )
    next_obs, _, terminated, truncated, _ = w.step(completion)
    assert not terminated and not truncated
    assert next_obs["input_ids"].shape[1] > completion.shape[1]


def test_trajectory_buffer_reset_trajectory_success_path() -> None:
    env = _SyncStubEnv()
    traj = Trajectory(
        env=env,
        batch_idx=0,
        group_idx=0,
        prompt={},
        done=True,
    )
    buf = TrajectoryBuffer(batch_size=1, group_size=1)
    buf.add_trajectory(traj)
    buf.reset_trajectory(seed=5, env_idx=0)
    assert buf[0].done is False
    assert buf[0].prompt["input_ids"].shape == (1, 3)
    assert env.reset_calls[-1] == 5


def test_sync_gem_vec_env_reset_reuses_existing_trajectories() -> None:
    vec_env = SyncMultiTurnVecEnv(
        env_factory=lambda: _SyncStubEnv(),
        batch_size=2,
        group_size=2,
    )
    _ = vec_env.reset(seed=10)
    _ = vec_env.reset(seed=20)
    seen = [traj.env.reset_calls[-1] for traj in vec_env.trajectories]
    assert seen == [20, 20, 21, 21]
