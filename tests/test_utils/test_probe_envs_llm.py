import agilerl.utils.probe_envs_llm as probe_envs_llm


class TestConstantTargetEnvReset:
    def test_returns_prompt_and_empty_info(self):
        env = probe_envs_llm.ConstantTargetEnv(target_digit="3", prompt="11")
        obs, info = env.reset()
        assert obs == "11"
        assert info == {}

    def test_updates_seed_when_seed_is_provided(self):
        env = probe_envs_llm.ConstantTargetEnv(seed=1)
        env.reset(seed=99)
        assert env.seed == 99


class TestConstantTargetEnvStep:
    def test_returns_positive_reward_when_action_contains_target_digit(self):
        env = probe_envs_llm.ConstantTargetEnv(target_digit="3")
        obs, reward, terminated, truncated, info = env.step("abc3xyz")
        assert obs == ""
        assert reward == 1.0
        assert terminated is True
        assert truncated is False
        assert info == {}

    def test_returns_negative_reward_when_action_does_not_contain_target_digit(self):
        env = probe_envs_llm.ConstantTargetEnv(target_digit="3")
        _, reward, _, _, _ = env.step("999")
        assert reward == -1.0


class TestConditionalTargetEnvReset:
    def test_returns_digit_observation_and_target_info(self):
        env = probe_envs_llm.ConditionalTargetEnv(digits=(1, 2, 3), seed=0)
        obs, info = env.reset()
        assert obs in {"1", "2", "3"}
        assert "target" in info

    def test_target_matches_rule_mod_three_plus_one(self):
        env = probe_envs_llm.ConditionalTargetEnv(digits=(2,), seed=0)
        obs, info = env.reset()
        assert obs == "2"
        assert info["target"] == "3"
        assert env.target == "3"

    def test_seeded_resets_are_reproducible(self):
        env1 = probe_envs_llm.ConditionalTargetEnv(seed=123)
        env2 = probe_envs_llm.ConditionalTargetEnv(seed=123)
        obs1, info1 = env1.reset()
        obs2, info2 = env2.reset()
        assert obs1 == obs2
        assert info1 == info2

    def test_seed_argument_reinitializes_rng_sequence(self):
        env = probe_envs_llm.ConditionalTargetEnv(seed=0)
        obs1, info1 = env.reset(seed=7)
        obs2, info2 = env.reset(seed=7)
        assert obs1 == obs2
        assert info1 == info2


class TestConditionalTargetEnvStep:
    def test_returns_positive_reward_when_action_contains_current_target(self):
        env = probe_envs_llm.ConditionalTargetEnv(digits=(1,), seed=0)
        _, info = env.reset()
        _, reward, terminated, truncated, step_info = env.step(f"x{info['target']}y")
        assert reward == 1.0
        assert terminated is True
        assert truncated is False
        assert step_info == {}

    def test_returns_negative_reward_when_action_does_not_contain_current_target(self):
        env = probe_envs_llm.ConditionalTargetEnv(digits=(1,), seed=0)
        env.reset()
        _, reward, _, _, _ = env.step("9")
        assert reward == -1.0


class TestMultiInputConditionalEnvReset:
    def test_returns_two_digit_observation_and_target_info(self):
        env = probe_envs_llm.MultiInputConditionalEnv(digits=(1, 2, 3), seed=0)
        obs, info = env.reset()
        assert len(obs) == 2
        assert set(obs).issubset({"1", "2", "3"})
        assert "target" in info

    def test_target_matches_sum_rule_mod_three_plus_one(self):
        env = probe_envs_llm.MultiInputConditionalEnv(digits=(2,), seed=0)
        obs, info = env.reset()
        assert obs == "22"
        assert info["target"] == "2"
        assert env.target == "2"

    def test_seeded_resets_are_reproducible(self):
        env1 = probe_envs_llm.MultiInputConditionalEnv(seed=99)
        env2 = probe_envs_llm.MultiInputConditionalEnv(seed=99)
        obs1, info1 = env1.reset()
        obs2, info2 = env2.reset()
        assert obs1 == obs2
        assert info1 == info2

    def test_seed_argument_reinitializes_rng_sequence(self):
        env = probe_envs_llm.MultiInputConditionalEnv(seed=0)
        obs1, info1 = env.reset(seed=13)
        obs2, info2 = env.reset(seed=13)
        assert obs1 == obs2
        assert info1 == info2


class TestMultiInputConditionalEnvStep:
    def test_returns_positive_reward_when_action_contains_current_target(self):
        env = probe_envs_llm.MultiInputConditionalEnv(digits=(1,), seed=0)
        _, info = env.reset()
        _, reward, terminated, truncated, step_info = env.step(f"a{info['target']}b")
        assert reward == 1.0
        assert terminated is True
        assert truncated is False
        assert step_info == {}

    def test_returns_negative_reward_when_action_does_not_contain_current_target(self):
        env = probe_envs_llm.MultiInputConditionalEnv(digits=(1,), seed=0)
        env.reset()
        _, reward, _, _, _ = env.step("9")
        assert reward == -1.0


class TestGridNavigationEnvReset:
    def test_returns_obs_with_position_and_target_in_info(self):
        env = probe_envs_llm.GridNavigationEnv(grid_size=4, seed=0)
        obs, info = env.reset()
        assert obs == f"{info['position']}{info['target']}"
        assert 0 <= info["position"] < 4
        assert 0 <= info["target"] < 4

    def test_target_is_never_equal_to_start_position(self):
        env = probe_envs_llm.GridNavigationEnv(grid_size=4, seed=0)
        _, info = env.reset()
        assert info["position"] != info["target"]

    def test_seeded_resets_are_reproducible(self):
        env1 = probe_envs_llm.GridNavigationEnv(grid_size=4, seed=17)
        env2 = probe_envs_llm.GridNavigationEnv(grid_size=4, seed=17)
        obs1, info1 = env1.reset()
        obs2, info2 = env2.reset()
        assert obs1 == obs2
        assert info1 == info2

    def test_seed_argument_reinitializes_rng_sequence(self):
        env = probe_envs_llm.GridNavigationEnv(grid_size=4, seed=0)
        obs1, info1 = env.reset(seed=21)
        obs2, info2 = env.reset(seed=21)
        assert obs1 == obs2
        assert info1 == info2


class TestGridNavigationEnvStep:
    def test_moves_left_when_action_contains_one(self):
        env = probe_envs_llm.GridNavigationEnv(grid_size=4, max_turns=5, step_cost=-0.1)
        env.position = 2
        env.target = 3
        env.turn = 0
        obs, reward, terminated, truncated, info = env.step("1")
        assert obs == "1"
        assert reward == -0.1
        assert terminated is False
        assert truncated is False
        assert info == {}

    def test_moves_right_when_action_contains_three(self):
        env = probe_envs_llm.GridNavigationEnv(grid_size=4, max_turns=5, step_cost=-0.1)
        env.position = 1
        env.target = 0
        env.turn = 0
        obs, reward, terminated, _, _ = env.step("3")
        assert obs == "2"
        assert reward == -0.1
        assert terminated is False

    def test_stays_when_action_contains_two(self):
        env = probe_envs_llm.GridNavigationEnv(grid_size=4, max_turns=5, step_cost=-0.1)
        env.position = 1
        env.target = 3
        env.turn = 0
        obs, reward, terminated, _, _ = env.step("2")
        assert obs == "1"
        assert reward == -0.1
        assert terminated is False

    def test_ignores_actions_without_valid_digit_and_applies_step_cost(self):
        env = probe_envs_llm.GridNavigationEnv(grid_size=4, max_turns=5, step_cost=-0.2)
        env.position = 1
        env.target = 3
        env.turn = 0
        obs, reward, terminated, _, _ = env.step("abc")
        assert obs == "1"
        assert reward == -0.2
        assert terminated is False

    def test_uses_first_valid_digit_when_action_contains_multiple_digits(self):
        env = probe_envs_llm.GridNavigationEnv(grid_size=4, max_turns=5, step_cost=-0.1)
        env.position = 1
        env.target = 0
        env.turn = 0
        obs, _, _, _, _ = env.step("x31")
        assert obs == "2"

    def test_clamps_left_boundary_at_zero(self):
        env = probe_envs_llm.GridNavigationEnv(grid_size=4, max_turns=5, step_cost=-0.1)
        env.position = 0
        env.target = 3
        env.turn = 0
        obs, _, _, _, _ = env.step("1")
        assert obs == "0"

    def test_clamps_right_boundary_at_grid_max(self):
        env = probe_envs_llm.GridNavigationEnv(grid_size=4, max_turns=5, step_cost=-0.1)
        env.position = 3
        env.target = 0
        env.turn = 0
        obs, _, _, _, _ = env.step("3")
        assert obs == "3"

    def test_returns_success_reward_and_done_when_target_reached(self):
        env = probe_envs_llm.GridNavigationEnv(grid_size=4, max_turns=5, step_cost=-0.1)
        env.position = 1
        env.target = 2
        env.turn = 0
        obs, reward, terminated, truncated, info = env.step("3")
        assert obs == "2"
        assert reward == 1.0
        assert terminated is True
        assert truncated is False
        assert info == {"success": True}

    def test_returns_failure_reward_and_done_when_max_turns_reached(self):
        env = probe_envs_llm.GridNavigationEnv(grid_size=4, max_turns=1, step_cost=-0.1)
        env.position = 0
        env.target = 3
        env.turn = 0
        obs, reward, terminated, truncated, info = env.step("2")
        assert obs == "0"
        assert reward == -1.0
        assert terminated is True
        assert truncated is False
        assert info == {"success": False}

    def test_returns_step_cost_and_not_done_for_intermediate_steps(self):
        env = probe_envs_llm.GridNavigationEnv(grid_size=4, max_turns=3, step_cost=-0.3)
        env.position = 0
        env.target = 3
        env.turn = 0
        obs, reward, terminated, truncated, info = env.step("2")
        assert obs == "0"
        assert reward == -0.3
        assert terminated is False
        assert truncated is False
        assert info == {}


class TestProbeEnvsLlmModuleExports:
    def test_all_contains_expected_public_names(self):
        assert probe_envs_llm.__all__ == [
            "ConditionalTargetEnv",
            "ConstantTargetEnv",
            "GridNavigationEnv",
            "MultiInputConditionalEnv",
        ]
