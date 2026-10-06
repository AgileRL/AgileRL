# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Env action text extracted from a generation."""

from agilerl.llm_envs.harness import RolloutHarness, env_action_text


class TestEnvActionText:
    def test_keeps_a_bare_action(self) -> None:
        assert env_action_text("click('12')") == "click('12')"

    def test_drops_reasoning_from_a_prefilled_open_think(self) -> None:
        text = "The lamp is $5.\nSubmit it.</think>\n\nsend_msg_to_user('$5')"

        assert env_action_text(text) == "send_msg_to_user('$5')"

    def test_drops_a_leading_closed_think_block(self) -> None:
        text = "<think>\nlook at bid 12\n</think>\nclick('12')"

        assert env_action_text(text) == "click('12')"

    def test_drops_an_empty_think_block(self) -> None:
        assert env_action_text("<think></think>click('1996')") == "click('1996')"

    def test_keeps_the_text_after_the_first_close(self) -> None:
        text = "first</think>click('3')</think>"

        assert env_action_text(text) == "click('3')</think>"

    def test_passes_unclosed_reasoning_through(self) -> None:
        text = "The page lists three lamps and the cheapest is"

        assert env_action_text(text) == text

    def test_reasoning_with_no_action_sends_nothing(self) -> None:
        assert env_action_text("<think>only reasoning</think>") == ""

    def test_quotes_an_unparsed_call(self) -> None:
        assert (
            env_action_text("send_msg_to_user($14.47-$23.50)")
            == "send_msg_to_user('$14.47-$23.50')"
        )

    def test_quotes_an_unparsed_call_after_reasoning(self) -> None:
        assert (
            env_action_text("range found</think>\nsend_msg_to_user($14.47-$23.50)")
            == "send_msg_to_user('$14.47-$23.50')"
        )

    def test_quotes_a_colon_call(self) -> None:
        assert (
            env_action_text("send_msg_to_user: $14.47-$23.50")
            == "send_msg_to_user('$14.47-$23.50')"
        )

    def test_unquotes_a_quoted_colon_payload(self) -> None:
        assert (
            env_action_text('send_msg_to_user: "$5 off"')
            == "send_msg_to_user('$5 off')"
        )

    def test_passes_a_call_with_no_payload_through(self) -> None:
        assert env_action_text("noop(") == "noop("

    def test_keeps_a_parsed_call(self) -> None:
        assert env_action_text("fill('55', 'red')") == "fill('55', 'red')"


class TestTrailingInstruction:
    def test_leaves_the_question_after_the_instruction(self) -> None:
        harness = RolloutHarness.__new__(RolloutHarness)
        harness._system_prompt = "Answer now."

        rendered = harness._with_trailing_instruction("goal\n\npage\n\nQuestion:\ngoal")

        assert rendered == "goal\n\npage\nAnswer now.\n\nQuestion:\ngoal"
