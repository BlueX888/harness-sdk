"""Regression tests for the sliding-window tool-pair trim fallback.

A trimmed history must remain a legal provider input: it starts with a user
message (GitHub #2085). The tool-pair fallback introduced for tool-heavy
histories must therefore retain the plain user message that anchored the
conversation, or decline to trim when no such anchor exists.
"""

import pytest

from strands.agent.agent import Agent
from strands.agent.conversation_manager.sliding_window_conversation_manager import (
    SlidingWindowConversationManager,
)
from strands.models import Model
from strands.types.exceptions import ContextWindowOverflowException


def _tool_use(tool_use_id: str) -> dict:
    return {"toolUse": {"toolUseId": tool_use_id, "name": "get_file", "input": {}}}


def _tool_result(tool_use_id: str) -> dict:
    return {"toolResult": {"toolUseId": tool_use_id, "content": [{"text": "data"}], "status": "success"}}


class _OverflowOnceModel(Model):
    """Raise one context overflow, then answer; records the roles each model call receives."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def update_config(self, **model_config: dict) -> None:
        pass

    def get_config(self) -> dict:
        return {}

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        raise NotImplementedError

    async def stream(self, messages, tool_specs=None, system_prompt=None, tool_choice=None, **kwargs):
        self.calls.append([message["role"] for message in messages])
        if len(self.calls) == 1:
            raise ContextWindowOverflowException(RuntimeError("Input is too long for requested model"))
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}
        yield {"contentBlockDelta": {"delta": {"text": "Done."}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}


def test_tool_pair_fallback_keeps_the_user_message_that_started_the_conversation():
    manager = SlidingWindowConversationManager(window_size=4, should_truncate_results=False)
    messages = [
        {"role": "user", "content": [{"text": "Review this PR"}]},
        {"role": "assistant", "content": [_tool_use("1")]},
        {"role": "user", "content": [_tool_result("1")]},
        {"role": "assistant", "content": [_tool_use("2")]},
        {"role": "user", "content": [_tool_result("2")]},
        {"role": "assistant", "content": [_tool_use("3")]},
        {"role": "user", "content": [_tool_result("3")]},
        {"role": "assistant", "content": [{"text": "Here is my review"}]},
    ]
    agent = Agent(messages=messages)

    manager.reduce_context(agent, e=Exception("context window overflow"))

    assert [message["role"] for message in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[0]["content"] == [{"text": "Review this PR"}]
    assert messages[1]["content"][0]["toolUse"]["toolUseId"] == "3"


def test_tool_pair_fallback_declines_without_a_plain_user_anchor():
    manager = SlidingWindowConversationManager(window_size=4, should_truncate_results=False)
    messages = [
        {"role": "assistant", "content": [_tool_use("1")]},
        {"role": "user", "content": [_tool_result("1")]},
        {"role": "assistant", "content": [_tool_use("2")]},
        {"role": "user", "content": [_tool_result("2")]},
        {"role": "assistant", "content": [{"text": "Here is my review"}]},
    ]
    agent = Agent(messages=messages)

    with pytest.raises(ContextWindowOverflowException):
        manager.reduce_context(agent, e=Exception("context window overflow"))


def test_tool_pair_fallback_with_pin_first_keeps_the_pinned_prefix_and_alternation():
    manager = SlidingWindowConversationManager(window_size=3, pin_first=1, should_truncate_results=False)
    messages = [
        {"role": "user", "content": [{"text": "hi"}]},
        {"role": "assistant", "content": [{"text": "hello"}]},
        {"role": "user", "content": [{"text": "do task"}]},
        {"role": "assistant", "content": [_tool_use("1")]},
        {"role": "user", "content": [_tool_result("1")]},
        {"role": "assistant", "content": [_tool_use("2")]},
        {"role": "user", "content": [_tool_result("2")]},
    ]
    agent = Agent(messages=messages)

    manager.reduce_context(agent, e=Exception("context window overflow"))

    roles = [message["role"] for message in messages]
    assert roles == ["user", "assistant", "user"]
    assert all(left != right for left, right in zip(roles, roles[1:], strict=False))
    assert messages[0]["content"] == [{"text": "hi"}]
    assert messages[0]["metadata"]["custom"]["pinned"] is True


def test_tool_pair_fallback_declines_when_the_pinned_prefix_is_not_user_first():
    manager = SlidingWindowConversationManager(window_size=3, pin_first=1, should_truncate_results=False)
    messages = [
        {"role": "assistant", "content": [_tool_use("1")]},
        {"role": "user", "content": [_tool_result("1")]},
        {"role": "assistant", "content": [_tool_use("2")]},
        {"role": "user", "content": [_tool_result("2")]},
        {"role": "assistant", "content": [{"text": "Here is my review"}]},
    ]
    agent = Agent(messages=messages)

    with pytest.raises(ContextWindowOverflowException):
        manager.reduce_context(agent, e=Exception("context window overflow"))


@pytest.mark.asyncio
async def test_reactive_overflow_retry_with_pin_first_sends_a_valid_alternating_history():
    model = _OverflowOnceModel()
    manager = SlidingWindowConversationManager(window_size=3, pin_first=1, should_truncate_results=False)
    messages = [
        {"role": "user", "content": [{"text": "hi"}]},
        {"role": "assistant", "content": [{"text": "hello"}]},
        {"role": "user", "content": [{"text": "do task"}]},
        {"role": "assistant", "content": [_tool_use("1")]},
        {"role": "user", "content": [_tool_result("1")]},
        {"role": "assistant", "content": [_tool_use("2")]},
        {"role": "user", "content": [_tool_result("2")]},
        {"role": "assistant", "content": [{"text": "ready"}]},
    ]
    agent = Agent(model=model, conversation_manager=manager, messages=messages)

    await agent.invoke_async()

    retried_roles = model.calls[1]
    assert retried_roles == ["user", "assistant", "user", "assistant"]
    assert all(left != right for left, right in zip(retried_roles, retried_roles[1:], strict=False))
    assert agent.messages[0]["content"] == [{"text": "hi"}]
    assert agent.messages[0]["metadata"]["custom"]["pinned"] is True
