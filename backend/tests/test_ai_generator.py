from unittest.mock import MagicMock, patch
import pytest
from ai_generator import AIGenerator

# ---------------------------------------------------------------------------
# Mock response helpers
# ---------------------------------------------------------------------------


def _text_block(text):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _tool_use_block(name="search_course_content", tool_id="tool_123", input_data=None):
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.id = tool_id
    block.input = input_data or {"query": "test"}
    return block


def _mock_response(content_blocks, stop_reason="end_turn"):
    resp = MagicMock()
    resp.content = content_blocks
    resp.stop_reason = stop_reason
    return resp


# ---------------------------------------------------------------------------
# Tests — Direct Answers
# ---------------------------------------------------------------------------


class TestAIGeneratorDirectAnswer:
    @patch("ai_generator.anthropic.Anthropic")
    def test_direct_answer_no_tools(self, MockAnthropic):
        client = MockAnthropic.return_value
        client.messages.create.return_value = _mock_response(
            [_text_block("Hello!")], stop_reason="end_turn"
        )

        gen = AIGenerator(api_key="fake", model="claude-sonnet-4-20250514")
        result = gen.generate_response(query="Hi")

        assert result == "Hello!"
        client.messages.create.assert_called_once()

    @patch("ai_generator.anthropic.Anthropic")
    def test_direct_answer_tools_available_but_unused(self, MockAnthropic):
        client = MockAnthropic.return_value
        client.messages.create.return_value = _mock_response(
            [_text_block("General answer")], stop_reason="end_turn"
        )

        gen = AIGenerator(api_key="fake", model="claude-sonnet-4-20250514")
        tools = [
            {"name": "search_course_content", "description": "...", "input_schema": {}}
        ]
        result = gen.generate_response(query="What is 2+2?", tools=tools)

        assert result == "General answer"
        call_kwargs = client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tool_choice"] == {"type": "auto"}


# ---------------------------------------------------------------------------
# Tests — Single Tool Round
# ---------------------------------------------------------------------------


class TestAIGeneratorSingleToolRound:
    @patch("ai_generator.anthropic.Anthropic")
    def test_single_tool_round_flow(self, MockAnthropic):
        """One tool call → text response = 2 API calls."""
        client = MockAnthropic.return_value

        first_resp = _mock_response([_tool_use_block()], stop_reason="tool_use")
        second_resp = _mock_response(
            [_text_block("Based on the search, the answer is X.")],
            stop_reason="end_turn",
        )
        client.messages.create.side_effect = [first_resp, second_resp]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "search result text"

        gen = AIGenerator(api_key="fake", model="claude-sonnet-4-20250514")
        tools = [
            {"name": "search_course_content", "description": "...", "input_schema": {}}
        ]
        result = gen.generate_response(
            query="What is ML?", tools=tools, tool_manager=tool_manager
        )

        assert result == "Based on the search, the answer is X."
        assert client.messages.create.call_count == 2
        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="test"
        )

    @patch("ai_generator.anthropic.Anthropic")
    def test_second_call_message_structure(self, MockAnthropic):
        """Verify the second API call includes assistant + tool_result messages."""
        client = MockAnthropic.return_value

        tool_block = _tool_use_block(tool_id="abc")
        first_resp = _mock_response([tool_block], stop_reason="tool_use")
        second_resp = _mock_response([_text_block("final")], stop_reason="end_turn")
        client.messages.create.side_effect = [first_resp, second_resp]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "result"

        gen = AIGenerator(api_key="fake", model="claude-sonnet-4-20250514")
        gen.generate_response(
            query="q",
            tools=[{"name": "t", "description": "", "input_schema": {}}],
            tool_manager=tool_manager,
        )

        second_call_kwargs = client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]

        # messages[0] = user, messages[1] = assistant (tool_use), messages[2] = user (tool_result)
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        tool_result = messages[2]["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "abc"
        assert tool_result["content"] == "result"

    @patch("ai_generator.anthropic.Anthropic")
    def test_second_call_has_tools(self, MockAnthropic):
        """Round 2 call still includes tools since MAX_TOOL_ROUNDS > 1."""
        client = MockAnthropic.return_value

        first_resp = _mock_response([_tool_use_block()], stop_reason="tool_use")
        second_resp = _mock_response([_text_block("done")], stop_reason="end_turn")
        client.messages.create.side_effect = [first_resp, second_resp]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "result"

        gen = AIGenerator(api_key="fake", model="claude-sonnet-4-20250514")
        gen.generate_response(
            query="q",
            tools=[{"name": "t", "description": "", "input_schema": {}}],
            tool_manager=tool_manager,
        )

        second_call_kwargs = client.messages.create.call_args_list[1][1]
        assert "tools" in second_call_kwargs


# ---------------------------------------------------------------------------
# Tests — Two Tool Rounds
# ---------------------------------------------------------------------------


class TestAIGeneratorTwoToolRounds:
    @patch("ai_generator.anthropic.Anthropic")
    def test_two_tool_rounds_flow(self, MockAnthropic):
        """Two tool calls → final text = 3 API calls."""
        client = MockAnthropic.return_value

        first_resp = _mock_response(
            [_tool_use_block(tool_id="t1")], stop_reason="tool_use"
        )
        second_resp = _mock_response(
            [_tool_use_block(tool_id="t2", input_data={"query": "second"})],
            stop_reason="tool_use",
        )
        final_resp = _mock_response(
            [_text_block("Combined answer.")], stop_reason="end_turn"
        )
        client.messages.create.side_effect = [first_resp, second_resp, final_resp]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "result"

        gen = AIGenerator(api_key="fake", model="claude-sonnet-4-20250514")
        tools = [
            {"name": "search_course_content", "description": "...", "input_schema": {}}
        ]
        result = gen.generate_response(
            query="Compare courses", tools=tools, tool_manager=tool_manager
        )

        assert result == "Combined answer."
        assert client.messages.create.call_count == 3
        assert tool_manager.execute_tool.call_count == 2

    @patch("ai_generator.anthropic.Anthropic")
    def test_two_rounds_message_accumulation(self, MockAnthropic):
        """Final call has 5 messages: user, asst, tool_result, asst, tool_result."""
        client = MockAnthropic.return_value

        first_resp = _mock_response(
            [_tool_use_block(tool_id="t1")], stop_reason="tool_use"
        )
        second_resp = _mock_response(
            [_tool_use_block(tool_id="t2")], stop_reason="tool_use"
        )
        final_resp = _mock_response([_text_block("done")], stop_reason="end_turn")
        client.messages.create.side_effect = [first_resp, second_resp, final_resp]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "r"

        gen = AIGenerator(api_key="fake", model="claude-sonnet-4-20250514")
        gen.generate_response(
            query="q",
            tools=[{"name": "t", "description": "", "input_schema": {}}],
            tool_manager=tool_manager,
        )

        final_call_kwargs = client.messages.create.call_args_list[2][1]
        messages = final_call_kwargs["messages"]

        assert len(messages) == 5
        assert messages[0]["role"] == "user"  # original query
        assert messages[1]["role"] == "assistant"  # tool_use 1
        assert messages[2]["role"] == "user"  # tool_result 1
        assert messages[3]["role"] == "assistant"  # tool_use 2
        assert messages[4]["role"] == "user"  # tool_result 2

    @patch("ai_generator.anthropic.Anthropic")
    def test_two_rounds_final_call_has_no_tools(self, MockAnthropic):
        """After exhausting MAX_TOOL_ROUNDS, the final call excludes tools."""
        client = MockAnthropic.return_value

        first_resp = _mock_response(
            [_tool_use_block(tool_id="t1")], stop_reason="tool_use"
        )
        second_resp = _mock_response(
            [_tool_use_block(tool_id="t2")], stop_reason="tool_use"
        )
        final_resp = _mock_response([_text_block("done")], stop_reason="end_turn")
        client.messages.create.side_effect = [first_resp, second_resp, final_resp]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "r"

        gen = AIGenerator(api_key="fake", model="claude-sonnet-4-20250514")
        gen.generate_response(
            query="q",
            tools=[{"name": "t", "description": "", "input_schema": {}}],
            tool_manager=tool_manager,
        )

        final_call_kwargs = client.messages.create.call_args_list[2][1]
        assert "tools" not in final_call_kwargs

    @patch("ai_generator.anthropic.Anthropic")
    def test_max_rounds_enforced(self, MockAnthropic):
        """Even if Claude keeps requesting tools, stops after MAX_TOOL_ROUNDS."""
        client = MockAnthropic.return_value

        tool_resp = _mock_response([_tool_use_block()], stop_reason="tool_use")
        final_resp = _mock_response(
            [_text_block("forced answer")], stop_reason="end_turn"
        )
        # Provide enough tool responses + final
        client.messages.create.side_effect = [tool_resp, tool_resp, final_resp]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "r"

        gen = AIGenerator(api_key="fake", model="claude-sonnet-4-20250514")
        result = gen.generate_response(
            query="q",
            tools=[{"name": "t", "description": "", "input_schema": {}}],
            tool_manager=tool_manager,
        )

        assert result == "forced answer"
        assert client.messages.create.call_count == 3  # 2 rounds + 1 final
        assert tool_manager.execute_tool.call_count == 2

    @patch("ai_generator.anthropic.Anthropic")
    def test_early_stop_no_tool_use_round_2(self, MockAnthropic):
        """Claude uses tool in round 1, answers directly in round 2 → 2 API calls."""
        client = MockAnthropic.return_value

        first_resp = _mock_response([_tool_use_block()], stop_reason="tool_use")
        second_resp = _mock_response(
            [_text_block("answer after one search")], stop_reason="end_turn"
        )
        client.messages.create.side_effect = [first_resp, second_resp]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "r"

        gen = AIGenerator(api_key="fake", model="claude-sonnet-4-20250514")
        result = gen.generate_response(
            query="q",
            tools=[{"name": "t", "description": "", "input_schema": {}}],
            tool_manager=tool_manager,
        )

        assert result == "answer after one search"
        assert client.messages.create.call_count == 2


# ---------------------------------------------------------------------------
# Tests — Error Handling
# ---------------------------------------------------------------------------


class TestAIGeneratorErrorHandling:
    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_exception_sends_error_and_terminates(self, MockAnthropic):
        """Exception → error string as tool_result, loop breaks, final call without tools."""
        client = MockAnthropic.return_value

        first_resp = _mock_response(
            [_tool_use_block(tool_id="t1")], stop_reason="tool_use"
        )
        final_resp = _mock_response([_text_block("recovered")], stop_reason="end_turn")
        client.messages.create.side_effect = [first_resp, final_resp]

        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = RuntimeError("connection failed")

        gen = AIGenerator(api_key="fake", model="claude-sonnet-4-20250514")
        result = gen.generate_response(
            query="q",
            tools=[{"name": "t", "description": "", "input_schema": {}}],
            tool_manager=tool_manager,
        )

        assert result == "recovered"
        assert client.messages.create.call_count == 2

        # Verify the error was passed as tool_result
        final_call_kwargs = client.messages.create.call_args_list[1][1]
        assert "tools" not in final_call_kwargs
        messages = final_call_kwargs["messages"]
        tool_result_content = messages[2]["content"][0]["content"]
        assert "Error executing tool" in tool_result_content
        assert "connection failed" in tool_result_content

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_error_string_flows_through(self, MockAnthropic):
        """Tool returns error string (not exception) → flows normally as tool_result."""
        client = MockAnthropic.return_value

        first_resp = _mock_response([_tool_use_block()], stop_reason="tool_use")
        second_resp = _mock_response(
            [_text_block("no results found")], stop_reason="end_turn"
        )
        client.messages.create.side_effect = [first_resp, second_resp]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "Tool 'x' not found"

        gen = AIGenerator(api_key="fake", model="claude-sonnet-4-20250514")
        result = gen.generate_response(
            query="q",
            tools=[{"name": "t", "description": "", "input_schema": {}}],
            tool_manager=tool_manager,
        )

        assert result == "no results found"

    @patch("ai_generator.anthropic.Anthropic")
    def test_no_tool_manager_returns_text(self, MockAnthropic):
        """tool_manager=None, stop_reason=tool_use → returns available text, no crash."""
        client = MockAnthropic.return_value

        resp = _mock_response(
            [_text_block("partial"), _tool_use_block()], stop_reason="tool_use"
        )
        client.messages.create.return_value = resp

        gen = AIGenerator(api_key="fake", model="claude-sonnet-4-20250514")
        result = gen.generate_response(
            query="q",
            tools=[{"name": "t", "description": "", "input_schema": {}}],
            tool_manager=None,
        )

        assert result == "partial"


# ---------------------------------------------------------------------------
# Tests — Params and Constants
# ---------------------------------------------------------------------------


class TestAIGeneratorParams:
    @patch("ai_generator.anthropic.Anthropic")
    def test_conversation_history_in_system_prompt(self, MockAnthropic):
        client = MockAnthropic.return_value
        client.messages.create.return_value = _mock_response(
            [_text_block("ok")], stop_reason="end_turn"
        )

        gen = AIGenerator(api_key="fake", model="claude-sonnet-4-20250514")
        gen.generate_response(
            query="q", conversation_history="User: hi\nAssistant: hello"
        )

        call_kwargs = client.messages.create.call_args[1]
        assert "Previous conversation:" in call_kwargs["system"]
        assert "User: hi" in call_kwargs["system"]

    @patch("ai_generator.anthropic.Anthropic")
    def test_no_history_system_prompt(self, MockAnthropic):
        client = MockAnthropic.return_value
        client.messages.create.return_value = _mock_response(
            [_text_block("ok")], stop_reason="end_turn"
        )

        gen = AIGenerator(api_key="fake", model="claude-sonnet-4-20250514")
        gen.generate_response(query="q")

        call_kwargs = client.messages.create.call_args[1]
        assert "Previous conversation:" not in call_kwargs["system"]

    @patch("ai_generator.anthropic.Anthropic")
    def test_base_params_applied(self, MockAnthropic):
        client = MockAnthropic.return_value
        client.messages.create.return_value = _mock_response(
            [_text_block("ok")], stop_reason="end_turn"
        )

        gen = AIGenerator(api_key="fake", model="claude-sonnet-4-20250514")
        gen.generate_response(query="q")

        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert call_kwargs["temperature"] == 0
        assert call_kwargs["max_tokens"] == 800

    def test_system_prompt_allows_two_searches(self):
        assert "2 searches" in AIGenerator.SYSTEM_PROMPT

    def test_max_tool_rounds_constant(self):
        assert AIGenerator.MAX_TOOL_ROUNDS == 2
