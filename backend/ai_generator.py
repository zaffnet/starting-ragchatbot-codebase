import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **Up to 2 searches per query** — use a second search only if the first didn't fully answer the question or you need information from a different course/lesson
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to MAX_TOOL_ROUNDS of tool calls before forcing a text response.
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [{"role": "user", "content": query}]

        for round_num in range(self.MAX_TOOL_ROUNDS):
            api_params = {**self.base_params, "messages": messages, "system": system_content}
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            response = self.client.messages.create(**api_params)

            if response.stop_reason != "tool_use" or not tool_manager:
                return self._extract_text(response)

            # Execute tools, accumulate messages
            messages, tool_failed = self._execute_tool_round(messages, response, tool_manager)
            if tool_failed:
                break

        # Exhausted rounds or tool failed — final call WITHOUT tools
        final_params = {**self.base_params, "messages": messages, "system": system_content}
        final_response = self.client.messages.create(**final_params)
        return self._extract_text(final_response)

    def _execute_tool_round(self, messages, response, tool_manager):
        """
        Execute tool calls from a response, append results to messages.

        Returns:
            (updated_messages, tool_failed) tuple
        """
        messages = messages.copy()
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        tool_failed = False

        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name,
                        **content_block.input
                    )
                except Exception as e:
                    tool_result = f"Error executing tool: {e}"
                    tool_failed = True

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        return messages, tool_failed

    def _extract_text(self, response):
        """Extract the first text block from a response."""
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""
