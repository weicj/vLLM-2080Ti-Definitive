# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import (
    Function as FunctionCallTool,
)
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputItem
from openai.types.responses.response import ToolChoice
from openai.types.responses.response_function_tool_call_output_item import (
    ResponseFunctionToolCallOutputItem,
)
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_reasoning_item import ResponseReasoningItem
from openai.types.responses.tool import Tool

from vllm import envs
from vllm.entrypoints.constants import MCP_PREFIX
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionMessageParam
from vllm.entrypoints.openai.responses.protocol import ResponseInputOutputItem
from vllm.logger import init_logger

logger = init_logger(__name__)


def should_continue_final_message(
    request_input: str | list[ResponseInputOutputItem],
) -> bool:
    """
    Determine if the last input message is a partial assistant message
    that should be continued rather than starting a new generation.

    This enables partial message completion similar to Anthropic's Messages API,
    where users can provide an incomplete assistant message and have the model
    continue from where it left off.

    A message is considered partial if:
    1. It's a ResponseOutputMessage or ResponseReasoningItem
    2. Its status is "in_progress" or "incomplete"

    Args:
        request_input: The input to the Responses API request

    Returns:
        True if the final message should be continued, False otherwise
    """
    if isinstance(request_input, str):
        # Simple string input is always a user message
        return False

    if not request_input:
        return False

    last_item = request_input[-1]

    # Check if the last item is a partial assistant message
    if isinstance(last_item, ResponseOutputMessage):
        return last_item.status in ("in_progress", "incomplete")

    # Check if the last item is a partial reasoning item
    if isinstance(last_item, ResponseReasoningItem):
        return last_item.status in ("in_progress", "incomplete")

    if isinstance(last_item, dict):
        # only support partial completion for messages for now
        if last_item.get("type", "message") not in ("message", "reasoning"):
            return False
        return last_item.get("status") in ("in_progress", "incomplete")

    return False


def construct_input_messages(
    *,
    request_instructions: str | None = None,
    request_input: str | list[ResponseInputOutputItem],
    prev_msg: list[ChatCompletionMessageParam] | None = None,
    prev_response_output: list[ResponseOutputItem] | None = None,
):
    if isinstance(request_input, str):
        input_messages: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": request_input}
        ]
    else:
        input_messages = construct_chat_messages_with_tool_call(request_input)

    # [local 2080ti fork] Clients such as codex send the system prompt BOTH
    # in top-level `instructions` AND as a role="developer" input message.
    # Naively emitting both as system messages yields [system, system, user,
    # ...], which Qwen-family chat templates reject ("System message must be
    # at the beginning."). When the input already opens with the system
    # message converted from role="developer" and there is no earlier
    # server-side history, merge `instructions` into it.
    lead: ChatCompletionMessageParam | None = None
    if request_instructions:
        if (
            prev_msg is None
            and prev_response_output is None
            and input_messages
            and input_messages[0].get("role") == "system"
        ):
            first = input_messages[0]
            old_content = first.get("content")
            if isinstance(old_content, str):
                if old_content:
                    first["content"] = f"{request_instructions}\n{old_content}"
                else:
                    first["content"] = request_instructions
            elif isinstance(old_content, list):
                # Non-string content (e.g. it still carries non-text parts
                # such as images) must not be replaced: prepend the
                # instructions as the first text part so the original parts
                # are preserved.
                if not old_content:
                    first["content"] = request_instructions
                else:
                    first["content"] = [
                        {"type": "text", "text": request_instructions}
                    ] + list(old_content)
            else:
                first["content"] = request_instructions
        else:
            lead = {"role": "system", "content": request_instructions}

    messages: list[ChatCompletionMessageParam] = []
    if lead is not None:
        messages.append(lead)
    # Prepend the conversation history.
    if prev_msg is not None:
        # Filter out system messages from previous conversation -- per the
        # OpenAI spec, instructions should NOT carry over across responses.
        messages.extend(m for m in prev_msg if m.get("role") != "system")
    if prev_response_output is not None:
        # Add the previous output.
        for output_item in prev_response_output:
            # NOTE: We skip the reasoning output.
            if isinstance(output_item, ResponseOutputMessage):
                for content in output_item.content:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content.text,
                        }
                    )

    # Append the new input (the leading system message is already merged in
    # via `input_messages` when applicable).
    messages.extend(input_messages)
    return messages


def _maybe_combine_reasoning_and_tool_call(
    item: ResponseInputOutputItem, messages: list[ChatCompletionMessageParam]
) -> ChatCompletionMessageParam | None:
    """Many models treat MCP calls and reasoning as a single message.
    This function checks if the last message is a reasoning message and
    the current message is a tool call"""
    if not (
        isinstance(item, ResponseFunctionToolCall)
        and item.id
        and item.id.startswith(MCP_PREFIX)
    ):
        return None
    if len(messages) == 0:
        return None
    last_message = messages[-1]
    if not (
        last_message.get("role") == "assistant"
        and last_message.get("reasoning") is not None
    ):
        return None

    last_message["tool_calls"] = [
        ChatCompletionMessageToolCallParam(
            id=item.call_id,
            function=FunctionCallTool(
                name=item.name,
                arguments=item.arguments,
            ),
            type="function",
        )
    ]
    return last_message


def construct_chat_messages_with_tool_call(
    input_messages: list[ResponseInputOutputItem],
) -> list[ChatCompletionMessageParam]:
    """This function wraps _construct_single_message_from_response_item
    Because some chatMessages come from multiple response items
    for example a reasoning item and a MCP tool call are two response items
    but are one chat message
    """
    messages: list[ChatCompletionMessageParam] = []
    for item in input_messages:
        maybe_combined_message = _maybe_combine_reasoning_and_tool_call(item, messages)
        if maybe_combined_message is not None:
            messages[-1] = maybe_combined_message
        else:
            messages.append(_construct_single_message_from_response_item(item))

    return messages


def _construct_single_message_from_response_item(
    item: ResponseInputOutputItem,
) -> ChatCompletionMessageParam:
    if isinstance(item, ResponseFunctionToolCall):
        # Append the function call as a tool call.
        return ChatCompletionAssistantMessageParam(
            role="assistant",
            tool_calls=[
                ChatCompletionMessageToolCallParam(
                    id=item.call_id,
                    function=FunctionCallTool(
                        name=item.name,
                        arguments=item.arguments,
                    ),
                    type="function",
                )
            ],
        )
    elif isinstance(item, ResponseReasoningItem):
        reasoning = ""
        if item.encrypted_content:
            raise ValueError("Encrypted content is not supported.")
        elif item.content and len(item.content) >= 1:
            reasoning = item.content[0].text
        elif len(item.summary) >= 1:
            reasoning = item.summary[0].text
            logger.warning(
                "Using summary text as reasoning content for item %s. "
                "Please use content instead of summary for "
                "reasoning items.",
                item.id,
            )
        return {
            "role": "assistant",
            "reasoning": reasoning,
        }
    elif isinstance(item, ResponseOutputMessage):
        return {
            "role": "assistant",
            "content": item.content[0].text,
        }
    elif isinstance(item, ResponseFunctionToolCallOutputItem):
        return ChatCompletionToolMessageParam(
            role="tool",
            content=item.output,
            tool_call_id=item.call_id,
        )
    elif isinstance(item, dict) and item.get("type") == "function_call_output":
        # Append the function call output as a tool message.
        return ChatCompletionToolMessageParam(
            role="tool",
            content=item.get("output"),
            tool_call_id=item.get("call_id"),
        )
    # A missing `type` defaults to "message" in the Responses API, so
    # untyped message dicts (as some codex-style clients send them) must
    # go through the same normalization; otherwise role="developer" would
    # slip through unconverted and be rejected by chat templates.
    elif isinstance(item, dict) and item.get("type", "message") == "message":
        # [local 2080ti fork] Normalize Responses-API input messages to
        # chat-template roles. Clients such as codex submit system context
        # as role="developer" (the OpenAI-recommended alias of "system"),
        # which Qwen-family chat templates reject with
        # "Unexpected message role.". Map it to "system" (same aliasing the
        # chat-completions path performs) and flatten input_text parts.
        role = item.get("role") or "user"
        if role == "developer":
            role = "system"
        content = item.get("content")
        if isinstance(content, list):
            texts: list[str] = []
            kept: list[dict] = []
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type")
                    in ("input_text", "output_text", "text")
                ):
                    if part.get("text"):
                        texts.append(part["text"])
                else:
                    # Non-text parts (images, files, ...): pass through
                    # untouched for now; codex-style text traffic never
                    # carries them.
                    kept.append(part)
            if not kept:
                # Newline-join (same separator as the instructions merge
                # above) so part boundaries are preserved in the prompt
                # instead of "first" + "second" becoming "firstsecond".
                content = "\n".join(texts)
            else:
                content = [{"type": "text", "text": t} for t in texts] + kept
        msg: dict = {"role": role, "content": content}
        if item.get("name") and role != "system":
            msg["name"] = item.get("name")
        return msg
    return item  # type: ignore


def extract_tool_types(tools: list[Tool]) -> set[str]:
    """
    Extracts the tool types from the given tools.
    """
    tool_types: set[str] = set()
    for tool in tools:
        if tool.type == "mcp":
            # Allow the MCP Tool type to enable built in tools if the
            # server_label is allowlisted in
            # envs.VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS
            if tool.server_label in envs.VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS:
                tool_types.add(tool.server_label)
        else:
            tool_types.add(tool.type)
    return tool_types


def convert_tool_responses_to_completions_format(tool: dict) -> dict:
    """
    Convert a flat tool schema:
        {"type": "function", "name": "...", "description": "...", "parameters": {...}}
    into:
        {"type": "function", "function": {...}}
    """
    return {
        "type": "function",
        "function": tool,
    }


def construct_tool_dicts(
    tools: list[Tool], tool_choice: ToolChoice
) -> list[dict[str, Any]] | None:
    if not tools or (tool_choice == "none"):
        tool_dicts = None
    else:
        tool_dicts = [
            convert_tool_responses_to_completions_format(tool.model_dump())
            for tool in tools
        ]
    return tool_dicts
