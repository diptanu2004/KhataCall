"""Gnani Evon v3.3 over an OpenAI-compatible chat-completions API.

Evon is not on Gnani's API platform; it is open weights served by us
(Kaggle / vLLM — see DECISIONS.md D-001). Targeting the OpenAI shape means the
host can change without touching this code: hosted endpoint, vLLM,
llama-server, or a Hugging Face Inference Endpoint all speak it.

Two things are deliberately **not** assumed:

1. **Native tool calling.** If Evon supports it, the server returns
   `message.tool_calls` and we parse them. If it does not, `tool_mode="prompt"`
   puts the schemas in the system prompt and parses one JSON object out of the
   reply (CLAUDE.md §7). Both modes return the same `LLMReply`, so nothing
   upstream changes.
2. **The thinking flag.** Evon is a reasoning model, and reasoning tokens would
   wreck spoken latency. The vLLM convention is
   `chat_template_kwargs: {"enable_thinking": false}`, which is what a
   participant repo reports using. `# VERIFY:` against the model card once the
   weights land — hence `extra_body`, so it can be corrected by config.
"""

import json
import re
from collections.abc import Sequence
from typing import Any

import httpx

from app.adapters.base import LLMReply, Message, ToolCall

# VERIFY: vLLM's convention for Qwen3-style thinking control. Confirm on the Evon model card.
DEFAULT_EXTRA_BODY: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class EvonError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status

    def __str__(self) -> str:
        return (f"[HTTP {self.status}] " if self.status else "") + super().__str__()


def _tool_prompt(tools: Sequence[dict[str, Any]]) -> str:
    """System-prompt instructions for servers without native tool calling."""
    schemas = json.dumps([t.get("function", t) for t in tools], ensure_ascii=False, indent=2)
    return (
        "You have these tools:\n"
        f"{schemas}\n\n"
        "To use one, reply with exactly one JSON object and nothing else:\n"
        '{"tool": "<name>", "arguments": {...}}\n'
        "If no tool applies, reply with plain text instead."
    )


def _parse_prompted_tool_call(text: str) -> ToolCall | None:
    """Pull a {"tool": ..., "arguments": {...}} object out of a text reply."""
    match = _JSON_OBJECT.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    name = obj.get("tool") or obj.get("name")
    args = obj.get("arguments", obj.get("args", {}))
    if not isinstance(name, str) or not isinstance(args, dict):
        return None
    return ToolCall(name=name, arguments=args)


class EvonLLM:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        model: str = "gnani/gnani-evon-v3.3-30B-A3B",
        tool_mode: str = "native",
        temperature: float = 0.2,
        max_tokens: int = 300,
        extra_body: dict[str, Any] | None = None,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 60.0,
        retry_once_on_parse_failure: bool = True,
    ) -> None:
        if tool_mode not in ("native", "prompt"):
            raise ValueError(f"tool_mode must be 'native' or 'prompt', got {tool_mode!r}")
        self.model_id = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self._tool_mode = tool_mode
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._extra_body = DEFAULT_EXTRA_BODY if extra_body is None else extra_body
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._retry_once = retry_once_on_parse_failure

    def _body(self, messages: Sequence[Message], tools: Sequence[dict[str, Any]]) -> dict[str, Any]:
        payload: list[dict[str, str]] = [{"role": m.role, "content": m.content} for m in messages]
        body: dict[str, Any] = {
            "model": self.model_id,
            "messages": payload,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if tools:
            if self._tool_mode == "native":
                body["tools"] = list(tools)
                body["tool_choice"] = "auto"
            else:
                instructions = _tool_prompt(tools)
                if payload and payload[0]["role"] == "system":
                    payload[0] = {**payload[0], "content": payload[0]["content"] + "\n\n" + instructions}
                else:
                    payload.insert(0, {"role": "system", "content": instructions})
        return body | self._extra_body

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        try:
            response = await self._client.post(self._url, json=body, headers=headers)
        except httpx.HTTPError as e:
            raise EvonError(f"could not reach Evon at {self._url}: {e}") from e
        try:
            parsed = response.json()
        except ValueError:
            raise EvonError(response.text[:200] or "non-JSON response", status=response.status_code) from None
        if response.status_code != 200:
            error = parsed.get("error")
            message = error.get("message") if isinstance(error, dict) else str(error or "request failed")
            raise EvonError(message, status=response.status_code)
        return parsed

    async def complete(
        self, messages: Sequence[Message], *, tools: Sequence[dict[str, Any]] = ()
    ) -> LLMReply:
        body = self._body(messages, tools)
        reply = self._to_reply(await self._post(body), expect_tool=bool(tools))

        # One retry when a tool call was expected but the reply could not be parsed as one
        # (CLAUDE.md §7). A second failure is passed upstream as plain text to re-ask.
        if tools and self._retry_once and not reply.tool_calls and self._tool_mode == "prompt":
            nudged = dict(body)
            nudged["messages"] = [
                *body["messages"],
                {"role": "assistant", "content": reply.text or ""},
                {"role": "user", "content": "Reply with only the JSON object, no other text."},
            ]
            retried = self._to_reply(await self._post(nudged), expect_tool=True)
            if retried.tool_calls:
                return retried
        return reply

    def _to_reply(self, parsed: dict[str, Any], *, expect_tool: bool) -> LLMReply:
        choices = parsed.get("choices") or []
        if not choices:
            raise EvonError("response had no choices")
        message = choices[0].get("message") or {}
        text = message.get("content") or None

        calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            function = raw.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    continue  # malformed args: drop the call, let the agent re-ask
            if isinstance(arguments, dict) and function.get("name"):
                calls.append(ToolCall(name=function["name"], arguments=arguments, id=raw.get("id")))

        if not calls and expect_tool and self._tool_mode == "prompt" and text:
            if (parsed_call := _parse_prompted_tool_call(text)) is not None:
                return LLMReply(text=None, tool_calls=(parsed_call,))

        return LLMReply(text=text, tool_calls=tuple(calls))

    async def aclose(self) -> None:
        await self._client.aclose()
