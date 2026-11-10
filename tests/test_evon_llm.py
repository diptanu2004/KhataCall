"""Evon adapter against a mocked OpenAI-compatible endpoint. No network."""

import asyncio
import json

import httpx
import pytest

from app.adapters.base import LLM, Message
from app.adapters.evon_llm import EvonError, EvonLLM

SYSTEM = Message("system", "Tum ek kirana dukaan ke assistant ho.")
USER = Message("user", "sharma ji ka dedh sau udhaar likho")

ADD_CREDIT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "add_credit",
        "description": "Record credit against a customer",
        "parameters": {
            "type": "object",
            "properties": {"customer_name": {"type": "string"}, "amount_spoken": {"type": "string"}},
            "required": ["customer_name", "amount_spoken"],
        },
    },
}


def chat_response(*, content=None, tool_calls=None) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"index": 0, "message": message, "finish_reason": "stop"}]}


def native_tool_call(name: str, arguments, call_id: str = "call_1") -> dict:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": raw}}


def build(responses: list, **kwargs) -> tuple[EvonLLM, list[httpx.Request]]:
    seen: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        nxt = queue.pop(0) if queue else chat_response(content="")
        return nxt if isinstance(nxt, httpx.Response) else httpx.Response(200, json=nxt)

    kwargs.setdefault("base_url", "http://localhost:8001/v1")
    llm = EvonLLM(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), **kwargs)
    return llm, seen


def run(llm: EvonLLM, tools=()):
    return asyncio.run(llm.complete([SYSTEM, USER], tools=tools))


# --- request shape ---------------------------------------------------------


def test_posts_openai_shaped_request_with_thinking_disabled():
    llm, seen = build([chat_response(content="Ji, boliye.")], model="evon-test", api_key="k")
    run(llm)

    request = seen[0]
    body = json.loads(request.content)
    assert str(request.url) == "http://localhost:8001/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer k"
    assert body["model"] == "evon-test"
    assert body["messages"] == [
        {"role": "system", "content": SYSTEM.content},
        {"role": "user", "content": USER.content},
    ]
    # Reasoning tokens would wreck spoken latency, so thinking is off by default.
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_omits_auth_header_when_no_key_is_set():
    """A local vLLM server usually needs no key."""
    llm, seen = build([chat_response(content="ok")])
    run(llm)
    assert "Authorization" not in seen[0].headers


def test_extra_body_is_overridable():
    llm, seen = build([chat_response(content="ok")], extra_body={"reasoning_effort": "low"})
    run(llm)
    body = json.loads(seen[0].content)
    assert body["reasoning_effort"] == "low"
    assert "chat_template_kwargs" not in body


def test_native_mode_sends_tools_in_the_request():
    llm, seen = build([chat_response(content="ok")])
    run(llm, tools=[ADD_CREDIT_SCHEMA])
    body = json.loads(seen[0].content)
    assert body["tools"] == [ADD_CREDIT_SCHEMA]
    assert body["tool_choice"] == "auto"


def test_prompt_mode_puts_schemas_in_the_system_message_instead():
    llm, seen = build([chat_response(content="ok")], tool_mode="prompt")
    run(llm, tools=[ADD_CREDIT_SCHEMA])
    body = json.loads(seen[0].content)
    assert "tools" not in body
    system = body["messages"][0]
    assert system["role"] == "system"
    assert SYSTEM.content in system["content"]
    assert "add_credit" in system["content"]


# --- reply parsing ---------------------------------------------------------


def test_plain_text_reply():
    llm, _ = build([chat_response(content="Namaste, kya likhna hai?")])
    reply = run(llm)
    assert reply.text == "Namaste, kya likhna hai?"
    assert reply.tool_calls == ()
    assert isinstance(llm, LLM)


def test_native_tool_call_is_parsed_with_json_string_arguments():
    args = {"customer_name": "Sharma ji", "amount_spoken": "dedh sau"}
    llm, _ = build([chat_response(tool_calls=[native_tool_call("add_credit", args)])])

    reply = run(llm, tools=[ADD_CREDIT_SCHEMA])

    assert len(reply.tool_calls) == 1
    call = reply.tool_calls[0]
    assert (call.name, call.arguments, call.id) == ("add_credit", args, "call_1")
    # Amounts must reach us as spoken text so nlu/numbers.py parses them (hard rule 7).
    assert call.arguments["amount_spoken"] == "dedh sau"


def test_tool_call_with_object_arguments_is_also_accepted():
    """Some servers return arguments already decoded."""
    args = {"customer_name": "Gupta aunty", "amount_spoken": "do hazaar"}
    llm, _ = build([chat_response(tool_calls=[{"id": "c", "function": {"name": "add_credit", "arguments": args}}])])
    assert run(llm, tools=[ADD_CREDIT_SCHEMA]).tool_calls[0].arguments == args


def test_tool_call_with_malformed_arguments_is_dropped_not_guessed():
    llm, _ = build([chat_response(content=None, tool_calls=[native_tool_call("add_credit", "{not json")])])
    reply = run(llm, tools=[ADD_CREDIT_SCHEMA])
    # Dropped, so the agent re-asks rather than writing a guessed amount.
    assert reply.tool_calls == ()


def test_prompt_mode_parses_a_json_object_from_text():
    text = 'Sure.\n{"tool": "add_credit", "arguments": {"customer_name": "Sharma ji", "amount_spoken": "dedh sau"}}'
    llm, _ = build([chat_response(content=text)], tool_mode="prompt")

    reply = run(llm, tools=[ADD_CREDIT_SCHEMA])

    assert reply.tool_calls[0].name == "add_credit"
    assert reply.tool_calls[0].arguments["amount_spoken"] == "dedh sau"


def test_prompt_mode_retries_once_then_returns_text():
    llm, seen = build(
        [chat_response(content="I will add the credit."), chat_response(content="Still not JSON.")],
        tool_mode="prompt",
    )

    reply = run(llm, tools=[ADD_CREDIT_SCHEMA])

    assert len(seen) == 2, "should re-prompt exactly once"
    nudge = json.loads(seen[1].content)["messages"][-1]
    assert "only the JSON object" in nudge["content"]
    assert reply.tool_calls == ()
    assert reply.text == "I will add the credit."


def test_prompt_mode_retry_succeeds():
    good = '{"tool": "add_credit", "arguments": {"customer_name": "Sharma ji", "amount_spoken": "sau"}}'
    llm, seen = build([chat_response(content="ok sure"), chat_response(content=good)], tool_mode="prompt")
    reply = run(llm, tools=[ADD_CREDIT_SCHEMA])
    assert len(seen) == 2
    assert reply.tool_calls[0].arguments["amount_spoken"] == "sau"


def test_native_mode_does_not_retry():
    """A native-tool-calling server that returns text meant to return text."""
    llm, seen = build([chat_response(content="Sharma ji ka balance kitna?")])
    run(llm, tools=[ADD_CREDIT_SCHEMA])
    assert len(seen) == 1


def test_native_mode_arguments_written_as_text_are_never_spoken():
    """Seen Nov 10: real Evon wrote a call's arguments as the reply, with no tool name."""
    text = '{"customer_name": "Mehra ji", "amount_spoken": "saadhe chhe sau", "note": "bill"}'
    llm, seen = build([chat_response(content=text)])
    reply = run(llm, tools=[ADD_CREDIT_SCHEMA])
    assert reply.text is None and reply.tool_calls == ()  # the session says "phir se boliye"
    assert len(seen) == 1


def test_native_mode_a_whole_call_written_as_text_is_used():
    text = '{"tool": "add_credit", "arguments": {"customer_name": "Sharma ji", "amount_spoken": "dedh sau"}}'
    llm, _ = build([chat_response(content=text)])
    reply = run(llm, tools=[ADD_CREDIT_SCHEMA])
    assert reply.tool_calls[0].name == "add_credit"
    assert reply.tool_calls[0].arguments["amount_spoken"] == "dedh sau"


# --- failures --------------------------------------------------------------


def test_http_error_raises_with_server_message():
    llm, _ = build([httpx.Response(400, json={"error": {"message": "context length exceeded"}})])
    with pytest.raises(EvonError, match="context length exceeded") as err:
        run(llm)
    assert err.value.status == 400


def test_a_dead_tunnel_says_so_instead_of_passing_on_html():
    """An ended Kaggle session: Cloudflare answers 530 with an HTML page (Nov 10)."""
    page = "<!doctype html>\n<!--[if lt IE 7]> <html class=\"no-js ie6 oldie\" lang=\"en-US\"> <![endif]-->"
    llm, _ = build([httpx.Response(530, text=page, headers={"content-type": "text/html"})])
    with pytest.raises(EvonError, match="tunnel is down") as err:
        run(llm)
    assert err.value.status == 530
    assert "<" not in str(err.value)


def test_unreachable_endpoint_names_the_url():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    llm = EvonLLM(
        base_url="http://localhost:8001/v1",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(EvonError, match="could not reach Evon at http://localhost:8001/v1"):
        run(llm)


def test_empty_choices_raises():
    llm, _ = build([{"choices": []}])
    with pytest.raises(EvonError, match="no choices"):
        run(llm)


def test_rejects_unknown_tool_mode():
    with pytest.raises(ValueError, match="tool_mode"):
        EvonLLM(base_url="http://x/v1", tool_mode="magic")


# --- connection reuse (D-046) ------------------------------------------------


def test_warm_opens_the_connection_with_a_health_request():
    llm, seen = build([httpx.Response(200, json={"status": "ok"})], api_key="k")
    asyncio.run(llm.warm())
    assert [str(r.url) for r in seen] == ["http://localhost:8001/health"]
    assert seen[0].headers["Authorization"] == "Bearer k"


def test_a_failed_warm_up_is_silent():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    llm = EvonLLM(base_url="http://localhost:8001/v1", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    asyncio.run(llm.warm())  # no exception: complete() is what reports an outage


def test_idle_connections_outlive_the_gap_between_turns():
    from app.adapters.evon_llm import KEEPALIVE_S

    llm = EvonLLM(base_url="http://localhost:8001/v1")
    pool = llm._client._transport._pool
    assert pool._keepalive_expiry == KEEPALIVE_S >= 30
    asyncio.run(llm.aclose())
