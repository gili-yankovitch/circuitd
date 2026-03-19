"""LLM chat clients with tool-calling support (Ollama and OpenAI)."""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from abc import ABC, abstractmethod
from typing import Any, Callable

import requests

from . import config
from .prompts import STATE_SUMMARY_PROMPT

logger = logging.getLogger(__name__)

# OpenAI / httpx expect valid UTF-8 and JSON. PDF/tool blobs may contain NULs or
# lone surrogates that confuse some HTTP stacks or middleboxes.
def _sanitize_openai_text(s: str) -> str:
    if not s:
        return s
    s = s.replace("\x00", "")
    try:
        s.encode("utf-8")
    except UnicodeEncodeError:
        s = s.encode("utf-8", errors="replace").decode("utf-8")
    return s


def _sanitize_openai_messages(messages: list[dict[str, Any]]) -> None:
    """Normalize message strings in place before each OpenAI HTTP request."""
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            m["content"] = _sanitize_openai_text(c)
        tcs = m.get("tool_calls")
        if not tcs:
            continue
        for tc in tcs:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                fn["arguments"] = _sanitize_openai_text(args)


def _tool_args_to_json_str(arguments: dict[str, Any]) -> str:
    try:
        return json.dumps(
            arguments,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        logger.warning("Tool arguments not JSON-safe (%s); using default=str", exc)
        return json.dumps(
            arguments,
            ensure_ascii=True,
            allow_nan=False,
            default=str,
            separators=(",", ":"),
        )


# Strip older ```decl ...``` blocks from chat history (keep latest only).
_DECL_BLOCK_RE = re.compile(r"```decl\s*\n.*?```", re.DOTALL | re.IGNORECASE)

# Max chars per message in prompts log (avoid huge files from tool results)
_PROMPTS_LOG_MAX_MESSAGE_CHARS = 100_000


def _log_prompts_to_file(messages: list[dict], label: str = "") -> None:
    """Append a formatted dump of messages (prompts sent to the AI) to the prompts log file."""
    path = getattr(config, "PROMPTS_LOG_PATH", None)
    if path is None:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write(f"[{datetime.now(timezone.utc).isoformat()}] {label}\n")
            f.write("=" * 80 + "\n")
            for i, m in enumerate(messages):
                role = m.get("role", "?")
                content = m.get("content") or ""
                if len(content) > _PROMPTS_LOG_MAX_MESSAGE_CHARS:
                    content = content[:_PROMPTS_LOG_MAX_MESSAGE_CHARS] + "\n... (truncated)\n"
                f.write(f"\n--- message {i + 1} role={role} ---\n")
                f.write(content)
                f.write("\n")
                tool_calls = m.get("tool_calls") or []
                if tool_calls:
                    f.write("[tool_calls]\n")
                    for tc in tool_calls:
                        fn = tc.get("function") or tc
                        name = fn.get("name", "?")
                        args = fn.get("arguments", "")
                        if isinstance(args, dict):
                            args = json.dumps(args, indent=2, default=str)
                        if len(args) > 5000:
                            args = args[:5000] + "\n... (truncated)"
                        f.write(f"  {name}: {args}\n")
            f.write("\n")
    except OSError as exc:
        logger.warning("Failed to write prompts log: %s", exc)


def _log_response_to_file(
    content: str,
    tool_calls: list[dict] | None = None,
    label: str = "",
) -> None:
    """Append the LLM response (content and tool_calls) to the prompts log file."""
    path = getattr(config, "PROMPTS_LOG_PATH", None)
    if path is None:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n")
            f.write("-" * 80 + "\n")
            f.write(f"[{datetime.now(timezone.utc).isoformat()}] {label}\n")
            f.write("-" * 80 + "\n")
            f.write("\n--- response content ---\n")
            if len(content) > _PROMPTS_LOG_MAX_MESSAGE_CHARS:
                content = content[:_PROMPTS_LOG_MAX_MESSAGE_CHARS] + "\n... (truncated)\n"
            f.write(content)
            f.write("\n")
            if tool_calls:
                f.write("\n--- response tool_calls ---\n")
                for tc in tool_calls:
                    name = tc.get("name", "?")
                    args = tc.get("arguments", "")
                    if isinstance(args, dict):
                        args = json.dumps(args, indent=2, default=str)
                    if len(str(args)) > 5000:
                        args = str(args)[:5000] + "\n... (truncated)"
                    f.write(f"  {name}: {args}\n")
            f.write("\n")
    except OSError as exc:
        logger.warning("Failed to write response to prompts log: %s", exc)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ChatBase(ABC):
    """Common interface for LLM chat backends."""

    SUMMARY_CHAR_THRESHOLD = 50000
    KEEP_RECENT_MESSAGES = 10

    def __init__(
        self,
        system_prompt: str,
        tools: list[dict],
        tool_dispatch: dict[str, Callable[..., str]],
    ):
        self.tools = tools
        self.tool_dispatch = tool_dispatch
        self._system_prompt = system_prompt
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
        self._last_validated: str | None = None

    @property
    def last_validated_content(self) -> str | None:
        return self._last_validated

    def send(self, user_message: str) -> str:
        self._squeeze_context_no_llm()
        if self._total_chars() >= self.SUMMARY_CHAR_THRESHOLD:
            self._summarize_via_llm()
        self.messages.append({"role": "user", "content": user_message})
        return self._run_loop()

    def _execute_tool(self, name: str, arguments: dict) -> str:
        fn = self.tool_dispatch.get(name)
        if fn is None:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            result = fn(**arguments)
            return result if isinstance(result, str) else json.dumps(result)
        except Exception as exc:
            logger.warning("Tool %s raised: %s", name, exc)
            return json.dumps({"error": str(exc)})

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def _total_chars(self) -> int:
        total = 0
        for m in self.messages:
            total += len(m.get("content") or "")
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                total += len(fn.get("arguments") or "")
        return total

    def _extract_state_json(self, text: str) -> str:
        """Extract JSON string from summary response (e.g. from ```json block)."""
        for pattern in [re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL), re.compile(r"(\{.*\})", re.DOTALL)]:
            for match in pattern.findall(text):
                s = match.strip() if isinstance(match, str) else match
                if s.startswith("{"):
                    return s
        return text

    def _squeeze_context_no_llm(self) -> None:
        """Shrink history without an LLM call: stale DECL + old tool blobs."""
        if len(self.messages) <= 1:
            return
        before = self._total_chars()
        self._strip_stale_decl_blocks_inplace()
        self._truncate_old_tool_results_inplace()
        after = self._total_chars()
        if after < before:
            logger.info(
                "Context squeezed (no LLM): %d -> %d chars, %d messages",
                before, after, len(self.messages),
            )

    def _strip_stale_decl_blocks_inplace(self) -> None:
        """Replace all but the last ```decl ...``` fenced block with a short placeholder.

        Only messages that actually contain a full ```decl ... ``` fence are considered,
        so placeholders and prose mentioning 'decl' do not confuse the keeper logic.
        """
        indices: list[int] = []
        for i, m in enumerate(self.messages):
            if i == 0:
                continue
            c = m.get("content") or ""
            if _DECL_BLOCK_RE.search(c):
                indices.append(i)
        if len(indices) <= 1:
            return
        ph = "[earlier .decl revision omitted — latest schematic is in the last decl fence above]"
        for i in indices[:-1]:
            c = self.messages[i].get("content") or ""
            self.messages[i]["content"] = _DECL_BLOCK_RE.sub(ph, c)

    def _truncate_old_tool_results_inplace(
        self,
        keep_full_last: int = 4,
        max_old_len: int = 1200,
    ) -> None:
        tool_indices = [
            i for i, m in enumerate(self.messages) if m.get("role") == "tool"
        ]
        if len(tool_indices) <= keep_full_last:
            return
        for i in tool_indices[:-keep_full_last]:
            c = self.messages[i].get("content") or ""
            if len(c) > max_old_len:
                self.messages[i]["content"] = (
                    c[:max_old_len]
                    + f"\n... ({len(c) - max_old_len} chars omitted) ..."
                )

    def _hard_squeeze_if_still_oversized(self) -> None:
        """Last resort when still over threshold: aggressive tool trim + stale decl."""
        self._strip_stale_decl_blocks_inplace()
        self._truncate_old_tool_results_inplace(keep_full_last=2, max_old_len=600)
        for i, m in enumerate(self.messages):
            if i == 0 or m.get("role") != "user":
                continue
            if i >= len(self.messages) - 2:
                continue
            c = m.get("content") or ""
            if len(c) > 6000:
                m["content"] = c[:4000] + "\n... (long user message truncated) ..."

    def _summarize_via_llm(self) -> None:
        """Summarize older messages via LLM (expensive — only from send(), not each tool round)."""
        total = self._total_chars()
        if total < self.SUMMARY_CHAR_THRESHOLD:
            return

        logger.info(
            "Context at %d chars (threshold %d), summarizing via LLM...",
            total, self.SUMMARY_CHAR_THRESHOLD,
        )

        system_msg = self.messages[0]
        rest = self.messages[1:]

        keep = min(self.KEEP_RECENT_MESSAGES, len(rest))
        start_keep = len(rest) - keep
        while start_keep > 0 and rest[start_keep].get("role") == "tool":
            start_keep -= 1
        to_summarize = rest[:start_keep]
        to_keep = rest[start_keep:]

        if not to_summarize:
            self._hard_squeeze_if_still_oversized()
            return

        transcript = self._format_for_summary(to_summarize)

        try:
            summary = self._call_summary_api(transcript)
            logger.info("Summary produced (%d chars)", len(summary))
        except Exception as exc:
            logger.warning("Summary call failed (%s), falling back to trim", exc)
            self._trim_tool_results()
            return

        state_json = summary
        try:
            parsed = json.loads(self._extract_state_json(summary))
            if isinstance(parsed, dict):
                state_json = json.dumps(parsed, indent=2)
        except (json.JSONDecodeError, TypeError):
            pass

        self.messages = [
            system_msg,
            {"role": "user", "content": f"[STATE]\n{state_json}"},
            {"role": "assistant", "content": "Continuing."},
            *to_keep,
        ]
        self._squeeze_context_no_llm()
        if self._total_chars() >= self.SUMMARY_CHAR_THRESHOLD:
            self._hard_squeeze_if_still_oversized()
        logger.info(
            "Context compacted via LLM: was %d chars -> now %d chars, %d messages",
            total, self._total_chars(), len(self.messages),
        )

    def _format_for_summary(self, messages: list[dict]) -> str:
        """Turn a slice of messages into a compact transcript for the summarizer.

        Only keeps the essentials: user requests, tool call names, and the
        latest DECL/JSON output.  Tool results and intermediate assistant prose
        are aggressively truncated.
        """
        parts: list[str] = []
        for m in messages:
            role = m.get("role", "?")
            content = m.get("content") or ""
            tool_calls = m.get("tool_calls") or []

            if role == "tool":
                parts.append(f"[tool result]: {content[:300]}")
            elif tool_calls:
                names = []
                for tc in tool_calls:
                    fn = tc.get("function") or tc
                    names.append(fn.get("name", "?"))
                parts.append(f"[assistant called]: {', '.join(names)}")
            elif role == "assistant":
                parts.append(f"[assistant]: {content[:600]}")
            elif role == "user":
                parts.append(f"[user]: {content[:800]}")

        full = "\n".join(parts)
        max_len = 4000
        if len(full) > max_len:
            full = full[-max_len:]
        return full

    def _trim_tool_results(self):
        """Fallback: shorten old tool results if summary fails."""
        for m in self.messages:
            if m.get("role") == "tool" and len(m.get("content") or "") > 300:
                m["content"] = m["content"][:250] + "\n... (trimmed) ..."

    @abstractmethod
    def _call_summary_api(self, transcript: str) -> str:
        """Call the LLM (no tools) to produce a summary of the transcript."""

    @abstractmethod
    def _call_api(self) -> tuple[str, list[dict]]:
        """Call the LLM API. Returns (content, tool_calls).

        Each tool_call is {"name": str, "arguments": dict, "id": str|None}.
        """

    @abstractmethod
    def _append_assistant(self, content: str, tool_calls: list[dict]) -> None:
        """Append the assistant's message to self.messages (backend-specific format)."""

    @abstractmethod
    def _append_tool_result(self, tool_call: dict, result: str) -> None:
        """Append a tool result to self.messages (backend-specific format)."""

    def _run_loop(self) -> str:
        self._last_validated = None
        validate_ok_count = 0

        for _ in range(config.MAX_AGENT_ITERATIONS):
            # Never call the summary LLM here — only cheap squeezes. LLM summary runs at
            # most once per send() to avoid thrashing during repair tool loops.
            self._squeeze_context_no_llm()
            if self._total_chars() >= self.SUMMARY_CHAR_THRESHOLD:
                self._hard_squeeze_if_still_oversized()
            _log_prompts_to_file(self.messages, label="LLM request")
            content, tool_calls = self._call_api()
            _log_response_to_file(content, tool_calls, label="LLM response")
            self._append_assistant(content, tool_calls)

            if not tool_calls:
                return content

            for tc in tool_calls:
                fn_name = tc["name"]
                fn_args = tc["arguments"]

                logger.info("Tool call: %s(%s)", fn_name, json.dumps(fn_args, default=str)[:200])
                result = self._execute_tool(fn_name, fn_args)
                logger.info("Tool result (%s): %s", fn_name, result[:300])

                if fn_name == "validate_decl" and "VALIDATION PASSED" in result:
                    self._last_validated = fn_args.get("content", "")
                    validate_ok_count += 1
                    if validate_ok_count >= 2:
                        logger.info("Validated OK twice -- returning content directly")
                        return content or f"```decl\n{self._last_validated}\n```"

                self._append_tool_result(tc, result)

        return content


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

class OllamaChat(ChatBase):

    def __init__(
        self,
        system_prompt: str,
        tools: list[dict],
        tool_dispatch: dict[str, Callable[..., str]],
        *,
        model: str | None = None,
        base_url: str | None = None,
    ):
        super().__init__(system_prompt, tools, tool_dispatch)
        self.base_url = (base_url or config.OLLAMA_URL).rstrip("/")
        self.model = model or config.OLLAMA_MODEL
        self.SUMMARY_CHAR_THRESHOLD = getattr(
            config, "OLLAMA_SUMMARY_CHAR_THRESHOLD", 40_000
        )

    def send(self, user_message: str) -> str:
        self._squeeze_context_no_llm()
        if self._total_chars() >= self.SUMMARY_CHAR_THRESHOLD:
            self._summarize_via_llm()
        self.messages.append({"role": "user", "content": "/nothink\n" + user_message})
        return self._run_loop()

    def _call_summary_api(self, transcript: str) -> str:
        summary_messages = [
            {"role": "system", "content": STATE_SUMMARY_PROMPT},
            {"role": "user", "content": "/nothink\n" + transcript},
        ]
        _log_prompts_to_file(summary_messages, label="Summary/state compression request")
        payload = {
            "model": self.model,
            "messages": summary_messages,
            "stream": False,
            "options": {"num_ctx": config.OLLAMA_NUM_CTX, "num_gpu": 99},
        }
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=config.OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")
        _log_response_to_file(content, None, label="Summary/state compression response")
        return content

    def _call_api(self) -> tuple[str, list[dict]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self.messages,
            "stream": False,
            "keep_alive": -1,
            "options": {"num_ctx": config.OLLAMA_NUM_CTX, "num_gpu": 99},
        }
        if self.tools:
            payload["tools"] = self.tools

        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=config.OLLAMA_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.exceptions.ReadTimeout:
                logger.warning("Ollama timeout (attempt %d/3)", attempt + 1)
                if attempt == 2:
                    raise
            except requests.exceptions.ConnectionError:
                logger.warning("Ollama connection error (attempt %d/3)", attempt + 1)
                if attempt == 2:
                    raise
                time.sleep(5)

        msg = data.get("message", {})
        content = msg.get("content", "")
        raw_calls = msg.get("tool_calls") or []
        tool_calls = []
        for tc in raw_calls:
            fn = tc.get("function", {})
            tool_calls.append({
                "name": fn.get("name", ""),
                "arguments": fn.get("arguments", {}),
                "id": None,
                "_raw": tc,
            })
        return content, tool_calls

    def _append_assistant(self, content: str, tool_calls: list[dict]) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = [tc["_raw"] for tc in tool_calls]
        self.messages.append(msg)

    def _append_tool_result(self, tool_call: dict, result: str) -> None:
        self.messages.append({"role": "tool", "content": result})


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

class OpenAIChat(ChatBase):

    def __init__(
        self,
        system_prompt: str,
        tools: list[dict],
        tool_dispatch: dict[str, Callable[..., str]],
        *,
        model: str | None = None,
        api_key: str | None = None,
    ):
        super().__init__(
            _sanitize_openai_text(system_prompt),
            tools,
            tool_dispatch,
        )
        self.SUMMARY_CHAR_THRESHOLD = getattr(
            config, "OPENAI_SUMMARY_CHAR_THRESHOLD", 110_000
        )
        self.model = (model or config.OPENAI_MODEL or "gpt-4o").strip() or "gpt-4o"
        raw_key = api_key or config.OPENAI_API_KEY
        self._api_key = "".join((raw_key or "").split())
        if not self._api_key:
            raise RuntimeError("OpenAI API key not found. Place it in openai.key")

        from openai import OpenAI

        client_kw: dict[str, Any] = {
            "api_key": self._api_key,
            "timeout": config.OPENAI_TIMEOUT,
        }
        base_url = getattr(config, "OPENAI_BASE_URL", None)
        if base_url:
            client_kw["base_url"] = base_url
        self._client = OpenAI(**client_kw)

    def send(self, user_message: str) -> str:
        user_message = _sanitize_openai_text(user_message)
        self._squeeze_context_no_llm()
        if self._total_chars() >= self.SUMMARY_CHAR_THRESHOLD:
            self._summarize_via_llm()
        self.messages.append({"role": "user", "content": user_message})
        return self._run_loop()

    def _call_summary_api(self, transcript: str) -> str:
        summary_messages = [
            {"role": "system", "content": _sanitize_openai_text(STATE_SUMMARY_PROMPT)},
            {"role": "user", "content": _sanitize_openai_text(transcript)},
        ]
        _log_prompts_to_file(summary_messages, label="Summary/state compression request (OpenAI)")
        response = self._client.chat.completions.create(
            model=self.model,
            messages=summary_messages,
        )
        content = response.choices[0].message.content or ""
        _log_response_to_file(content, None, label="Summary/state compression response (OpenAI)")
        return content

    def _call_api(self) -> tuple[str, list[dict]]:
        _sanitize_openai_messages(self.messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self.messages,
        }
        if self.tools:
            kwargs["tools"] = self.tools

        from openai import APIStatusError

        for attempt in range(3):
            try:
                response = self._client.chat.completions.create(**kwargs)
                break
            except APIStatusError as exc:
                resp = getattr(exc, "response", None)
                body_preview = ""
                if resp is not None:
                    try:
                        body_preview = (resp.text or "")[:800]
                    except Exception:
                        body_preview = ""
                logger.warning(
                    "OpenAI HTTP error (attempt %d/3): %s — response body (truncated): %r",
                    attempt + 1,
                    exc,
                    body_preview,
                )
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
            except Exception as exc:
                logger.warning("OpenAI error (attempt %d/3): %s", attempt + 1, exc)
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

        choice = response.choices[0]
        msg = choice.message
        content = msg.content or ""
        tool_calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append({
                "name": tc.function.name,
                "arguments": args,
                "id": tc.id,
                "_raw_msg": msg,
            })
        return content, tool_calls

    def _append_assistant(self, content: str, tool_calls: list[dict]) -> None:
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": _sanitize_openai_text(content or ""),
        }
        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": _tool_args_to_json_str(tc["arguments"]),
                    },
                }
                for tc in tool_calls
            ]
        self.messages.append(msg)

    def _append_tool_result(self, tool_call: dict, result: str) -> None:
        text = result if isinstance(result, str) else str(result)
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": _sanitize_openai_text(text),
        })


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_chat(
    backend: str,
    system_prompt: str,
    tools: list[dict],
    tool_dispatch: dict[str, Callable[..., str]],
    *,
    model: str | None = None,
    ollama_url: str | None = None,
    api_key: str | None = None,
) -> ChatBase:
    """Create the appropriate chat backend."""
    if backend == "openai":
        return OpenAIChat(
            system_prompt, tools, tool_dispatch,
            model=model, api_key=api_key,
        )
    return OllamaChat(
        system_prompt, tools, tool_dispatch,
        model=model, base_url=ollama_url,
    )
