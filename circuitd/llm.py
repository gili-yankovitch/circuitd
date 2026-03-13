"""Ollama chat client with native tool-calling support."""

import json
import logging
import requests
from typing import Any, Callable

from . import config

logger = logging.getLogger(__name__)


class OllamaChat:
    """Manages a multi-turn conversation with an Ollama model that supports tool calls."""

    def __init__(
        self,
        system_prompt: str,
        tools: list[dict],
        tool_dispatch: dict[str, Callable[..., str]],
        *,
        model: str | None = None,
        base_url: str | None = None,
    ):
        self.base_url = (base_url or config.OLLAMA_URL).rstrip("/")
        self.model = model or config.OLLAMA_MODEL
        self.tools = tools
        self.tool_dispatch = tool_dispatch
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]

    def _call_api(self, *, include_tools: bool = True) -> dict:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self.messages,
            "stream": False,
            "options": {"num_ctx": config.OLLAMA_NUM_CTX, "num_gpu": 99},
        }
        if include_tools and self.tools:
            payload["tools"] = self.tools

        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=config.OLLAMA_TIMEOUT,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.ReadTimeout:
                logger.warning("Ollama timeout (attempt %d/3)", attempt + 1)
                if attempt == 2:
                    raise
            except requests.exceptions.ConnectionError:
                logger.warning("Ollama connection error (attempt %d/3)", attempt + 1)
                if attempt == 2:
                    raise
                import time
                time.sleep(5)

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

    def send(self, user_message: str) -> str:
        """Send a user message and run the full tool-call loop until
        the model produces a final text response.

        Returns the assistant's final text content.
        """
        self.messages.append({"role": "user", "content": user_message})
        return self._run_loop()

    @property
    def last_validated_content(self) -> str | None:
        """The last .decl content that passed validation via a tool call."""
        return self._last_validated

    def _run_loop(self) -> str:
        self._last_validated: str | None = None
        validate_ok_count = 0

        for _ in range(config.MAX_AGENT_ITERATIONS):
            data = self._call_api()
            msg = data.get("message", {})
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")

            self.messages.append(msg)

            if not tool_calls:
                return content

            for tc in tool_calls:
                fn_info = tc.get("function", {})
                fn_name = fn_info.get("name", "")
                fn_args = fn_info.get("arguments", {})

                logger.info("Tool call: %s(%s)", fn_name, json.dumps(fn_args, default=str)[:200])
                result = self._execute_tool(fn_name, fn_args)
                logger.info("Tool result (%s): %s", fn_name, result[:300])

                if fn_name == "validate_decl" and "VALIDATION PASSED" in result:
                    self._last_validated = fn_args.get("content", "")
                    validate_ok_count += 1
                    if validate_ok_count >= 2:
                        logger.info("Model validated OK twice via tool -- returning content directly")
                        return content or f"```decl\n{self._last_validated}\n```"

                self.messages.append({"role": "tool", "content": result})

        return content
