"""Configuration constants for the circuitd agent."""

import os
from pathlib import Path

# -- LLM Backend Selection ---------------------------------------------------

BACKEND = "ollama"

# -- Ollama ------------------------------------------------------------------

OLLAMA_URL = "http://10.0.0.10:11434"
OLLAMA_MODEL = "qwen3-coder:30b-a3b-q4_K_M"
OLLAMA_TIMEOUT = 600  # seconds per LLM call
OLLAMA_NUM_CTX = 16384  # context window (tokens)
# When total message chars exceed this, optional LLM summarize runs (once per send()).
OLLAMA_SUMMARY_CHAR_THRESHOLD = 40_000

# -- OpenAI ------------------------------------------------------------------

_key_file = Path(__file__).resolve().parent.parent / "openai.key"
# Strip BOM/newlines/whitespace — accidental line breaks in the key file or env
# can corrupt the Authorization header and cause bizarre HTTP 400s from the API.
def _normalize_openai_secret(raw: str) -> str:
    return "".join(raw.strip().split())


_env_key = os.environ.get("OPENAI_API_KEY", "").strip()
if _env_key:
    OPENAI_API_KEY = _normalize_openai_secret(_env_key)
elif _key_file.is_file():
    OPENAI_API_KEY = _normalize_openai_secret(
        _key_file.read_text(encoding="utf-8-sig", errors="replace")
    )
else:
    OPENAI_API_KEY = ""
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o").strip() or "gpt-4o"
# Optional (Azure, proxies, local gateways). Empty = default OpenAI API.
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "").strip() or None
OPENAI_TIMEOUT = 300
# Summarization runs only when context exceeds this (chars). OpenAI models have large
# windows — keep this high to avoid thrashing on full DECL + validator output.
OPENAI_SUMMARY_CHAR_THRESHOLD = 110_000

# -- Parts search ------------------------------------------------------------

PARTS_API_URL = "http://localhost:8811"

# -- DECL stdlib / checker ---------------------------------------------------

STDLIB_PATH = Path.home() / "decl" / "stdlib"
# Agent-writable subdir for datasheet-derived components (created on first save)
STDLIB_AGENT_SUBDIR = "components/agent"
DECL_CHECK_CMD = "decl"

# -- Agent loop --------------------------------------------------------------

MAX_AGENT_ITERATIONS = 15
MAX_VALIDATION_RETRIES = 3

# -- Tool settings -----------------------------------------------------------

DATASHEET_MAX_CHARS = 4000
PARTS_DEFAULT_TOP = 3

# -- Prompt logging ----------------------------------------------------------

# Path to append all prompts and responses sent to the AI. Set to None to disable.
PROMPTS_LOG_PATH = Path.cwd() / "circuitd_prompts.log"
