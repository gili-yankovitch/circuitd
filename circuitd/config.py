"""Configuration constants for the circuitd agent."""

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
OPENAI_API_KEY = _key_file.read_text().strip() if _key_file.is_file() else ""
OPENAI_MODEL = "gpt-4o"
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
