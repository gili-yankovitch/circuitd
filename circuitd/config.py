"""Configuration constants for the circuitd agent."""

from pathlib import Path

OLLAMA_URL = "http://10.0.0.10:11434"
OLLAMA_MODEL = "qwen3-coder:30b-a3b-q4_K_M"

PARTS_API_URL = "http://localhost:8811"

STDLIB_PATH = Path.home() / "decl" / "stdlib"
DECL_CHECK_CMD = "decl"

MAX_AGENT_ITERATIONS = 15
MAX_VALIDATION_RETRIES = 3

DATASHEET_MAX_CHARS = 4000
PARTS_DEFAULT_TOP = 3

OLLAMA_TIMEOUT = 600  # seconds per LLM call
OLLAMA_NUM_CTX = 16384  # context window (tokens)
