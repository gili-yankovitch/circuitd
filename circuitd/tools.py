"""Tool implementations exposed to the LLM agent via Ollama tool calling."""

import json
import logging
import subprocess
import tempfile
from pathlib import Path

import requests

from . import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ollama tool-calling schema definitions
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_parts",
            "description": (
                "Search the JLCPCB electronic component database by natural-language "
                "query or part number. Returns matching components with MPN, package, "
                "description, datasheet URL, attributes, price, and stock."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, e.g. 'LDO 3.3V SOT-23' or 'STM32F103'",
                    },
                    "top": {
                        "type": "integer",
                        "description": "Max results to return (default 5)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_part_datasheet",
            "description": (
                "Download a component datasheet PDF from a URL and extract its text "
                "content. Returns the first ~8000 characters covering pinout, "
                "electrical characteristics, and absolute maximum ratings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL to the datasheet PDF",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_stdlib",
            "description": (
                "List all components and protocols available in the DECL standard "
                "library. Returns file names grouped by category."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_stdlib_file",
            "description": (
                "Read the full content of a DECL standard library file. "
                "Pass a relative path like 'components/resistor.decl' or 'protocols/spi.decl'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path inside stdlib, e.g. 'components/resistor.decl'",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_decl",
            "description": (
                "Validate a .decl source string by running the DECL checker. "
                "Returns 'OK' with warnings, or error messages if invalid."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The full .decl source code to validate",
                    },
                },
                "required": ["content"],
            },
        },
    },
]

TOOL_DISPATCH: dict = {}


def _register(name: str):
    def decorator(fn):
        TOOL_DISPATCH[name] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

@_register("search_parts")
def search_parts(query: str, top: int | None = None) -> str:
    top = top or config.PARTS_DEFAULT_TOP
    try:
        resp = requests.get(
            f"{config.PARTS_API_URL}/search",
            params={"q": query, "top": top},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return json.dumps({"error": f"Parts search failed: {exc}"})

    results = []
    for comp in data.get("results", []):
        results.append({
            "lcsc": comp.get("lcsc"),
            "mpn": comp.get("mpn") or comp.get("mfr"),
            "manufacturer": comp.get("manufacturer"),
            "package": comp.get("package"),
            "description": (comp.get("description") or "")[:200],
            "datasheet": comp.get("datasheet"),
            "attributes": comp.get("attributes"),
            "price": comp.get("price"),
            "stock": comp.get("stock"),
        })
    return json.dumps({"query": query, "count": len(results), "results": results}, indent=2)


@_register("get_part_datasheet")
def get_part_datasheet(url: str) -> str:
    if not url or not url.startswith("http"):
        return json.dumps({"error": "Invalid datasheet URL"})

    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "circuitd/1.0"})
        resp.raise_for_status()
    except Exception as exc:
        return json.dumps({"error": f"Failed to download datasheet: {exc}"})

    try:
        import pymupdf
    except ImportError:
        return json.dumps({"error": "pymupdf not installed -- run: pip install pymupdf"})

    try:
        pdf_bytes = resp.content
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        text_parts: list[str] = []
        total = 0
        for page in doc:
            page_text = page.get_text()
            text_parts.append(page_text)
            total += len(page_text)
            if total >= config.DATASHEET_MAX_CHARS:
                break
        doc.close()
        full_text = "\n".join(text_parts)[:config.DATASHEET_MAX_CHARS]
    except Exception as exc:
        return json.dumps({"error": f"Failed to parse PDF: {exc}"})

    return json.dumps({"url": url, "text": full_text})


@_register("list_stdlib")
def list_stdlib() -> str:
    stdlib = config.STDLIB_PATH
    result: dict[str, list[str]] = {}
    if not stdlib.is_dir():
        return json.dumps({"error": f"stdlib not found at {stdlib}"})

    for subdir in sorted(stdlib.iterdir()):
        if subdir.is_dir():
            files = sorted(f.name for f in subdir.glob("*.decl"))
            if files:
                result[subdir.name] = files
    return json.dumps(result, indent=2)


@_register("read_stdlib_file")
def read_stdlib_file(path: str) -> str:
    target = config.STDLIB_PATH / path
    if not target.is_file():
        return json.dumps({"error": f"File not found: {path}"})
    try:
        return target.read_text()
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@_register("validate_decl")
def validate_decl(content: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".decl", delete=False) as f:
        f.write(content)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [config.DECL_CHECK_CMD, "check", tmp_path],
            capture_output=True,
            text=True,
            timeout=15,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        output = ""
        if stdout:
            output += stdout
        if stderr:
            output += ("\n" if output else "") + stderr

        if not output:
            return "(no output)"

        is_ok = output.strip().startswith("OK:")
        output = _annotate_errors(output, content)

        if is_ok:
            output += (
                "\n\nVALIDATION PASSED. W001 (unconnected pin) warnings are expected "
                "for unused MCU pins and can be ignored. Do NOT re-validate. "
                "Output the final .decl file in a ```decl code block now."
            )
        return output
    except FileNotFoundError:
        return "Error: 'decl' command not found. Is decl installed?"
    except Exception as exc:
        return f"Validation error: {exc}"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _annotate_errors(output: str, source: str) -> str:
    """Append the offending source lines to error/warning messages."""
    import re
    lines = source.splitlines()
    annotations: list[str] = []
    for match in re.finditer(r":(\d+):\d+: (.+)", output):
        lineno = int(match.group(1))
        start = max(0, lineno - 2)
        end = min(len(lines), lineno + 2)
        snippet = "\n".join(
            f"  {'>>>' if i + 1 == lineno else '   '} {i + 1:3d}| {lines[i]}"
            for i in range(start, end)
        )
        annotations.append(f"\nContext around line {lineno}:\n{snippet}")
    if annotations:
        output += "\n" + "\n".join(annotations)
    return output
