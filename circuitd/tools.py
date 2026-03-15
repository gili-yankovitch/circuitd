"""Tool implementations exposed to the LLM agent via Ollama tool calling."""

import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path

import requests

from . import config

logger = logging.getLogger(__name__)

# Regex for parsing decl checker output: optional path, then line:col: message (E001-E010)
_VALIDATION_ERROR_RE = re.compile(
    r"(?:(?:^|\n)(?:\S+?)?)?:(\d+):\d*:\s*(.*?)(?=\n\S|\n\n|$)",
    re.DOTALL,
)
_ERROR_CODE_RE = re.compile(r"\b(E00[1-9]|E010)\b")

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
    {
        "type": "function",
        "function": {
            "name": "save_to_stdlib",
            "description": (
                "Save a validated DECL fragment (component, variant, or protocol) to the "
                "stdlib agent directory so it can be reused in future designs. Use after "
                "producing DECL from a datasheet. Path is relative to stdlib, e.g. 'components/agent/W25Q128.decl'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path under stdlib, must start with components/agent/ (e.g. components/agent/W25Q128.decl)",
                    },
                    "content": {
                        "type": "string",
                        "description": "The DECL source to save (will be validated first)",
                    },
                },
                "required": ["path", "content"],
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


@_register("save_to_stdlib")
def save_to_stdlib(path: str, content: str) -> str:
    """Save validated DECL to stdlib agent dir (e.g. components/agent/<part>.decl)."""
    normalized = path.strip().replace("\\", "/").lstrip("/")
    if ".." in normalized:
        return json.dumps({"error": "Path must not contain ..", "path": path})
    if not normalized.startswith(config.STDLIB_AGENT_SUBDIR + "/"):
        return json.dumps({
            "error": f"Path must be under {config.STDLIB_AGENT_SUBDIR}/ (e.g. components/agent/MYPART.decl)",
            "path": path,
        })
    target = config.STDLIB_PATH / normalized
    fixed = _fix_common_issues(content)
    _, structured = validate_decl_structured(fixed)
    if structured:
        return json.dumps({
            "error": "DECL validation failed; fix before saving",
            "validation_errors": structured[:5],
        })
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(fixed + "\n")
        return json.dumps({"ok": True, "path": normalized, "message": f"Saved to {target}"})
    except OSError as exc:
        return json.dumps({"error": str(exc), "path": path})


def _fix_common_issues(content: str) -> str:
    """Auto-fix common LLM mistakes in .decl source before validation."""
    import re

    # Replace # in pin names with _N (e.g., HOLD# -> HOLD_N, RESET# -> RESET_N)
    content = re.sub(r'(\b\w+)#', r'\1_N', content)

    # Fix D+ -> DP and D- -> DN (common USB pin names with invalid chars)
    content = re.sub(r'\bD\+', 'DP', content)
    content = re.sub(r'\bD-', 'DN', content)

    # Remove attributes with invalid types the LLM likes to invent
    invalid_type_pattern = re.compile(
        r'^\s+\w+:\s*(?:Force|Distance|String|Boolean|Integer|Float|Count)\b.*$',
        re.MULTILINE,
    )
    content = invalid_type_pattern.sub('', content)

    # `package` is a reserved keyword in DECL -- rename to `pkg`
    content = re.sub(
        r'^(\s+)package(\s*:\s*Package\b)',
        r'\1pkg\2',
        content,
        flags=re.MULTILINE,
    )

    # Fix -> used in connect statements (should be --)
    # But only in connect lines, NOT in feature pin mappings
    content = re.sub(r'^(\s*connect\s+\S+\s+)->\s*', r'\1-- ', content, flags=re.MULTILINE)

    # Remove commas (common LLM mistake)
    content = re.sub(r',\s*$', '', content, flags=re.MULTILINE)
    content = re.sub(r',(\s*\n)', r'\1', content)

    # Remove semicolons
    content = re.sub(r';\s*$', '', content, flags=re.MULTILINE)

    # Remove blank lines that result from attribute removal (collapse double+ blank lines)
    content = re.sub(r'\n{3,}', '\n\n', content)

    return content


@_register("validate_decl")
def validate_decl(content: str) -> str:
    """Validate .decl source; returns human-readable output (backward compatible)."""
    output, _ = validate_decl_structured(content)
    return output


def _annotate_errors(output: str, source: str) -> str:
    """Append the offending source lines to error/warning messages."""
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


def _parse_validation_errors(raw_output: str) -> list[dict]:
    """Parse decl checker stdout/stderr into structured errors for repair.

    Returns a list of dicts: {"code": "E003", "line": 23, "message": "...", "entities": []}.
    If the checker does not emit parseable codes/lines, returns [].
    """
    structured: list[dict] = []
    seen: set[tuple[int, str]] = set()
    for match in re.finditer(r":(\d+):\d*:\s*([^\n]+)", raw_output):
        line_str, msg = match.group(1), match.group(2).strip()
        try:
            line_no = int(line_str)
        except ValueError:
            continue
        code_match = _ERROR_CODE_RE.search(msg)
        code = code_match.group(1) if code_match else None
        if (line_no, msg) in seen:
            continue
        seen.add((line_no, msg))
        # Try to extract entity names (e.g. mcu.VDD, ldo.VOUT) from message
        entities: list[str] = []
        for part in re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z0-9_]+)\b", msg):
            entities.append(part)
        structured.append({
            "code": code or "E000",
            "line": line_no,
            "message": msg[:500],
            "entities": entities[:10],
        })
    return structured


def _run_decl_check(content: str) -> tuple[str, str, bool]:
    """Run decl checker on content. Returns (stdout, stderr, success)."""
    content = _fix_common_issues(content)
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
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        ok = stdout.strip().startswith("OK:")
        return stdout, stderr, ok
    except FileNotFoundError:
        return "", "Error: 'decl' command not found. Is decl installed?", False
    except Exception as exc:
        return "", f"Validation error: {exc}", False
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def validate_decl_structured(content: str) -> tuple[str, list[dict]]:
    """Validate .decl content; return (human-readable output, structured errors).

    Structured errors are list of {"code", "line", "message", "entities"} for use
    in the repair phase. If the checker output is not parseable, structured_errors
    is empty but human_output is still returned.
    """
    stdout, stderr, is_ok = _run_decl_check(content)
    output = ""
    if stdout:
        output += stdout
    if stderr:
        output += ("\n" + stderr) if output else stderr
    if not output:
        output = "(no output)"
    output = _annotate_errors(output, _fix_common_issues(content))
    structured = [] if is_ok else _parse_validation_errors(stdout + "\n" + stderr)
    if is_ok:
        output += (
            "\n\nVALIDATION PASSED. W001 (unconnected pin) warnings are expected "
            "for unused MCU pins and can be ignored. Do NOT re-validate. "
            "Output the final .decl file in a ```decl code block now."
        )
    return output, structured
