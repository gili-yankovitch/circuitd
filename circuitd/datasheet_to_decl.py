"""
Standalone datasheet → DECL converter. Use in-flow (from agent) or out-of-flow via CLI.

  python -m circuitd.datasheet_to_decl datasheet.txt   # plain text
  python -m circuitd.datasheet_to_decl datasheet.pdf   # PDF (text extracted via pymupdf)
  python -m circuitd.datasheet_to_decl -o out.decl      # stdin → stdout
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from . import config
from .llm import create_chat
from .prompts import DATASHEET_TO_DECL_PROMPT
from .tools import _fix_common_issues, save_to_stdlib, validate_decl_structured

logger = logging.getLogger(__name__)

_DECL_FENCE_RE = re.compile(r"```(?:decl)?\s*\n(.*?)```", re.DOTALL)
_FIRST_COMPONENT_RE = re.compile(r"\bcomponent\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{", re.MULTILINE)

DEFAULT_MAX_ATTEMPTS = 20


def _read_input_text(path: str | Path) -> str:
    """Read input as text. If path is a .pdf file, extract text with pymupdf."""
    path = Path(path) if path != "-" else None
    if path is None or str(path) == "-":
        return sys.stdin.read()
    if path.suffix.lower() == ".pdf":
        try:
            import pymupdf
        except ImportError:
            raise SystemExit("PDF input requires pymupdf. Install with: pip install pymupdf")
        pdf_bytes = path.read_bytes()
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        text_parts: list[str] = []
        total = 0
        max_chars = getattr(config, "DATASHEET_MAX_CHARS", 8000)
        for page in doc:
            text_parts.append(page.get_text())
            total += len(text_parts[-1])
            if total >= max_chars:
                break
        doc.close()
        return "\n".join(text_parts)[:max_chars]
    return path.read_text()


def _extract_decl(text: str) -> str | None:
    """Extract the last fenced decl block from the response."""
    matches = _DECL_FENCE_RE.findall(text)
    if not matches:
        return None
    candidate = matches[-1].strip()
    if not candidate:
        return None
    if any(kw in candidate for kw in ("component ", "schematic ", "protocol ", "variant ")):
        return candidate
    return None


def first_component_name_from_decl(decl: str) -> str | None:
    """Return the name of the first component defined in decl, or None."""
    m = _FIRST_COMPONENT_RE.search(decl)
    return m.group(1) if m else None


def convert_datasheet_to_decl(
    datasheet_text: str,
    url: str = "",
    *,
    backend: str | None = None,
    model: str | None = None,
    ollama_url: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> str | None:
    """
    Convert a datasheet excerpt to a valid DECL string. Retries until valid or max_attempts.
    Returns the DECL string on success, None on failure. Does not save to disk.
    """
    if not datasheet_text or len(datasheet_text.strip()) < 100:
        logger.debug("Datasheet text too short to convert")
        return None
    backend = backend or getattr(config, "BACKEND", "ollama")
    prompt_prefix = (
        "Convert this datasheet excerpt to a single DECL component (and optional variants). "
        "Output ONLY a ```decl block.\n\n" + datasheet_text[:8000]
    )
    try:
        chat = create_chat(
            backend,
            DATASHEET_TO_DECL_PROMPT,
            tools=[],
            tool_dispatch={},
            model=model,
            ollama_url=ollama_url,
        )
        feedback = ""
        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                msg = (
                    f"Previous attempt had issues. {feedback}\n\n"
                    "Fix the DECL and output ONLY a ```decl block (no other text)."
                )
            else:
                msg = prompt_prefix
            response = chat.send(msg)
            decl = _extract_decl(response)
            if not decl:
                feedback = "No ```decl code block was found in the response."
                logger.warning("Attempt %d/%d: %s", attempt, max_attempts, feedback)
                if attempt == max_attempts:
                    return None
                continue
            decl = _fix_common_issues(decl)
            human_output, errors = validate_decl_structured(decl)
            if errors:
                # Save failing DECL under ~/decl/ so user can open it; line numbers in output refer to this file
                decl_dir = Path.home() / "decl"
                decl_dir.mkdir(parents=True, exist_ok=True)
                failed_path = decl_dir / "datasheet_failed.decl"
                failed_path.write_text(decl)
                feedback = (
                    f"Validation failed. Failing DECL saved to {failed_path} "
                    "(line numbers below refer to this file).\n\n"
                    f"{human_output}"
                )
                logger.warning(
                    "Attempt %d/%d: validation failed (see %s): %s",
                    attempt, max_attempts, failed_path,
                    "; ".join(e.get("message", str(e)) for e in errors[:3]),
                )
                if attempt == max_attempts:
                    return None
                continue
            return decl
    except Exception as exc:
        logger.warning("Datasheet-to-DECL failed: %s", exc)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a datasheet excerpt to DECL (component + optional variants)."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="-",
        help="Input file: .txt or .pdf (default: stdin)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="-",
        help="Output .decl file (default: stdout)",
    )
    parser.add_argument(
        "--url",
        default="",
        help="Source URL (for naming / logging)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save result to stdlib (components/agent/<ComponentName>.decl)",
    )
    parser.add_argument(
        "--backend",
        default=getattr(config, "BACKEND", "ollama"),
        help="LLM backend (ollama | openai)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default from config)",
    )
    parser.add_argument(
        "--ollama-url",
        default=getattr(config, "OLLAMA_URL", "http://localhost:11434"),
        help="Ollama base URL",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=f"Max conversion attempts (default: {DEFAULT_MAX_ATTEMPTS})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    try:
        text = _read_input_text(args.input)
    except Exception as e:
        logger.error("Failed to read input: %s", e)
        return 1

    decl = convert_datasheet_to_decl(
        text,
        url=args.url,
        backend=args.backend,
        model=args.model,
        ollama_url=args.ollama_url,
        max_attempts=args.max_attempts,
    )
    if not decl:
        logger.error("Conversion failed")
        return 1

    if args.save:
        import hashlib
        short_hash = hashlib.sha256(args.url.encode() or b"").hexdigest()[:12]
        fallback_path = f"components/agent/datasheet_{short_hash}.decl"
        main_name = first_component_name_from_decl(decl)
        save_path = f"components/agent/{main_name}.decl" if main_name else fallback_path
        result = save_to_stdlib(save_path, decl)
        out = json.loads(result) if result.strip().startswith("{") else {}
        if out.get("ok"):
            print(f"Saved to stdlib: {save_path}", file=sys.stderr)
        else:
            logger.error("Save failed: %s", out.get("error", result))
            return 1
    else:
        if args.output == "-":
            sys.stdout.write(decl)
            if not decl.endswith("\n"):
                sys.stdout.write("\n")
        else:
            Path(args.output).write_text(decl)

    return 0


if __name__ == "__main__":
    sys.exit(main())
