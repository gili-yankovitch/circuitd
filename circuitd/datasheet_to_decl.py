"""
Standalone datasheet → DECL converter. Use in-flow (from agent) or out-of-flow via CLI.

  python -m circuitd.datasheet_to_decl datasheet.txt   # plain text (single-shot)
  python -m circuitd.datasheet_to_decl datasheet.pdf  # PDF: page-by-page tools + LLM
  python -m circuitd.datasheet_to_decl -o out.decl    # stdin → stdout
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import TextIO

from . import config
from .llm import create_chat
from .prompts import DATASHEET_TO_DECL_PROMPT, DATASHEET_PDF_ITERATIVE_INSTR
from .tools import (
    TOOL_DEFINITIONS,
    TOOL_DISPATCH,
    _fix_common_issues,
    save_to_stdlib,
    validate_decl_structured,
)

logger = logging.getLogger(__name__)

_DECL_FENCE_RE = re.compile(r"```(?:decl)?\s*\n(.*?)```", re.DOTALL)
_FIRST_COMPONENT_RE = re.compile(r"\bcomponent\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{", re.MULTILINE)

DEFAULT_MAX_ATTEMPTS = 30

# Tools for PDF conversion: iterative datasheet read + optional self-check
_DATASHEET_PDF_TOOL_NAMES = frozenset({
    "get_part_datasheet",
    "read_datasheet_pages",
    "validate_decl",
})


def _pdf_tool_definitions() -> list[dict]:
    return [t for t in TOOL_DEFINITIONS if t["function"]["name"] in _DATASHEET_PDF_TOOL_NAMES]


def _pdf_tool_dispatch() -> dict:
    return {n: TOOL_DISPATCH[n] for n in _DATASHEET_PDF_TOOL_NAMES}


def _read_input_text(path: str | Path) -> str:
    """Read input as plain text. PDFs should use convert_pdf_path_to_decl instead."""
    path = Path(path) if path != "-" else None
    if path is None or str(path) == "-":
        return sys.stdin.read()
    if path.suffix.lower() == ".pdf":
        raise ValueError("PDF inputs use the tool-based converter; do not call _read_input_text")
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


def _validation_trace_print(
    phase: str,
    iteration: int,
    human: str,
    errors: list[dict],
    *,
    stream: TextIO | None = None,
) -> None:
    """Print every structured error and checker text to stderr (or stream)."""
    if stream is None:
        stream = sys.stderr
    print(f"\n=== datasheet_to_decl: {phase} (iteration {iteration}) ===", file=stream)
    if errors:
        for i, e in enumerate(errors, 1):
            line = e.get("line") or 0
            code = e.get("code", "")
            msg = e.get("message", str(e))
            print(f"  [{i}] {code} line {line}: {msg}", file=stream)
    else:
        print("  (no structured errors; raw checker output below)", file=stream)
    h = (human or "").strip()
    if h:
        print(h, file=stream)
    print(f"=== end iteration {iteration} ===\n", file=stream)


def _repair_follow_payload(human: str, errors: list[dict], max_chars: int = 12000) -> str:
    """Build text we send back to the LLM so it can fix DECL (bounded size)."""
    lines = [e.get("message", str(e)) for e in errors]
    body = "\n".join(lines) if lines else "(see decl check output)"
    h = (human or "").strip()
    if h:
        if len(h) > max_chars:
            half = max_chars // 2
            h = h[:half] + "\n...[truncated]...\n" + h[-half:]
        body = body + "\n\n--- decl check output ---\n" + h
    return body


def _stdlib_validation_base() -> Path | None:
    sp = getattr(config, "STDLIB_PATH", None)
    if sp is None:
        return None
    p = Path(sp)
    return p if p.is_dir() else None


def convert_pdf_path_to_decl(
    pdf_path: Path,
    *,
    backend: str | None = None,
    model: str | None = None,
    ollama_url: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    quiet: bool = False,
    validation_base: Path | None = None,
) -> str | None:
    """
    Convert a local (or file://) datasheet PDF using get_part_datasheet + read_datasheet_pages,
    then validate / repair in further chat turns.

    Unless ``quiet`` is True, prints every validation failure and missing-fence nudge to stderr.
    ``validation_base`` defaults to ``config.STDLIB_PATH`` for import expansion during ``decl check``.
    """
    if not pdf_path.is_file():
        logger.error("PDF not found: %s", pdf_path)
        return None

    backend = backend or getattr(config, "BACKEND", "ollama")
    uri = pdf_path.resolve().as_uri()
    system_prompt = DATASHEET_TO_DECL_PROMPT + "\n\n" + DATASHEET_PDF_ITERATIVE_INSTR

    tools = _pdf_tool_definitions()
    dispatch = _pdf_tool_dispatch()

    initial_user = (
        "Convert this datasheet PDF into one DECL `component` (and optional `variant` blocks).\n\n"
        "**Mandatory:**\n"
        f"1. Call get_part_datasheet with this exact url string (copy it verbatim):\n{uri}\n"
        "2. Use read_datasheet_pages with the same url and 0-based page indices. Read multiple "
        "ranges until you have full pin names, power pins, and any digital buses (SPI/I2C/UART/USB).\n"
        "3. When the DECL is ready, your reply must end with a single ```decl code block only.\n"
        "4. You may call validate_decl on draft DECL text before the final block.\n\n"
        "Begin with get_part_datasheet now."
    )

    chat = None
    response: str | None = None
    last_sess_err: Exception | None = None
    for session_try in range(3):
        try:
            chat = create_chat(
                backend,
                system_prompt,
                tools,
                dispatch,
                model=model,
                ollama_url=ollama_url,
            )
        except Exception as exc:
            last_sess_err = exc
            logger.warning(
                "PDF convert: create_chat failed (%s), session %d/3",
                exc,
                session_try + 1,
            )
            chat = None
            continue
        try:
            response = chat.send(initial_user)
            break
        except Exception as exc:
            last_sess_err = exc
            logger.warning(
                "PDF convert: first LLM call failed (%s), session %d/3",
                exc,
                session_try + 1,
            )
            chat = None
            continue

    if chat is None or response is None:
        logger.error("Could not start PDF conversion after 3 tries: %s", last_sess_err)
        return None

    decl = _extract_decl(response)
    vbase = validation_base if validation_base is not None else _stdlib_validation_base()

    for attempt in range(max_attempts):
        iter_display = attempt + 1
        if decl:
            fixed = _fix_common_issues(decl)
            human, errors = validate_decl_structured(fixed, base_path=vbase)
            if not errors:
                if not quiet:
                    print(
                        f"\n=== datasheet_to_decl: validation OK after {iter_display} iteration(s) ===\n",
                        file=sys.stderr,
                    )
                return fixed
            if not quiet:
                _validation_trace_print("validation_failed", iter_display, human, errors)
            if attempt >= max_attempts - 1:
                logger.warning(
                    "VALIDATION still failing after %d attempts — last errors: %s",
                    max_attempts,
                    "; ".join(e.get("message", "") for e in errors[:8]),
                )
                return None
            payload = _repair_follow_payload(human, errors)
            follow = (
                "`decl check` / validate_decl reported errors. Fix EVERY issue and reply with "
                "ONE complete ```decl block only (no prose).\n\n"
                f"{payload}"
            )
            try:
                response = chat.send(follow)
            except Exception as exc:
                logger.warning("Repair LLM call failed (%s); retrying short nudge", exc)
                try:
                    response = chat.send(
                        "Output only the fixed complete DECL inside one ```decl fence. Nothing else."
                    )
                except Exception as exc2:
                    logger.error("Second LLM call failed: %s", exc2)
                    return None
            decl = _extract_decl(response)
            continue

        if attempt >= max_attempts - 1:
            logger.warning("No ```decl block after %d attempts", max_attempts)
            if not quiet:
                _validation_trace_print(
                    "missing_decl_fence",
                    iter_display,
                    "Last model reply had no extractable ```decl block.",
                    [],
                )
            return None
        if not quiet:
            _validation_trace_print(
                "missing_decl_fence",
                iter_display,
                "Nudging model: expected a ```decl fenced block in the reply.",
                [],
            )
        try:
            response = chat.send(
                "Your last message had no ```decl block. Read more pages if needed with "
                "read_datasheet_pages, then output exactly one ```decl fenced block with the component."
            )
        except Exception as exc:
            logger.warning("Follow-up LLM call failed (%s); retrying short nudge", exc)
            try:
                response = chat.send(
                    "Output one complete ```decl block only. Use read_datasheet_pages if you still need pin data."
                )
            except Exception as exc2:
                logger.error("Follow-up LLM call failed: %s", exc2)
                return None
        decl = _extract_decl(response)

    return None


def convert_datasheet_to_decl(
    datasheet_text: str,
    url: str = "",
    *,
    backend: str | None = None,
    model: str | None = None,
    ollama_url: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    quiet: bool = False,
    validation_base: Path | None = None,
) -> str | None:
    """
    Convert a plain-text datasheet excerpt to a valid DECL string (no PDF tools).
    Retries until valid or max_attempts.
    """
    if not datasheet_text or len(datasheet_text.strip()) < 100:
        logger.debug("Datasheet text too short to convert")
        return None
    backend = backend or getattr(config, "BACKEND", "ollama")
    prompt_prefix = (
        "Convert this datasheet excerpt to a single DECL component (and optional variants). "
        "Output ONLY a ```decl block.\n\n" + datasheet_text[:8000]
    )
    last_error: Exception | None = None
    vbase = validation_base if validation_base is not None else _stdlib_validation_base()

    for session_try in range(3):
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
                    if not quiet:
                        _validation_trace_print(
                            "text_path_missing_fence",
                            attempt,
                            feedback,
                            [],
                        )
                    if attempt == max_attempts:
                        break
                    continue
                decl = _fix_common_issues(decl)
                human_output, errors = validate_decl_structured(decl, base_path=vbase)
                if errors:
                    decl_dir = Path.home() / "decl"
                    decl_dir.mkdir(parents=True, exist_ok=True)
                    failed_path = decl_dir / "datasheet_failed.decl"
                    failed_path.write_text(decl)
                    if not quiet:
                        _validation_trace_print(
                            "text_path_validation_failed",
                            attempt,
                            human_output,
                            errors,
                        )
                    feedback = (
                        f"Validation failed. Failing DECL saved to {failed_path} "
                        "(line numbers below refer to this file).\n\n"
                        f"{human_output}"
                    )
                    logger.warning(
                        "Attempt %d/%d: validation failed (see %s): %s",
                        attempt,
                        max_attempts,
                        failed_path,
                        "; ".join(e.get("message", str(e)) for e in errors[:3]),
                    )
                    if attempt == max_attempts:
                        break
                    continue
                if not quiet:
                    print(
                        f"\n=== datasheet_to_decl: text path validation OK (attempt {attempt}) ===\n",
                        file=sys.stderr,
                    )
                return decl
            break
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Datasheet-to-DECL session try %d/3 failed: %s", session_try + 1, exc
            )
    if last_error:
        logger.warning("Giving up after exceptions: %s", last_error)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a datasheet excerpt to DECL (component + optional variants). "
        "PDFs use page-by-page datasheet tools; text uses a single excerpt.",
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
        help=f"Max conversion / repair attempts (default: {DEFAULT_MAX_ATTEMPTS})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-iteration validation traces on stderr (default: print all decl errors)",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    vbase = _stdlib_validation_base()

    inp = args.input
    if inp != "-":
        pdf_path = Path(inp)
        if pdf_path.suffix.lower() == ".pdf":
            if not pdf_path.is_file():
                logger.error("PDF not found: %s", pdf_path)
                return 1
            decl = convert_pdf_path_to_decl(
                pdf_path,
                backend=args.backend,
                model=args.model,
                ollama_url=args.ollama_url,
                max_attempts=args.max_attempts,
                quiet=args.quiet,
                validation_base=vbase,
            )
            if not decl:
                logger.error("Conversion failed")
                return 1
        else:
            try:
                text = _read_input_text(inp)
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
                quiet=args.quiet,
                validation_base=vbase,
            )
            if not decl:
                logger.error("Conversion failed")
                return 1
    else:
        try:
            text = _read_input_text("-")
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
            quiet=args.quiet,
            validation_base=vbase,
        )
        if not decl:
            logger.error("Conversion failed")
            return 1

    if args.save:
        import hashlib

        short_hash = hashlib.sha256((args.url or "").encode() or b"").hexdigest()[:12]
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
            Path(args.output).write_text(decl + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
