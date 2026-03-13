"""Core agent loop: planning, building, validation, and completeness checking."""

import json
import logging
import re
import sys
from pathlib import Path

from . import config
from .llm import OllamaChat
from .prompts import SYSTEM_PROMPT, PLANNING_PROMPT
from .tools import TOOL_DEFINITIONS, TOOL_DISPATCH, validate_decl

logger = logging.getLogger(__name__)

_DECL_FENCE_RE = re.compile(r"```(?:decl)?\s*\n(.*?)```", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def _extract_decl(text: str) -> str | None:
    """Extract the last fenced code block from the assistant's response."""
    matches = _DECL_FENCE_RE.findall(text)
    if not matches:
        return None
    candidate = matches[-1].strip()
    if not candidate:
        return None
    if any(kw in candidate for kw in ("component ", "schematic ", "protocol ", "variant ")):
        return candidate
    return None


def _extract_json(text: str) -> dict | None:
    """Try to extract a JSON object from the response."""
    for pattern in [_JSON_FENCE_RE, re.compile(r"\{.*\}", re.DOTALL)]:
        for match in pattern.findall(text):
            candidate = match.strip() if isinstance(match, str) else match
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except (json.JSONDecodeError, TypeError):
                continue
    return None


def _check_completeness(decl_code: str, inventory: list[dict]) -> list[str]:
    """Check which inventory items are missing from the generated .decl code."""
    code_lower = decl_code.lower()
    missing = []
    for item in inventory:
        name = item.get("name", "")
        check_terms = item.get("check_terms", [name])
        if not any(term.lower() in code_lower for term in check_terms):
            missing.append(f"- {name}: {item.get('purpose', '(no description)')}")
    return missing


# ---------------------------------------------------------------------------
# Phase 1: Planning -- extract requirements into a structured inventory
# ---------------------------------------------------------------------------

def _run_planning_phase(prompt: str, *, model: str | None, ollama_url: str | None) -> tuple[list[dict], str]:
    """Ask the LLM to extract a structured component inventory from the prompt.

    Returns (inventory_list, planning_text).
    """
    _log_step("Phase 1: Planning -- extracting component inventory")
    planner = OllamaChat(
        system_prompt=PLANNING_PROMPT,
        tools=[],
        tool_dispatch={},
        model=model,
        base_url=ollama_url,
    )

    response = planner.send(prompt)
    _log_step(f"Planning response ({len(response)} chars)")

    inventory = []
    obj = _extract_json(response)
    if obj and "inventory" in obj:
        inventory = obj["inventory"]
    elif obj and isinstance(obj, dict):
        for key in ("components", "items", "parts"):
            if key in obj and isinstance(obj[key], list):
                inventory = obj[key]
                break

    if not inventory:
        _log_step("WARNING: Could not parse structured inventory, using raw plan")
        return [], response

    _log_step(f"Extracted inventory: {len(inventory)} items")
    for item in inventory:
        _log_step(f"  - {item.get('name', '?')}: {item.get('purpose', '?')}")
    return inventory, response


# ---------------------------------------------------------------------------
# Phase 2: Building -- generate the .decl file with inventory enforcement
# ---------------------------------------------------------------------------

def _format_inventory_prompt(prompt: str, inventory: list[dict], plan_text: str) -> str:
    """Build the user message for the build phase, including the mandatory inventory."""
    lines = [
        f"Design request: {prompt}",
        "",
        "## MANDATORY COMPONENT INVENTORY",
        "",
        "You MUST include ALL of the following in your .decl output.",
        "Every item must appear as a `component` definition AND as an `instance` in the schematic.",
        "Do NOT skip any item. Do NOT substitute different parts than what is listed.",
        "If the user specified a particular MCU or IC, use EXACTLY that part.",
        "",
    ]
    for i, item in enumerate(inventory, 1):
        name = item.get("name", "?")
        purpose = item.get("purpose", "")
        search_hint = item.get("search_hint", "")
        lines.append(f"{i}. **{name}** -- {purpose}")
        if search_hint:
            lines.append(f"   Search hint: `{search_hint}`")
    lines.append("")
    lines.append(
        "## WORKFLOW\n"
        "1. Search parts DB for each non-passive component to get real pin info.\n"
        "2. Use `read_stdlib_file` for Resistor, Capacitor, LED, CH32V003 if available.\n"
        "3. For ICs: use ONLY the attributes/types supported by DECL (Resistance, Capacitance,\n"
        "   Voltage, Current, Power, Frequency, Percentage, DataSize, VoltageRange, Package, Color).\n"
        "   Do NOT invent new attribute types like Force, Distance, String.\n"
        "4. For pin names: use only alphanumeric characters and underscores. No # or special chars.\n"
        "5. Build the complete .decl, then validate_decl once.\n"
        "6. Fix any errors and output the final file in a ```decl code block."
    )
    return "\n".join(lines)


def run_agent(
    prompt: str,
    output_path: str = "output.decl",
    *,
    model: str | None = None,
    ollama_url: str | None = None,
) -> Path:
    """Run the circuit design agent and write the result to *output_path*."""

    # Phase 1: Planning
    inventory, plan_text = _run_planning_phase(
        prompt, model=model, ollama_url=ollama_url,
    )

    # Phase 2: Building
    _log_step("Phase 2: Building -- generating .decl file")
    chat = OllamaChat(
        system_prompt=SYSTEM_PROMPT,
        tools=TOOL_DEFINITIONS,
        tool_dispatch=TOOL_DISPATCH,
        model=model,
        base_url=ollama_url,
    )

    if inventory:
        build_prompt = _format_inventory_prompt(prompt, inventory, plan_text)
    else:
        build_prompt = prompt

    response = chat.send(build_prompt)

    # Phase 3: Validation loop with completeness checking
    for iteration in range(config.MAX_AGENT_ITERATIONS):
        decl_code = _extract_decl(response)

        if decl_code is None and chat.last_validated_content:
            _log_step(f"Iteration {iteration + 1}: using model's self-validated content")
            decl_code = chat.last_validated_content

        if decl_code is None:
            _log_step(f"Iteration {iteration + 1}: no .decl block yet, nudging model")
            response = chat.send(
                "Please produce the complete .decl file now. "
                "Put it inside a ```decl fenced code block. "
                "Make sure EVERY item from the mandatory inventory is included."
            )
            continue

        _log_step(f"Iteration {iteration + 1}: extracted .decl ({len(decl_code)} chars)")

        # Syntax/semantic validation
        validation = validate_decl(content=decl_code)
        _log_step(f"Validation: {validation[:200]}")

        is_ok = validation.strip().startswith("OK:")
        has_errors = "Error:" in validation and not is_ok

        if has_errors:
            _log_step(f"Iteration {iteration + 1}: syntax errors, feeding back")
            response = chat.send(
                f"The .decl file has validation errors. Fix them and output the "
                f"corrected file in a ```decl code block.\n\nValidation output:\n{validation}"
            )
            continue

        # Completeness check
        if inventory:
            missing = _check_completeness(decl_code, inventory)
            if missing and iteration < config.MAX_AGENT_ITERATIONS - 1:
                _log_step(f"Iteration {iteration + 1}: {len(missing)} missing components")
                for m in missing:
                    _log_step(f"  MISSING: {m}")
                missing_text = "\n".join(missing)
                response = chat.send(
                    f"The .decl file passes syntax validation but is INCOMPLETE. "
                    f"The following required components are MISSING from the schematic:\n\n"
                    f"{missing_text}\n\n"
                    f"You MUST search for these parts using `search_parts`, define them as "
                    f"components, and add them as instances in the schematic with proper "
                    f"connections. Output the complete corrected .decl in a ```decl code block."
                )
                continue

        if is_ok:
            out = Path(output_path)
            out.write_text(decl_code + "\n")
            warnings = [l for l in validation.splitlines() if l.startswith("Warning:")]
            print(f"\nCircuit written to {out}")
            if warnings:
                print(f"  ({len(warnings)} warning(s))")
            if inventory:
                final_missing = _check_completeness(decl_code, inventory)
                if final_missing:
                    print(f"  NOTE: {len(final_missing)} inventory item(s) may still be missing")
            return out

    _log_step("Max iterations reached")
    decl_code = _extract_decl(response) or (chat.last_validated_content if chat.last_validated_content else None)
    if decl_code:
        out = Path(output_path)
        out.write_text(decl_code + "\n")
        print(f"\nCircuit written to {out} (max iterations reached)")
        return out

    print("Error: agent did not produce a .decl file.", file=sys.stderr)
    sys.exit(1)


def _log_step(msg: str):
    logger.info(msg)
    print(f"  [{msg}]", file=sys.stderr)
