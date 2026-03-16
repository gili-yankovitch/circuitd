"""Core agent: 5-phase pipeline (requirements -> parts -> plan -> generate -> repair)."""

import hashlib
import json
import logging
import re
import sys
from pathlib import Path

from . import config
from .llm import create_chat
from .prompts import (
    REQUIREMENTS_PROMPT,
    PARTS_PROMPT,
    DESIGN_PLAN_PROMPT,
    DECL_GENERATION_PROMPT,
    REPAIR_PROMPT,
    COMPLETENESS_VERIFY_PROMPT,
    DATASHEET_TO_DECL_PROMPT,
)
from .tools import (
    TOOL_DEFINITIONS,
    TOOL_DISPATCH,
    validate_decl_structured,
    _fix_common_issues,
    save_to_stdlib,
    get_stdlib_component_decl,
)

logger = logging.getLogger(__name__)

_DECL_FENCE_RE = re.compile(r"```(?:decl)?\s*\n(.*?)```", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)

# Tools for Phase 2 only (parts discovery + save datasheet-derived DECL to stdlib)
PARTS_PHASE_TOOL_NAMES = (
    "list_stdlib", "read_stdlib_file", "search_parts", "get_part_datasheet", "save_to_stdlib",
)
PARTS_PHASE_TOOLS = [t for t in TOOL_DEFINITIONS if t["function"]["name"] in PARTS_PHASE_TOOL_NAMES]
PARTS_PHASE_DISPATCH = {k: v for k, v in TOOL_DISPATCH.items() if k in PARTS_PHASE_TOOL_NAMES}


def _convert_datasheet_to_decl_and_save(
    datasheet_text: str,
    url: str,
    *,
    backend: str,
    model: str | None,
    ollama_url: str | None,
) -> None:
    """Convert datasheet excerpt to DECL and save to stdlib. Logs success/failure; does not raise."""
    if not datasheet_text or len(datasheet_text.strip()) < 100:
        logger.debug("Datasheet text too short to convert")
        return
    short_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
    save_path = f"components/agent/datasheet_{short_hash}.decl"
    try:
        chat = create_chat(
            backend, DATASHEET_TO_DECL_PROMPT, tools=[], tool_dispatch={},
            model=model, ollama_url=ollama_url,
        )
        response = chat.send(
            "Convert this datasheet excerpt to a single DECL component (and optional variants). "
            "Output ONLY a ```decl block.\n\n" + datasheet_text[:8000]
        )
        decl = _extract_decl(response)
        if not decl:
            logger.warning("Datasheet-to-DECL: no decl block in LLM response")
            return
        decl = _fix_common_issues(decl)
        _, errors = validate_decl_structured(decl)
        if errors:
            logger.warning("Datasheet-to-DECL: validation failed for %s: %s", save_path, errors[:2])
            return
        result = save_to_stdlib(save_path, decl)
        out = json.loads(result) if result.strip().startswith("{") else {}
        if out.get("ok"):
            _log_step(f"Saved datasheet DECL to stdlib: {save_path}")
        else:
            logger.warning("Datasheet-to-DECL: save failed: %s", out.get("error", result))
    except Exception as exc:
        logger.warning("Datasheet-to-DECL failed: %s", exc)


def _parts_dispatch_with_auto_save(
    backend: str,
    model: str | None,
    ollama_url: str | None,
) -> dict:
    """Phase 2 tool dispatch that auto-converts every downloaded datasheet to DECL and saves."""
    dispatch = dict(PARTS_PHASE_DISPATCH)
    original_get = dispatch["get_part_datasheet"]

    def wrapped_get_part_datasheet(url: str) -> str:
        result = original_get(url=url)
        try:
            data = json.loads(result)
            if isinstance(data, dict) and "text" in data and "error" not in data:
                _convert_datasheet_to_decl_and_save(
                    data["text"],
                    data.get("url", url),
                    backend=backend,
                    model=model,
                    ollama_url=ollama_url,
                )
        except (json.JSONDecodeError, TypeError):
            pass
        return result

    dispatch["get_part_datasheet"] = wrapped_get_part_datasheet
    return dispatch


def _extract_decl(text: str) -> str | None:
    """Extract the last fenced decl block from the assistant's response."""
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


def _mandatory_inventory_from_requirements(requirements: dict) -> list[dict]:
    """Build inventory-like list for completeness check from requirements JSON."""
    inventory: list[dict] = []
    for item in requirements.get("explicit_parts", []):
        name = item.get("name") or item.get("part")
        if name:
            inventory.append({"name": name, "purpose": item.get("reason", ""), "check_terms": [name]})
    for item in requirements.get("implied_blocks", []):
        name = item.get("name")
        if name:
            inventory.append({"name": name, "purpose": item.get("reason", ""), "check_terms": [name]})
    for item in requirements.get("support_components", []):
        name = item.get("name")
        if name:
            inventory.append({"name": name, "purpose": item.get("reason", ""), "check_terms": [name]})
    return inventory


def _normalize_for_match(text: str) -> str:
    """Collapse whitespace, underscores, and hyphens so 'Green LED' matches 'Green_LED'."""
    return re.sub(r"[\s_\-]+", "", text).lower()


def _check_completeness(decl_code: str, inventory: list[dict]) -> list[str]:
    """Check which inventory items are missing from the generated .decl code (heuristic: substring match)."""
    code_lower = decl_code.lower()
    code_normalized = _normalize_for_match(decl_code)
    missing = []
    for item in inventory:
        name = item.get("name", "")
        check_terms = item.get("check_terms", [name])
        found = any(
            term.lower() in code_lower or _normalize_for_match(term) in code_normalized
            for term in check_terms
        )
        if not found:
            missing.append(f"- {name}: {item.get('purpose', '(no description)')}")
    return missing


def _verify_completeness_with_llm(
    decl_code: str,
    heuristic_missing: list[str],
    requirements: dict,
    *,
    backend: str,
    model: str | None,
    ollama_url: str | None,
) -> list[str]:
    """Ask LLM which heuristic 'missing' items are actually missing vs already present in DECL.
    Returns the list of items that are actually missing (for the add-missing step).
    """
    if not heuristic_missing:
        return []
    _log_phase_to_prompts_file("Completeness: LLM verification of missing items")
    chat = create_chat(
        backend, COMPLETENESS_VERIFY_PROMPT, tools=[], tool_dispatch={},
        model=model, ollama_url=ollama_url,
    )
    missing_blob = "\n".join(heuristic_missing)
    user_msg = (
        "Flagged as missing (from a simple text match; may be false positives):\n\n"
        f"{missing_blob}\n\n"
        "Requirements summary (for context):\n"
        + json.dumps(
            {
                "explicit_parts": requirements.get("explicit_parts", []),
                "implied_blocks": requirements.get("implied_blocks", []),
                "support_components": requirements.get("support_components", []),
            },
            indent=2,
        )
        + "\n\nDECL file to check:\n\n```decl\n"
        + decl_code
        + "\n```\n\n"
        "Output ONLY a JSON object with 'actually_missing' and 'already_present' arrays."
    )
    response = chat.send(user_msg)
    obj = _extract_json(response)
    if not obj or not isinstance(obj, dict):
        _log_step("Completeness verification: no valid JSON; treating all flagged as already present")
        return []
    actually_missing = obj.get("actually_missing") or []
    if not isinstance(actually_missing, list):
        actually_missing = []
    already_present = obj.get("already_present") or []
    if isinstance(already_present, list) and already_present:
        _log_step(f"Completeness verification: {len(already_present)} item(s) already present in DECL")
    if actually_missing:
        _log_step(f"Completeness verification: {len(actually_missing)} item(s) actually missing")
    return actually_missing


def _log_step(msg: str) -> None:
    logger.info(msg)
    print(f"  [{msg}]", file=sys.stderr)


def _log_phase_to_prompts_file(phase_label: str) -> None:
    """Write a phase header to the prompts log file so the next LLM block is identifiable."""
    path = getattr(config, "PROMPTS_LOG_PATH", None)
    if path is None:
        return
    try:
        from datetime import datetime, timezone
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n")
            f.write("# " + "=" * 76 + "\n")
            f.write(f"# AGENT PHASE: {phase_label}\n")
            f.write("# " + "=" * 76 + "\n\n")
    except OSError as exc:
        logger.debug("Failed to write phase to prompts log: %s", exc)


# ---------------------------------------------------------------------------
# Phase 1: Requirements extraction
# ---------------------------------------------------------------------------

def _run_phase1_requirements(
    user_prompt: str,
    *,
    backend: str,
    model: str | None,
    ollama_url: str | None,
) -> dict:
    _log_step("Phase 1: Requirements extraction")
    _log_phase_to_prompts_file("Phase 1: Requirements extraction")
    chat = create_chat(
        backend, REQUIREMENTS_PROMPT, tools=[], tool_dispatch={},
        model=model, ollama_url=ollama_url,
    )
    response = chat.send(user_prompt)
    obj = _extract_json(response)
    if not obj or not isinstance(obj, dict):
        _log_step("WARNING: No valid requirements JSON; using minimal brief")
        return {
            "functions": [],
            "constraints": [],
            "explicit_parts": [],
            "implied_blocks": [],
            "power_requirements": [],
            "interfaces": [],
            "support_components": [],
            "assumptions": [],
            "open_questions": [],
        }
    _log_step(f"Requirements: {len(obj.get('explicit_parts', []))} explicit, {len(obj.get('implied_blocks', []))} implied")
    return obj


# ---------------------------------------------------------------------------
# Phase 2: Part / stdlib discovery (with tool loop)
# ---------------------------------------------------------------------------

def _run_phase2_parts(
    requirements: dict,
    *,
    backend: str,
    model: str | None,
    ollama_url: str | None,
) -> dict:
    _log_step("Phase 2: Part selection / library discovery")
    _log_phase_to_prompts_file("Phase 2: Part selection / library discovery")
    dispatch = _parts_dispatch_with_auto_save(backend=backend, model=model, ollama_url=ollama_url)
    chat = create_chat(
        backend, PARTS_PROMPT, PARTS_PHASE_TOOLS, dispatch,
        model=model, ollama_url=ollama_url,
    )
    user_msg = (
        "Design brief (use the tools to look up stdlib and parts, then output the selection JSON):\n"
        + json.dumps(requirements, indent=2)
    )
    response = chat.send(user_msg)
    obj = _extract_json(response)
    if not obj or not isinstance(obj, dict):
        _log_step("WARNING: No valid parts JSON; using empty selection")
        return {"selected_components": [], "nets_needed": [], "design_rules": [], "unresolved": []}
    _log_step(f"Parts: {len(obj.get('selected_components', []))} components selected")
    return obj


# ---------------------------------------------------------------------------
# Phase 3: Design plan
# ---------------------------------------------------------------------------

def _run_phase3_design_plan(
    requirements: dict,
    parts: dict,
    *,
    backend: str,
    model: str | None,
    ollama_url: str | None,
) -> dict:
    _log_step("Phase 3: Design plan (connection plan)")
    _log_phase_to_prompts_file("Phase 3: Design plan (connection plan)")
    chat = create_chat(
        backend, DESIGN_PLAN_PROMPT, tools=[], tool_dispatch={},
        model=model, ollama_url=ollama_url,
    )
    user_msg = (
        "Requirements and selected parts. Output the connection plan JSON only.\n\n"
        "Requirements:\n" + json.dumps(requirements, indent=2) + "\n\n"
        "Parts:\n" + json.dumps(parts, indent=2)
    )
    response = chat.send(user_msg)
    obj = _extract_json(response)
    if not obj or not isinstance(obj, dict):
        _log_step("WARNING: No valid design plan JSON")
        return {"instances": [], "nets": [], "connections": [], "power_topology": [], "protocol_bindings": [], "checks": []}
    _log_step(f"Plan: {len(obj.get('instances', []))} instances, {len(obj.get('nets', []))} nets")
    return obj


# ---------------------------------------------------------------------------
# Phase 4: DECL generation
# ---------------------------------------------------------------------------

def _run_phase4_generate_decl(
    design_plan: dict,
    *,
    backend: str,
    model: str | None,
    ollama_url: str | None,
) -> str | None:
    _log_step("Phase 4: DECL generation")
    _log_phase_to_prompts_file("Phase 4: DECL generation")
    chat = create_chat(
        backend, DECL_GENERATION_PROMPT, tools=[], tool_dispatch={},
        model=model, ollama_url=ollama_url,
    )
    user_msg = (
        "Emit only a ```decl code block containing the complete .decl file for this plan:\n\n"
        + json.dumps(design_plan, indent=2)
    )
    # Include stdlib component definitions so the LLM uses correct protocol pin mappings (e.g. SPI CLK/MOSI/MISO/SS)
    comp_types = set()
    for inst in design_plan.get("instances", []):
        c = inst.get("component")
        if isinstance(c, str):
            comp_types.add(c)
    stdlib_refs: list[str] = []
    for comp_name in sorted(comp_types):
        decl_content = get_stdlib_component_decl(comp_name)
        if decl_content:
            stdlib_refs.append(f"\n--- stdlib reference for {comp_name} (use these pins for protocol connections) ---\n{decl_content[:12000]}")
    if stdlib_refs:
        user_msg += "\n\nReference definitions from stdlib (protocol pin mappings must be respected):" + "\n".join(stdlib_refs)
    response = chat.send(user_msg)
    decl = _extract_decl(response)
    if decl:
        decl = _fix_common_issues(decl)
        _log_step(f"Generated .decl ({len(decl)} chars)")
    else:
        _log_step("WARNING: No decl block in response")
    return decl


# ---------------------------------------------------------------------------
# Phase 5: Validator repair loop
# ---------------------------------------------------------------------------

def _run_phase5_repair_loop(
    decl_code: str,
    *,
    backend: str,
    model: str | None,
    ollama_url: str | None,
) -> str:
    _log_step("Phase 5: Validation and repair")
    _log_phase_to_prompts_file("Phase 5: Validation and repair")
    chat = create_chat(
        backend, REPAIR_PROMPT, tools=[], tool_dispatch={},
        model=model, ollama_url=ollama_url,
    )
    decl = decl_code
    for iteration in range(config.MAX_AGENT_ITERATIONS):
        human_output, structured_errors = validate_decl_structured(decl)
        is_ok = human_output.strip().startswith("OK:")

        if is_ok:
            _log_step("Validation passed")
            return decl

        _log_step(f"Repair iteration {iteration + 1}: {len(structured_errors)} structured errors")
        err_blob = human_output
        if structured_errors:
            err_blob += "\n\nStructured errors (for repair):\n" + json.dumps(structured_errors, indent=2)
        user_msg = (
            "Fix the following DECL so it passes validation. Output ONLY a ```decl block.\n\n"
            "Validation output:\n" + err_blob + "\n\n"
            "Current DECL (fix the errors in it):\n\n```decl\n" + decl + "\n```"
        )
        response = chat.send(user_msg)
        repaired = _extract_decl(response)
        if not repaired:
            _log_step("No decl block in repair response; keeping last version")
            break
        decl = _fix_common_issues(repaired)
    return decl


# ---------------------------------------------------------------------------
# Main entry: 5-phase pipeline
# ---------------------------------------------------------------------------

def run_agent(
    prompt: str,
    output_path: str = "output.decl",
    *,
    backend: str | None = None,
    model: str | None = None,
    ollama_url: str | None = None,
) -> Path:
    """Run the 5-phase circuit design agent and write the result to *output_path*."""
    backend = backend or config.BACKEND

    # Phase 1: Requirements
    requirements = _run_phase1_requirements(
        prompt, backend=backend, model=model, ollama_url=ollama_url,
    )

    # Phase 2: Parts (with tools)
    parts = _run_phase2_parts(
        requirements, backend=backend, model=model, ollama_url=ollama_url,
    )

    # Phase 3: Design plan
    design_plan = _run_phase3_design_plan(
        requirements, parts, backend=backend, model=model, ollama_url=ollama_url,
    )

    # Phase 4: Generate DECL
    decl_code = _run_phase4_generate_decl(
        design_plan, backend=backend, model=model, ollama_url=ollama_url,
    )
    if not decl_code:
        print("Error: agent did not produce a .decl file.", file=sys.stderr)
        sys.exit(1)

    # Phase 5: Validate and repair until OK
    decl_code = _run_phase5_repair_loop(
        decl_code, backend=backend, model=model, ollama_url=ollama_url,
    )

    # Completeness loop: heuristic flags possible missing items; LLM verifies which are actually missing
    inventory = _mandatory_inventory_from_requirements(requirements)
    for completeness_iteration in range(config.MAX_AGENT_ITERATIONS):
        if not inventory:
            break
        heuristic_missing = _check_completeness(decl_code, inventory)
        if not heuristic_missing:
            break
        _log_step(f"Completeness: heuristic flagged {len(heuristic_missing)} item(s); verifying with LLM")
        actually_missing = _verify_completeness_with_llm(
            decl_code, heuristic_missing, requirements,
            backend=backend, model=model, ollama_url=ollama_url,
        )
        if not actually_missing:
            _log_step("Completeness: all flagged items already present in DECL; done")
            break
        _log_step(f"Completeness: {len(actually_missing)} required item(s) actually missing; asking LLM to add them")
        for m in actually_missing:
            _log_step(f"  {m}")
        _log_phase_to_prompts_file("Completeness: add missing components")
        chat = create_chat(
            backend, REPAIR_PROMPT, tools=[], tool_dispatch={},
            model=model, ollama_url=ollama_url,
        )
        missing_text = "\n".join(actually_missing)
        user_msg = (
            "The .decl file below is valid but INCOMPLETE. The following required components "
            "are MISSING from the schematic:\n\n"
            f"{missing_text}\n\n"
            "Add these components (define them as component if needed), add instances for them, "
            "and connect them appropriately. Output the complete corrected .decl in a ```decl block.\n\n"
            "Current .decl:\n\n"
            f"```decl\n{decl_code}\n```"
        )
        response = chat.send(user_msg)
        new_decl = _extract_decl(response)
        if not new_decl:
            _log_step("No decl block in response; keeping current version")
            break
        decl_code = _fix_common_issues(new_decl)
        # Re-validate in case the LLM introduced errors
        decl_code = _run_phase5_repair_loop(
            decl_code, backend=backend, model=model, ollama_url=ollama_url,
        )

    out = Path(output_path)
    out.write_text(decl_code + "\n")
    print(f"\nCircuit written to {out}")
    return out
