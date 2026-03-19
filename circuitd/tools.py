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

# import "path" — relative to current file's directory (plus optional stdlib fallback)
_IMPORT_QUOTED_RE = re.compile(r'^\s*import\s+"([^"]+)"\s*$')
# import <path> — stdlib root only (C-style system include)
_IMPORT_SYSTEM_RE = re.compile(r"^\s*import\s+<([^>]+)>\s*$")


class DeclImportError(FileNotFoundError):
    """An import path could not be resolved to an existing file."""

    def __init__(self, import_path: str, tried: list[Path], *, system: bool = False):
        self.import_path = import_path
        self.tried = tried
        self.system = system
        paths = ", ".join(str(p) for p in tried) if tried else "(no paths)"
        kind = f"<{import_path}>" if system else f'"{import_path}"'
        super().__init__(f"Import not found: {kind} (tried: {paths})")


def _resolve_import_path(
    path_str: str,
    base_path: Path,
    stdlib_root: Path | None,
    *,
    is_system: bool,
) -> Path:
    """Resolve ``import <path>`` (stdlib root) or ``import \"path\"`` (relative + fallback)."""
    path_str = path_str.strip().replace("\\", "/")
    if is_system:
        if stdlib_root is None or not Path(stdlib_root).is_dir():
            raise DeclImportError(path_str, [], system=True)
        p = (Path(stdlib_root).resolve() / path_str).resolve()
        if p.is_file():
            return p
        raise DeclImportError(path_str, [p], system=True)
    primary = (base_path / path_str).resolve()
    if primary.is_file():
        return primary
    tried = [primary]
    if stdlib_root is not None:
        alt = (Path(stdlib_root).resolve() / path_str).resolve()
        if alt not in tried:
            tried.append(alt)
            if alt.is_file():
                return alt
    raise DeclImportError(path_str, tried, system=False)

# Protocol-pin validation: extract protocols, external features, and connections from DECL text
_LINE_TO_PIN_RE = re.compile(r"(\w+)\s*->\s*pin\s+(\w+)")
_VARIANT_OF_RE = re.compile(r"variant\s+(\w+)\s+of\s+(\w+)")
_INSTANCE_RE = re.compile(r"instance\s+(\w+)\s*:\s*(\w+)")
_CONNECT_RE = re.compile(
    r"connect\s+(\w+)\.(\w+)\s+--\s+(?:net\s+(\w+)|(\w+)\.(\w+))"
)
# Protocol: rules block "role1.LINE1 -- role2.LINE2"
_WIRING_RULE_RE = re.compile(r"(\w+)\.(\w+)\s*--\s*(\w+)\.(\w+)")
# External feature: any protocol and role
_EXTERNAL_FEATURE_RE = re.compile(
    r"external\s+\w+\s+using\s+protocol\s+(\w+)\s+role\s+(\w+)\s*\{([^}]+)\}",
    re.DOTALL,
)

# Fallback when one side has no protocol feature: treat these pin names as protocol line aliases (e.g. flash DI/DO/CS)
_PROTOCOL_LINE_ALIASES: dict[str, str] = {"DI": "MOSI", "DO": "MISO", "CS": "SS"}


def _extract_line_to_pin_from_block(block: str) -> dict[str, str]:
    """Parse 'CLK -> pin PC5  MOSI -> pin PC6' into { CLK: PC5, MOSI: PC6 }."""
    out: dict[str, str] = {}
    for m in _LINE_TO_PIN_RE.finditer(block):
        line_name, pin_name = m.group(1), m.group(2)
        out[line_name.upper()] = pin_name
    return out


def _extract_protocols_from_decl(decl: str) -> dict[str, tuple[frozenset[tuple[str, str, str, str]], frozenset[str]]]:
    """Extract protocol name -> (rules_set, line_names). rules_set = (role_a, line_a, role_b, line_b)."""
    result: dict[str, tuple[set[tuple[str, str, str, str]], set[str]]] = {}
    proto_block_re = re.compile(r"protocol\s+(\w+)\s*\{", re.MULTILINE)
    pos = 0
    while True:
        m = proto_block_re.search(decl, pos)
        if not m:
            break
        proto_name = m.group(1)
        start = m.end()
        depth = 1
        i = start
        while i < len(decl) and depth > 0:
            if decl[i] == "{":
                depth += 1
            elif decl[i] == "}":
                depth -= 1
            i += 1
        block = decl[start : i - 1]
        rules_set: set[tuple[str, str, str, str]] = set()
        line_names: set[str] = set()
        rules_m = re.search(r"rules\s*\{([^}]+)\}", block, re.DOTALL)
        if rules_m:
            for w in _WIRING_RULE_RE.finditer(rules_m.group(1)):
                r1, l1, r2, l2 = w.group(1), w.group(2).upper(), w.group(3), w.group(4).upper()
                rules_set.add((r1, l1, r2, l2))
                line_names.add(l1)
                line_names.add(l2)
        result[proto_name] = (frozenset(rules_set), frozenset(line_names))
        pos = i
    return {k: (v[0], v[1]) for k, v in result.items()}


def _extract_component_features_and_variants(
    decl: str,
) -> tuple[dict[str, list[dict]], dict[str, str]]:
    """Extract component name -> list of {protocol, role, line_to_pin}, and variant -> base."""
    comp_features: dict[str, list[dict]] = {}
    variant_base: dict[str, str] = {}
    comp_block_re = re.compile(r"component\s+(\w+)\s*\{", re.MULTILINE)
    pos = 0
    while True:
        m = comp_block_re.search(decl, pos)
        if not m:
            break
        comp_name = m.group(1)
        start = m.end()
        depth = 1
        i = start
        while i < len(decl) and depth > 0:
            if decl[i] == "{":
                depth += 1
            elif decl[i] == "}":
                depth -= 1
            i += 1
        block = decl[start : i - 1]
        for ext in _EXTERNAL_FEATURE_RE.finditer(block):
            proto, role = ext.group(1), ext.group(2)
            line_to_pin = _extract_line_to_pin_from_block(ext.group(3))
            if line_to_pin:
                comp_features.setdefault(comp_name, []).append({
                    "protocol": proto,
                    "role": role,
                    "line_to_pin": line_to_pin,
                })
        pos = i
    for m in _VARIANT_OF_RE.finditer(decl):
        variant_base[m.group(1)] = m.group(2)
    return comp_features, variant_base


def _load_stdlib_protocols_and_features_for_component(
    component_name: str,
) -> tuple[dict[str, list[dict]], dict[str, str], dict[str, tuple[frozenset[tuple[str, str, str, str]], frozenset[str]]]]:
    """Load from stdlib: comp_features, variant_base, and protocols for this component's file."""
    stdlib = getattr(config, "STDLIB_PATH", None)
    if not stdlib or not Path(stdlib).is_dir():
        return {}, {}, {}
    comp_features: dict[str, list[dict]] = {}
    variant_base: dict[str, str] = {}
    protocols: dict[str, tuple[frozenset[tuple[str, str, str, str]], frozenset[str]]] = {}
    comp_or_variant_re = re.compile(
        rf"(?:component|variant)\s+{re.escape(component_name)}\s+(?:\{{|of\s+)",
        re.MULTILINE,
    )
    for path in Path(stdlib).rglob("*.decl"):
        try:
            content = path.read_text()
        except OSError:
            continue
        if not comp_or_variant_re.search(content):
            continue
        root = Path(stdlib).resolve()
        pdir = path.parent.resolve()
        try:
            content = _expand_decl_imports(content, pdir, stdlib_root=root)
        except DeclImportError:
            continue
        protocols.update(_extract_protocols_from_decl(content))
        c_f, v_b = _extract_component_features_and_variants(content)
        comp_features.update(c_f)
        variant_base.update(v_b)
        break
    return comp_features, variant_base, protocols


def get_stdlib_component_decl(component_name: str) -> str | None:
    """Return expanded DECL content of the stdlib file that defines this component or variant."""
    stdlib = getattr(config, "STDLIB_PATH", None)
    if not stdlib or not Path(stdlib).is_dir():
        return None
    comp_or_variant_re = re.compile(
        rf"(?:component|variant)\s+{re.escape(component_name)}\s+(?:\{{|of\s+)",
        re.MULTILINE,
    )
    for path in Path(stdlib).rglob("*.decl"):
        try:
            content = path.read_text()
        except OSError:
            continue
        if comp_or_variant_re.search(content):
            root = Path(stdlib).resolve()
            pdir = path.parent.resolve()
            try:
                return _expand_decl_imports(content, pdir, stdlib_root=root)
            except DeclImportError:
                continue
    return None


def _extract_instances_and_connections(decl: str) -> tuple[dict[str, str], list[tuple[str, str, str, str]]]:
    """Return (inst_name -> component_type, list of (inst1, pin1, inst2, pin2))."""
    instances: dict[str, str] = {}
    for m in _INSTANCE_RE.finditer(decl):
        instances[m.group(1)] = m.group(2)
    pairs: list[tuple[str, str, str, str]] = []
    nets: dict[str, set[tuple[str, str]]] = {}
    for m in _CONNECT_RE.finditer(decl):
        inst1, pin1 = m.group(1), m.group(2)
        net_name = m.group(3)
        inst2, pin2 = m.group(4), m.group(5)
        if inst2 and pin2:
            pairs.append((inst1, pin1, inst2, pin2))
        elif net_name:
            nets.setdefault(net_name, set()).add((inst1, pin1))
    for endpoints in nets.values():
        el = list(endpoints)
        for i in range(len(el)):
            for j in range(i + 1, len(el)):
                pairs.append((el[i][0], el[i][1], el[j][0], el[j][1]))
    return instances, pairs


def _line_for_pin(pin: str, line_to_pin: dict[str, str]) -> str | None:
    """Return protocol line name if pin is in line_to_pin mapping."""
    pin_upper = pin.upper()
    for line, p in line_to_pin.items():
        if p.upper() == pin_upper:
            return line.upper()
    return None


def _pin_alias_to_line(pin_or_line: str, protocol_lines: frozenset[str]) -> str | None:
    """Map pin/line name to protocol line (e.g. DI->MOSI). Returns None if not a known line or alias."""
    u = pin_or_line.upper()
    if u in protocol_lines:
        return u
    return _PROTOCOL_LINE_ALIASES.get(u)


def _rules_allow_connection(
    proto_name: str,
    role1: str,
    line1: str,
    role2: str,
    line2: str,
    protocols: dict[str, tuple[frozenset[tuple[str, str, str, str]], frozenset[str]]],
) -> bool:
    """True if protocol rules say role1.line1 connects to role2.line2."""
    entry = protocols.get(proto_name)
    if not entry:
        return False
    rules, _ = entry
    return (role1, line1, role2, line2) in rules or (role2, line2, role1, line1) in rules


def validate_decl_protocol_pins(decl_content: str) -> list[dict]:
    """Check that protocol pins are connected according to the protocol rules from the DECL.

    Uses protocol definitions (rules { role1.L1 -- role2.L2 }) and component external
    features (external X using protocol P role R { L -> pin PIN }) from the decl and stdlib.
    Returns structured errors (code E011) for repair.
    """
    errors: list[dict] = []
    decl = _fix_common_issues(decl_content)
    protocols = _extract_protocols_from_decl(decl)
    comp_features, variant_base = _extract_component_features_and_variants(decl)
    instances, pairs = _extract_instances_and_connections(decl)
    stdlib_path = getattr(config, "STDLIB_PATH", None)

    def get_features(inst_name: str) -> list[dict]:
        comp_type = instances.get(inst_name)
        if not comp_type:
            return []
        base = variant_base.get(comp_type) or comp_type
        feats = comp_features.get(base) or comp_features.get(comp_type)
        if feats:
            return feats
        if stdlib_path:
            s_feat, s_var, s_proto = _load_stdlib_protocols_and_features_for_component(comp_type)
            protocols.update(s_proto)
            b = s_var.get(comp_type) or comp_type
            feats = s_feat.get(b) or s_feat.get(comp_type)
            if feats:
                return feats
        return []

    for (inst1, pin1, inst2, pin2) in pairs:
        feats1 = get_features(inst1)
        feats2 = get_features(inst2)
        if not feats1 and not feats2:
            continue

        # Resolve pin to (protocol, role, line) for each side
        def resolve_pin(feats: list[dict], pin: str) -> list[tuple[str, str, str, dict[str, str]]]:
            out = []
            for f in feats:
                line = _line_for_pin(pin, f["line_to_pin"])
                if line is not None:
                    out.append((f["protocol"], f["role"], line, f["line_to_pin"]))
            return out

        r1 = resolve_pin(feats1, pin1)
        r2 = resolve_pin(feats2, pin2)

        # Both sides have a protocol mapping for this pin: validate with protocol rules
        if r1 and r2:
            for (p1, role1, line1, _) in r1:
                for (p2, role2, line2, _) in r2:
                    if p1 != p2:
                        continue
                    if p1 not in protocols:
                        continue
                    if not _rules_allow_connection(p1, role1, line1, role2, line2, protocols):
                        errors.append({
                            "code": "E011",
                            "line": 0,
                            "message": (
                                f"Protocol pin mismatch: {inst1}.{pin1} ({p1} {role1}.{line1}) connected to "
                                f"{inst2}.{pin2} ({p2} {role2}.{line2}). Connection is not allowed by protocol {p1} rules."
                            ),
                            "entities": [f"{inst1}.{pin1}", f"{inst2}.{pin2}"],
                        })
            continue

        # One side has mapping, other does not: fallback using line aliases (e.g. flash DI/DO/CS)
        pin2_line = pin2.upper()
        pin1_line = pin1.upper()
        for (proto_name, role1, line1, line_to_pin1) in r1:
            entry = protocols.get(proto_name)
            proto_lines = entry[1] if entry else frozenset()
            other_line = _pin_alias_to_line(pin2_line, proto_lines) or (pin2_line if pin2_line in proto_lines else None)
            if other_line and other_line != line1:
                errors.append({
                    "code": "E011",
                    "line": 0,
                    "message": (
                        f"Protocol pin mismatch: {inst1}.{pin1} is {proto_name} {line1} but connected to {inst2}.{pin2}. "
                        f"{proto_name} {line1} should connect to the matching line on the other device."
                    ),
                    "entities": [f"{inst1}.{pin1}", f"{inst2}.{pin2}"],
                })
            elif not other_line and pin2_line in (proto_lines | frozenset(_PROTOCOL_LINE_ALIASES.keys())):
                other_line = _pin_alias_to_line(pin2_line, proto_lines) or pin2_line
                if other_line != line1:
                    errors.append({
                        "code": "E011",
                        "line": 0,
                        "message": (
                            f"Protocol pin mismatch: {inst1}.{pin1} is {proto_name} {line1} but connected to {inst2}.{pin2}. "
                            f"Expected matching {proto_name} line (e.g. {line1}–{line1})."
                        ),
                        "entities": [f"{inst1}.{pin1}", f"{inst2}.{pin2}"],
                    })
            # Mapped side using wrong physical pin for this line
            expected_pin = line_to_pin1.get(line1)
            if expected_pin and expected_pin.upper() != pin1.upper() and other_line == line1:
                errors.append({
                    "code": "E011",
                    "line": 0,
                    "message": (
                        f"Wrong pin on {inst1}: {inst2}.{pin2} ({proto_name} {line1}) is connected to {inst1}.{pin1}, "
                        f"but {inst1} should use {expected_pin} for {proto_name} {line1} (see protocol pin mapping)."
                    ),
                    "entities": [f"{inst1}.{pin1}", f"{inst2}.{pin2}"],
                })
            break

        for (proto_name, role2, line2, line_to_pin2) in r2:
            entry = protocols.get(proto_name)
            proto_lines = entry[1] if entry else frozenset()
            other_line = _pin_alias_to_line(pin1_line, proto_lines) or (pin1_line if pin1_line in proto_lines else None)
            if other_line and other_line != line2:
                errors.append({
                    "code": "E011",
                    "line": 0,
                    "message": (
                        f"Protocol pin mismatch: {inst2}.{pin2} is {proto_name} {line2} but connected to {inst1}.{pin1}. "
                        f"{proto_name} {line2} should connect to the matching line on the other device."
                    ),
                    "entities": [f"{inst1}.{pin1}", f"{inst2}.{pin2}"],
                })
            elif not other_line and pin1_line in (proto_lines | frozenset(_PROTOCOL_LINE_ALIASES.keys())):
                other_line = _pin_alias_to_line(pin1_line, proto_lines) or pin1_line
                if other_line != line2:
                    errors.append({
                        "code": "E011",
                        "line": 0,
                        "message": (
                            f"Protocol pin mismatch: {inst2}.{pin2} is {proto_name} {line2} but connected to {inst1}.{pin1}. "
                            f"Use the correct {proto_name} pin for {inst1} (check protocol pin mapping)."
                        ),
                        "entities": [f"{inst1}.{pin1}", f"{inst2}.{pin2}"],
                    })
            expected_pin = line_to_pin2.get(line2)
            if expected_pin and expected_pin.upper() != pin2.upper() and other_line == line2:
                errors.append({
                    "code": "E011",
                    "line": 0,
                    "message": (
                        f"Wrong pin on {inst2}: {inst1}.{pin1} ({proto_name} {line2}) is connected to {inst2}.{pin2}, "
                        f"but {inst2} should use {expected_pin} for {proto_name} {line2}."
                    ),
                    "entities": [f"{inst1}.{pin1}", f"{inst2}.{pin2}"],
                })
            break

        # One side has features but neither pin is in a mapping: check "wrong physical pin" (e.g. PA5 vs PC5 for CLK)
        if not r1 and feats1 and pin2_line in frozenset(_PROTOCOL_LINE_ALIASES.keys()) | frozenset(("CLK", "MOSI", "MISO", "SS")):
            for f in feats1:
                proto_name = f["protocol"]
                entry = protocols.get(proto_name)
                proto_lines = entry[1] if entry else frozenset()
                canonical = _pin_alias_to_line(pin2_line, proto_lines) or pin2_line
                expected_pin = f["line_to_pin"].get(canonical)
                if expected_pin and expected_pin.upper() != pin1.upper():
                    errors.append({
                        "code": "E011",
                        "line": 0,
                        "message": (
                            f"Wrong pin on {inst1}: {inst2}.{pin2} ({proto_name} {canonical}) is connected to {inst1}.{pin1}, "
                            f"but {inst1} should use {expected_pin} for {proto_name} {canonical} (see protocol pin mapping)."
                        ),
                        "entities": [f"{inst1}.{pin1}", f"{inst2}.{pin2}"],
                    })
                    break
        if not r2 and feats2 and pin1_line in frozenset(_PROTOCOL_LINE_ALIASES.keys()) | frozenset(("CLK", "MOSI", "MISO", "SS")):
            for f in feats2:
                proto_name = f["protocol"]
                entry = protocols.get(proto_name)
                proto_lines = entry[1] if entry else frozenset()
                canonical = _pin_alias_to_line(pin1_line, proto_lines) or pin1_line
                expected_pin = f["line_to_pin"].get(canonical)
                if expected_pin and expected_pin.upper() != pin2.upper():
                    errors.append({
                        "code": "E011",
                        "line": 0,
                        "message": (
                            f"Wrong pin on {inst2}: {inst1}.{pin1} ({proto_name} {canonical}) is connected to {inst2}.{pin2}, "
                            f"but {inst2} should use {expected_pin} for {proto_name} {canonical}."
                        ),
                        "entities": [f"{inst1}.{pin1}", f"{inst2}.{pin2}"],
                    })
                    break

    return errors


def _expand_decl_imports(
    content: str,
    base_path: Path,
    visited: set[Path] | None = None,
    *,
    stdlib_root: Path | None = None,
) -> str:
    """Resolve imports by reading and inlining imported files.

    - ``import <path>`` — only under ``stdlib_root`` (DECL stdlib layout).
    - ``import \"path\"`` — relative to ``base_path`` (current file's directory), then
      under ``stdlib_root`` if not found (compatibility).

    Missing imports raise :exc:`DeclImportError`.

    Circular imports are skipped (each file inlined at most once).
    """
    if visited is None:
        visited = set()
    base_path = base_path.resolve()
    if not base_path.is_dir():
        base_path = base_path.parent
    sr = Path(stdlib_root).resolve() if stdlib_root is not None else None
    out_lines: list[str] = []
    for line in content.splitlines():
        sm = _IMPORT_SYSTEM_RE.match(line)
        qm = _IMPORT_QUOTED_RE.match(line) if not sm else None
        if sm:
            path_str = sm.group(1).strip()
            resolved = _resolve_import_path(path_str, base_path, sr, is_system=True)
        elif qm:
            path_str = qm.group(1).strip()
            resolved = _resolve_import_path(path_str, base_path, sr, is_system=False)
        else:
            out_lines.append(line)
            continue
        if resolved in visited:
            continue
        visited.add(resolved)
        try:
            sub_content = resolved.read_text()
        except OSError as exc:
            raise DeclImportError(path_str, [resolved], system=bool(sm)) from exc
        expanded_sub = _expand_decl_imports(
            sub_content,
            resolved.parent,
            visited,
            stdlib_root=sr,
        )
        out_lines.append(expanded_sub)
    return "\n".join(out_lines)


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
                "Download a component datasheet PDF and return an overview: "
                "total page count, total character count, and the text of the "
                "first 2 pages. Use read_datasheet_pages to read further pages "
                "(e.g. pinout tables, electrical specs, application circuits)."
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
            "name": "read_datasheet_pages",
            "description": (
                "Read specific pages from a previously downloaded datasheet PDF. "
                "Call get_part_datasheet first to download and see page count. "
                "Use this to iteratively read pinout tables, electrical specs, "
                "application circuits, and other sections on later pages."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Same URL passed to get_part_datasheet",
                    },
                    "start_page": {
                        "type": "integer",
                        "description": "First page to read (0-indexed)",
                    },
                    "end_page": {
                        "type": "integer",
                        "description": "Last page to read (inclusive, 0-indexed)",
                    },
                },
                "required": ["url", "start_page", "end_page"],
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
                "Imports (import \"path\") are resolved and inlined so you get the complete expanded content. "
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


_pdf_cache: dict[str, list[str]] = {}

_DATASHEET_PAGE_CAP = 6000


def _ensure_pdf_cached(url: str) -> list[str] | str:
    """Download PDF if not cached. Returns list of page texts or an error string."""
    if url in _pdf_cache:
        return _pdf_cache[url]

    if not url or not url.startswith("http"):
        return "Invalid datasheet URL"

    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "circuitd/1.0"})
        resp.raise_for_status()
    except Exception as exc:
        return f"Failed to download datasheet: {exc}"

    try:
        import pymupdf
    except ImportError:
        return "pymupdf not installed -- run: pip install pymupdf"

    try:
        pdf_bytes = resp.content
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        pages: list[str] = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
    except Exception as exc:
        return f"Failed to parse PDF: {exc}"

    _pdf_cache[url] = pages
    return pages


@_register("get_part_datasheet")
def get_part_datasheet(url: str) -> str:
    pages = _ensure_pdf_cached(url)
    if isinstance(pages, str):
        return json.dumps({"error": pages})

    total_chars = sum(len(p) for p in pages)
    preview: dict[str, str] = {}
    preview_chars = 0
    for i, text in enumerate(pages[:3]):
        preview[str(i)] = text[:_DATASHEET_PAGE_CAP]
        preview_chars += len(preview[str(i)])
        if preview_chars >= _DATASHEET_PAGE_CAP:
            break

    return json.dumps({
        "url": url,
        "total_pages": len(pages),
        "total_chars": total_chars,
        "hint": (
            "First pages shown below. Use read_datasheet_pages(url, start_page, end_page) "
            "to read pinout tables, electrical specs, and application circuits on later pages."
        ),
        "pages": preview,
    })


@_register("read_datasheet_pages")
def read_datasheet_pages(url: str, start_page: int, end_page: int) -> str:
    pages = _ensure_pdf_cached(url)
    if isinstance(pages, str):
        return json.dumps({"error": pages})

    start_page = max(0, int(start_page))
    end_page = min(len(pages) - 1, int(end_page))
    if start_page > end_page:
        return json.dumps({"error": f"start_page ({start_page}) > end_page ({end_page})"})

    result: dict[str, str] = {}
    total = 0
    for i in range(start_page, end_page + 1):
        text = pages[i]
        if total + len(text) > _DATASHEET_PAGE_CAP:
            text = text[:max(0, _DATASHEET_PAGE_CAP - total)]
        result[str(i)] = text
        total += len(text)
        if total >= _DATASHEET_PAGE_CAP:
            break

    return json.dumps({
        "url": url,
        "pages_returned": f"{start_page}-{start_page + len(result) - 1}",
        "total_pages": len(pages),
        "pages": result,
    })


@_register("list_stdlib")
def list_stdlib() -> str:
    """List all .decl files in the stdlib, including in subfolders (e.g. components/agent)."""
    stdlib = Path(config.STDLIB_PATH)
    result: dict[str, list[str]] = {}
    if not stdlib.is_dir():
        return json.dumps({"error": f"stdlib not found at {stdlib}"})

    for path in sorted(stdlib.rglob("*.decl")):
        rel = path.relative_to(stdlib)
        parent_key = str(rel.parent).replace("\\", "/")
        if parent_key == ".":
            parent_key = ""
        if parent_key not in result:
            result[parent_key] = []
        result[parent_key].append(rel.name)
    for key in result:
        result[key] = sorted(result[key])
    return json.dumps(dict(sorted(result.items())), indent=2)


@_register("read_stdlib_file")
def read_stdlib_file(path: str) -> str:
    normalized = path.strip().replace("\\", "/").lstrip("/")
    target = config.STDLIB_PATH / normalized
    if not target.is_file():
        return json.dumps({"error": f"File not found: {path}"})
    try:
        content = target.read_text()
        root = config.STDLIB_PATH.resolve()
        pdir = target.parent.resolve()
        content = _expand_decl_imports(content, pdir, stdlib_root=root)
        return content
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
    _, structured = validate_decl_structured(fixed, base_path=target.parent)
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

    # Replace hyphens in identifiers with underscores (e.g., ATmega328P-AU -> ATmega328P_AU).
    # Only matches a single hyphen between word characters; `--` in connect statements has
    # spaces around it and is never touched.
    content = re.sub(r'(?<=\w)-(?=\w)', '_', content)

    # Replace # in pin names with _N (e.g., HOLD# -> HOLD_N, RESET# -> RESET_N)
    content = re.sub(r'(\b\w+)#', r'\1_N', content)

    # Fix bare "R" resistance suffix -> "ohm" (e.g., 220R -> 220ohm, 4.7kR -> 4.7kohm)
    content = re.sub(r'(\d)R\b', r'\1ohm', content)
    content = re.sub(r'(\d[kKmM])R\b', r'\1ohm', content)

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

    # Names must start with an alphabetic letter (not a digit). Prefix with C_/I_/N_ as needed.
    content = re.sub(
        r'\b(component|variant|protocol|schematic)\s+(\d[\w]*)',
        r'\1 C_\2',
        content,
    )
    content = re.sub(r'\bof\s+(\d[\w]*)', r'of C_\1', content)
    content = re.sub(
        r'\binstance\s+(\w+)\s*:\s*(\d[\w]*)',
        r'instance \1 : C_\2',
        content,
    )
    content = re.sub(
        r'\binstance\s+(\d[\w]*)\s*:',
        r'instance I_\1 :',
        content,
    )
    content = re.sub(r'\bnet\s+(\d[\w]*)', r'net N_\1', content)

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


def _run_decl_check(content: str, base_path: Path | None = None) -> tuple[str, str, bool]:
    """Run decl checker on content. Returns (stdout, stderr, success).

    If content contains import statements, they are resolved relative to base_path
    (default: config.STDLIB_PATH) so the checker sees a single expanded file.
    """
    content = _fix_common_issues(content)
    expand_base = base_path if base_path is not None else getattr(config, "STDLIB_PATH", None)
    if "import " in content and expand_base is not None:
        eb = Path(expand_base).resolve()
        if not eb.is_dir():
            eb = eb.parent
        sr = getattr(config, "STDLIB_PATH", None)
        stdlib = Path(sr).resolve() if sr and Path(sr).is_dir() else None
        try:
            content = _expand_decl_imports(content, eb, stdlib_root=stdlib)
        except DeclImportError as exc:
            return "", str(exc), False
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


# ---------------------------------------------------------------------------
# Requires extraction (for completeness checks)
# ---------------------------------------------------------------------------

_REQUIRES_ENTRY_RE = re.compile(
    r"(\w+)\s*(?:\{([^}]*)\})?\s*(?:\*\s*(\d+))?"
)


def extract_requires_from_decl(
    content: str,
) -> list[tuple[str, dict[str, str], int]]:
    """Parse requires blocks from DECL and return flat list of needed support components.

    For every component *instantiated* in the schematic, collects its requires entries.
    Handles variants by resolving to their base component's requires.
    Returns list of (component_type, {attr: value_str}, count) tuples.
    """
    comp_requires: dict[str, list[tuple[str, dict[str, str], int]]] = {}
    comp_re = re.compile(r"component\s+(\w+)\s*\{", re.MULTILINE)
    pos = 0
    while True:
        m = comp_re.search(content, pos)
        if not m:
            break
        comp_name = m.group(1)
        start = m.end()
        depth = 1
        i = start
        while i < len(content) and depth > 0:
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
            i += 1
        block = content[start : i - 1]

        req_m = re.search(r"requires\s*\{", block)
        entries: list[tuple[str, dict[str, str], int]] = []
        if req_m:
            rstart = req_m.end()
            rdepth = 1
            ri = rstart
            while ri < len(block) and rdepth > 0:
                if block[ri] == "{":
                    rdepth += 1
                elif block[ri] == "}":
                    rdepth -= 1
                ri += 1
            req_body = block[rstart : ri - 1]
            for line in req_body.strip().splitlines():
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                em = _REQUIRES_ENTRY_RE.match(line)
                if em:
                    comp_type = em.group(1)
                    attrs: dict[str, str] = {}
                    if em.group(2):
                        for pair in re.findall(r"(\w+)\s*=\s*(\S+)", em.group(2)):
                            attrs[pair[0]] = pair[1]
                    count = int(em.group(3)) if em.group(3) else 1
                    entries.append((comp_type, attrs, count))
        comp_requires[comp_name] = entries
        pos = i

    variant_to_base: dict[str, str] = {}
    for vm in re.finditer(r"variant\s+(\w+)\s+of\s+(\w+)", content):
        variant_to_base[vm.group(1)] = vm.group(2)

    instance_re = re.compile(r"instance\s+\w+\s*:\s*(\w+)", re.MULTILINE)
    result: list[tuple[str, dict[str, str], int]] = []
    seen_types: set[str] = set()
    for im in instance_re.finditer(content):
        ctype = im.group(1)
        if ctype in seen_types:
            continue
        seen_types.add(ctype)
        if ctype in comp_requires:
            result.extend(comp_requires[ctype])
        elif ctype in variant_to_base:
            base = variant_to_base[ctype]
            if base in comp_requires:
                result.extend(comp_requires[base])
    return result


# ---------------------------------------------------------------------------
# Unconnected pin promotion
# ---------------------------------------------------------------------------

_W001_RE = re.compile(
    r"\[W001\]\s*Pin\s+'(\w+)'\s+on\s+instance\s+'(\w+)'\s+\((\w+)\)\s+is\s+not\s+connected"
)


def _count_pins_per_component(decl: str) -> dict[str, int]:
    """Count declared pins for each component in the DECL source."""
    counts: dict[str, int] = {}
    comp_re = re.compile(r"component\s+(\w+)\s*\{", re.MULTILINE)
    pos = 0
    while True:
        m = comp_re.search(decl, pos)
        if not m:
            break
        comp_name = m.group(1)
        start = m.end()
        depth = 1
        i = start
        while i < len(decl) and depth > 0:
            if decl[i] == "{":
                depth += 1
            elif decl[i] == "}":
                depth -= 1
            i += 1
        block = decl[start : i - 1]
        pins_m = re.search(r"pins\s*\{([^}]*)\}", block, re.DOTALL)
        if pins_m:
            pin_lines = re.findall(r"(?:^\s*\d+\s*:|^\s*\w+\s*:)\s*\w+", pins_m.group(1), re.MULTILINE)
            counts[comp_name] = len(pin_lines)
        pos = i
    return counts


def _promote_w001_for_small_components(
    checker_output: str, decl: str,
) -> list[dict]:
    """Parse W001 warnings; promote to E012 errors for components with <=8 pins."""
    pin_counts = _count_pins_per_component(decl)
    errors: list[dict] = []
    for m in _W001_RE.finditer(checker_output):
        pin_name, inst_name, comp_type = m.group(1), m.group(2), m.group(3)
        n_pins = pin_counts.get(comp_type, 999)
        if n_pins <= 8:
            errors.append({
                "code": "E012",
                "line": 0,
                "message": (
                    f"Unconnected pin '{pin_name}' on instance '{inst_name}' ({comp_type}). "
                    f"All pins on support components (resistors, capacitors, connectors, LEDs) "
                    f"must be connected. Connect {inst_name}.{pin_name} to the appropriate net or pin."
                ),
                "entities": [f"{inst_name}.{pin_name}"],
            })
    return errors


def validate_decl_structured(content: str, base_path: Path | None = None) -> tuple[str, list[dict]]:
    """Validate .decl content; return (human-readable output, structured errors).

    Structured errors are list of {"code", "line", "message", "entities"} for use
    in the repair phase. If the checker output is not parseable, structured_errors
    is empty but human_output is still returned.

    If content contains import statements, base_path is used to resolve them
    (default: config.STDLIB_PATH).
    """
    stdout, stderr, is_ok = _run_decl_check(content, base_path=base_path)
    output = ""
    if stdout:
        output += stdout
    if stderr:
        output += ("\n" + stderr) if output else stderr
    if not output:
        output = "(no output)"
    fixed = _fix_common_issues(content)
    output = _annotate_errors(output, fixed)
    structured = [] if is_ok else _parse_validation_errors(stdout + "\n" + stderr)
    protocol_errors = validate_decl_protocol_pins(fixed)
    if protocol_errors:
        structured = structured + protocol_errors
        output += "\n\nProtocol pin validation failed:\n" + "\n".join(
            e.get("message", "") for e in protocol_errors
        )
    unconnected_errors = _promote_w001_for_small_components(stdout + "\n" + stderr, fixed)
    if unconnected_errors:
        structured = structured + unconnected_errors
        output += "\n\nUnconnected pin errors (support components):\n" + "\n".join(
            e.get("message", "") for e in unconnected_errors
        )
    if is_ok and not protocol_errors and not unconnected_errors:
        output += (
            "\n\nVALIDATION PASSED. W001 (unconnected pin) warnings on MCU GPIO "
            "pins can be ignored. Do NOT re-validate. "
            "Output the final .decl file in a ```decl code block now."
        )
    return output, structured
