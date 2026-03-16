# circuitd

**circuitd** is an AI agent that turns a natural-language circuit description into a valid **DECL** schematic file. You describe what you want (e.g. “USB-powered ATTiny85 board with status LED”), and the agent runs a multi-phase pipeline: requirements extraction, part selection (with optional stdlib/datasheet lookups), design planning, DECL generation, validation, and repair—including a completeness check so required blocks (LDO, decoupling caps, USB, etc.) are present.

## What is DECL?

DECL is a declarative language for describing circuit schematics: components (with pins and attributes), nets, and connections. A typical `.decl` file defines one or more `component` types, then a `schematic` that instantiates them and wires pins to nets or to each other. The `decl` tool (separate from this repo) checks DECL files for syntax and semantic errors.

## Requirements

- **Python 3.10+**
- **LLM backend**: [Ollama](https://ollama.ai) (default) or OpenAI API
- **Optional**: `decl` checker on `PATH` for validation and repair
- **Optional**: Parts search API (e.g. JLCPCB-style) and stdlib path for part discovery

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Quick start

Run the agent with a short circuit description. Output is written to `output.decl` by default:

```bash
python -m circuitd "USB-powered ATTiny85 with a status LED"
```

Specify output path and backend:

```bash
python -m circuitd "3.3V MCU board with SPI flash and status LED" -o my_board.decl --backend ollama
```

## Example: ATTiny85 + LED

**Prompt:**

```text
USB-powered ATTiny85 with a status LED on one GPIO
```

**What the agent does:**

1. **Phase 1 — Requirements**  
   Extracts explicit and implied blocks: ATTiny85, USB power input, 3.3V LDO, decoupling capacitors, current-limiting resistor for the LED, sink-side CC resistors for USB, optional pull-up for digital I/O.

2. **Phase 2 — Parts**  
   Uses tools to search/stdlib/datasheets (if configured), then outputs a normalized component list (MCU, LDO, USB connector, resistors, capacitors, LED).

3. **Phase 3 — Design plan**  
   Produces a connection plan: instances, nets, and how they connect (power topology, which pins go to which nets).

4. **Phase 4 — DECL generation**  
   Emits a single `.decl` file with all `component` definitions and one `schematic` that instantiates and connects them.

5. **Phase 5 — Validation and repair**  
   Runs the `decl` checker; if there are errors, the agent fixes the DECL and re-validates until it passes (or max iterations).

6. **Completeness**  
   The agent checks that every required item from the requirements (LDO, decoupling caps, USB, LED resistor, etc.) appears in the DECL. It uses a heuristic first, then an LLM pass to avoid false “missing” reports when the same function is implemented under different names (e.g. `LDO_3V3` vs “3.3V LDO”). Only items that are actually missing trigger an “add missing components” repair.

**Example output** (conceptually; actual names may vary):

- Components: `ATTINY85V_10SU`, `AMS1117_3_3V` (or generic LDO), USB connector (e.g. `USB_U302C_CJS`), `Resistor`, `Capacitor`, `LED`.
- Schematic: USB → 5V net → LDO → 3.3V net → MCU; decoupling caps on 5V and 3.3V; LED with series resistor on a GPIO; sink-side CC resistors on the USB connector; optional pull-up on the same or another pin.

A full example of an agent-generated ATTiny85 + LED design is in **`attiny85-led.decl`** in this repo (layout and part names may differ from run to run).

## Command-line options

| Option | Description |
|--------|-------------|
| `prompt` | Circuit description (positional). |
| `-o`, `--output` | Output `.decl` path (default: `output.decl`). |
| `--backend` | `ollama` or `openai`. |
| `--model` | Model name (default: backend-specific). |
| `--ollama-url` | Ollama server URL. |
| `--parts-url` | Parts search API base URL. |
| `-v`, `--verbose` | Verbose logging. |
| `--prompts-log` | Append prompts/responses to this file. |
| `--no-prompts-log` | Disable prompt logging. |

## Configuration

Edit **`circuitd/config.py`** (or set env/CLI) for:

- **Backend**: `BACKEND = "ollama"` or `"openai"`.
- **Ollama**: `OLLAMA_URL`, `OLLAMA_MODEL`, timeouts, context size.
- **OpenAI**: API key (e.g. from `openai.key`), model, timeout.
- **Parts API**: `PARTS_API_URL` for part search.
- **DECL**: `STDLIB_PATH`, `DECL_CHECK_CMD` (must be on `PATH` for validation/repair).
- **Agent**: `MAX_AGENT_ITERATIONS`, `MAX_VALIDATION_RETRIES`, etc.
- **Prompt log**: `PROMPTS_LOG_PATH`; set to `None` to disable.

## Pipeline summary

| Phase | Purpose |
|-------|--------|
| 1. Requirements | Natural language → structured brief (explicit/implied parts, power, interfaces, support components). |
| 2. Parts | Resolve parts via stdlib/search/datasheets; output selection JSON. |
| 3. Design plan | Instances, nets, connections, power topology. |
| 4. Generate DECL | One `.decl` file from the plan. |
| 5. Validate & repair | Run `decl check`; LLM fixes errors until OK. |
| Completeness | Heuristic + LLM check that required items are present; add only actually missing ones. |

## License and dependencies

See `requirements.txt` for Python dependencies. The `decl` checker and DECL language are separate projects. This README and the agent are provided as-is for designing circuits in DECL form.
