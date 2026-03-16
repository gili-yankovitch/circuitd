# Circuitd tests

Run all tests:

```bash
pytest tests/ -v
```

- **test_phase1_requirements.py** – Requirements extraction (JSON parsing, Phase 1 with mocked LLM).
- **test_phase2_parts.py** – Parts phase: wrapper invokes datasheet→DECL save when `get_part_datasheet` returns success; `_convert_datasheet_to_decl_and_save` calls `save_to_stdlib` when LLM returns valid DECL and validation passes.
- **test_phase3_design_plan.py** – Design plan (Phase 3) with mocked LLM.
- **test_phase4_generate_decl.py** – DECL generation (Phase 4) and `_extract_decl` helper.
- **test_phase5_repair.py** – Validation and repair loop (Phase 5): no repair when validation passes; repair invoked when validation fails.
