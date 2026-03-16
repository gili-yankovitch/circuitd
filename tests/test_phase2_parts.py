"""Tests for Phase 2: Parts discovery and datasheet-to-DECL auto-save."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from circuitd import agent
from circuitd.agent import (
    PARTS_PHASE_DISPATCH,
    _parts_dispatch_with_auto_save,
    _convert_datasheet_to_decl_and_save,
)


# Real W25Q128JV datasheet URL (Winbond)
W25Q128JV_DATASHEET_URL = "https://www.winbond.com/resource-files/w25q128jv%20revf%2003272018%20plus.pdf"


def _load_w25q128jv_excerpt() -> str:
    """Load W25Q128JV datasheet excerpt from fixtures (real content from Winbond PDF)."""
    path = Path(__file__).resolve().parent / "fixtures" / "w25q128jv_datasheet_excerpt.txt"
    return path.read_text()


# Valid single-component DECL (no schematic) for datasheet conversion
VALID_COMPONENT_DECL = """
component W25Q128JV {
  pins {
    CLK: Input
    DI: Input
    DO: Output
    CS: Input
    VCC: PowerInput
    GND: PowerInput
  }
}
"""


class TestPartsDispatchWithAutoSave:
    """Test that the wrapper invokes conversion when get_part_datasheet succeeds.

    Uses real W25Q128JV datasheet excerpt (no mock of _convert_datasheet_to_decl_and_save).
    """

    def test_wrapper_invokes_convert_on_success_response(self, mock_stdlib_path):
        """When get_part_datasheet returns real W25Q128JV datasheet text, convert runs and save is called."""
        excerpt = _load_w25q128jv_excerpt()
        success_result = json.dumps({"url": W25Q128JV_DATASHEET_URL, "text": excerpt})
        fake_dispatch = dict(PARTS_PHASE_DISPATCH)
        fake_dispatch["get_part_datasheet"] = lambda url: success_result
        with patch.object(agent, "PARTS_PHASE_DISPATCH", fake_dispatch):
            with patch("circuitd.datasheet_to_decl.create_chat") as mock_create_chat:
                with patch("circuitd.datasheet_to_decl.validate_decl_structured", return_value=("OK: file", [])):
                    with patch("circuitd.agent.save_to_stdlib", MagicMock(return_value='{"ok": true}')) as mock_save:
                        with patch("circuitd.tools.config.STDLIB_PATH", mock_stdlib_path), \
                             patch("circuitd.tools.config.STDLIB_AGENT_SUBDIR", "components/agent"):
                            mock_chat = MagicMock()
                            mock_chat.send.return_value = "```decl\n" + VALID_COMPONENT_DECL + "\n```"
                            mock_create_chat.return_value = mock_chat

                            dispatch = _parts_dispatch_with_auto_save(backend="ollama", model=None, ollama_url=None)
                            wrapped = dispatch["get_part_datasheet"]
                            out = wrapped(url=W25Q128JV_DATASHEET_URL)

        data = json.loads(out)
        assert data["text"] == excerpt
        assert data["url"] == W25Q128JV_DATASHEET_URL
        mock_save.assert_called_once()
        call_path = mock_save.call_args[0][0]
        assert call_path == "components/agent/W25Q128JV.decl"

    def test_wrapper_does_not_invoke_convert_when_error_in_response(self):
        """When get_part_datasheet returns JSON with "error", convert path does not run (save not called)."""
        error_result = json.dumps({"error": "Failed to download", "url": "https://x.com/d.pdf"})
        fake_dispatch = dict(PARTS_PHASE_DISPATCH)
        fake_dispatch["get_part_datasheet"] = lambda url: error_result
        with patch.object(agent, "PARTS_PHASE_DISPATCH", fake_dispatch):
            with patch("circuitd.agent.save_to_stdlib", MagicMock()) as mock_save:
                dispatch = _parts_dispatch_with_auto_save(backend="ollama", model=None, ollama_url=None)
                dispatch["get_part_datasheet"](url="https://x.com/d.pdf")
        mock_save.assert_not_called()

    def test_wrapper_does_not_invoke_convert_when_no_text(self):
        """When response has no "text" key, convert is not called (save not called)."""
        no_text_result = json.dumps({"url": "https://x.com/d.pdf"})
        fake_dispatch = dict(PARTS_PHASE_DISPATCH)
        fake_dispatch["get_part_datasheet"] = lambda url: no_text_result
        with patch.object(agent, "PARTS_PHASE_DISPATCH", fake_dispatch):
            with patch("circuitd.agent.save_to_stdlib", MagicMock()) as mock_save:
                dispatch = _parts_dispatch_with_auto_save(backend="ollama", model=None, ollama_url=None)
                dispatch["get_part_datasheet"](url="https://x.com/d.pdf")
        mock_save.assert_not_called()


class TestConvertDatasheetToDeclAndSave:
    """Test _convert_datasheet_to_decl_and_save with real W25Q128JV datasheet excerpt (LLM and save mocked)."""

    @pytest.mark.parametrize("text,should_skip", [
        ("", True),
        ("short", True),
        ("x" * 100, False),
    ])
    def test_skips_when_text_too_short(self, text, should_skip):
        with patch("circuitd.datasheet_to_decl.create_chat"), patch("circuitd.agent.save_to_stdlib", MagicMock()) as mock_save:
            _convert_datasheet_to_decl_and_save(
                text, "https://x.com/d.pdf",
                backend="ollama", model=None, ollama_url=None,
            )
            if should_skip:
                mock_save.assert_not_called()
            else:
                # With 100 chars we don't skip; create_chat.send would be called
                pass

    def test_calls_save_when_llm_returns_valid_decl_and_validation_passes(self, mock_stdlib_path):
        """With real W25Q128JV datasheet text, LLM returns valid DECL and validation passes -> save_to_stdlib called."""
        datasheet_text = _load_w25q128jv_excerpt()
        with patch("circuitd.datasheet_to_decl.create_chat") as mock_create_chat:
            with patch("circuitd.datasheet_to_decl.validate_decl_structured", return_value=("OK: file", [])):
                with patch("circuitd.agent.save_to_stdlib", MagicMock(return_value='{"ok": true}')) as mock_save:
                    with patch("circuitd.tools.config.STDLIB_PATH", mock_stdlib_path), \
                         patch("circuitd.tools.config.STDLIB_AGENT_SUBDIR", "components/agent"):
                        mock_chat = MagicMock()
                        mock_chat.send.return_value = "Here is the component:\n```decl\n" + VALID_COMPONENT_DECL + "\n```"
                        mock_create_chat.return_value = mock_chat

                        _convert_datasheet_to_decl_and_save(
                            datasheet_text,
                            W25Q128JV_DATASHEET_URL,
                            backend="ollama",
                            model=None,
                            ollama_url=None,
                        )

                        mock_save.assert_called_once()
                        call_path = mock_save.call_args[0][0]
                        assert call_path.startswith("components/agent/")
                        assert call_path.endswith(".decl")
                        assert call_path == "components/agent/W25Q128JV.decl"
