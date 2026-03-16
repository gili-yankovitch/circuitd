"""Tests for Phase 5: Validation and repair loop."""

from unittest.mock import MagicMock, patch

import pytest

from circuitd.agent import _run_phase5_repair_loop
from circuitd.tools import validate_decl_structured


VALID_DECL = """
component R { pins { 1: Passive as A  2: Passive as B } }
schematic S {
  instance R1: R
  net V
  connect R1.A -- net V
}
"""


class TestPhase5Repair:
    """Test validation and repair phase."""

    @patch("circuitd.agent.create_chat")
    def test_returns_decl_unchanged_when_validation_passes(self, create_chat):
        mock_chat = MagicMock()
        create_chat.return_value = mock_chat
        with patch("circuitd.agent.validate_decl_structured", return_value=("OK: file.decl (0 warning(s))", [])):
            result = _run_phase5_repair_loop(
                VALID_DECL,
                backend="ollama",
                model=None,
                ollama_url=None,
            )
            assert result == VALID_DECL
            # create_chat is called once to build the repair chat; send() should not be called (no repair needed)
            mock_chat.send.assert_not_called()

    @patch("circuitd.agent.create_chat")
    def test_calls_repair_when_validation_fails(self, create_chat):
        with patch("circuitd.agent.validate_decl_structured") as mock_validate:
            mock_validate.return_value = ("Error at line 1: undefined", [{"code": "E003", "line": 1, "message": "undefined", "entities": []}])
            mock_chat = MagicMock()
            mock_chat.send.return_value = "```decl\n" + VALID_DECL + "\n```"
            create_chat.return_value = mock_chat

            result = _run_phase5_repair_loop(
                VALID_DECL,
                backend="ollama",
                model=None,
                ollama_url=None,
            )
            assert result is not None
            create_chat.assert_called()
