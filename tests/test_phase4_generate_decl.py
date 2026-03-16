"""Tests for Phase 4: DECL generation."""

import json
from unittest.mock import patch, MagicMock

import pytest

from circuitd.agent import _run_phase4_generate_decl, _extract_decl


SAMPLE_PLAN = {
    "instances": [
        {"name": "U1", "component": "MCU"},
        {"name": "R1", "component": "Resistor"},
    ],
    "nets": ["VCC", "GND"],
    "connections": [],
    "power_topology": [],
    "protocol_bindings": [],
    "checks": [],
}

SAMPLE_DECL = """
component Resistor {
  pins { 1: Passive as A  2: Passive as B }
}
schematic Main {
  instance U1: MCU
  instance R1: Resistor
  net VCC
  net GND
}
"""


class TestPhase4GenerateDecl:
    """Test DECL generation phase."""

    @patch("circuitd.agent.create_chat")
    def test_returns_decl_when_llm_emits_decl_block(self, create_chat):
        mock_chat = MagicMock()
        mock_chat.send.return_value = "```decl\n" + SAMPLE_DECL + "\n```"
        create_chat.return_value = mock_chat

        result = _run_phase4_generate_decl(
            SAMPLE_PLAN,
            backend="ollama",
            model=None,
            ollama_url=None,
        )
        assert result is not None
        assert "schematic" in result
        assert "component" in result

    @patch("circuitd.agent.create_chat")
    def test_returns_none_when_no_decl_block(self, create_chat):
        mock_chat = MagicMock()
        mock_chat.send.return_value = "I cannot generate DECL."
        create_chat.return_value = mock_chat

        result = _run_phase4_generate_decl(
            SAMPLE_PLAN,
            backend="ollama",
            model=None,
            ollama_url=None,
        )
        assert result is None


class TestExtractDecl:
    """Test _extract_decl helper."""

    def test_extracts_last_fenced_block(self):
        text = "```decl\ncomponent X { }\n```\n\n```decl\ncomponent Y { }\n```"
        out = _extract_decl(text)
        assert out is not None
        assert "component Y" in out

    def test_returns_none_when_no_component_or_schematic(self):
        text = "```decl\nnet only\n```"
        out = _extract_decl(text)
        assert out is None

    def test_returns_none_when_no_fence(self):
        assert _extract_decl("plain text") is None
