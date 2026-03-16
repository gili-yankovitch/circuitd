"""Tests for Phase 3: Design plan (connection plan)."""

import json
from unittest.mock import patch, MagicMock

import pytest

from circuitd.agent import _run_phase3_design_plan, _extract_json


MINIMAL_PLAN = {
    "instances": [
        {"name": "U1", "component": "CH32V003F4P6"},
        {"name": "U2", "component": "W25Q128JV"},
    ],
    "nets": ["VCC_3V3", "GND", "SPI_CLK", "SPI_MOSI", "SPI_MISO", "SPI_CS"],
    "connections": [
        {"from": "U1.VDD", "to": "net VCC_3V3", "reason": "power"},
        {"from": "U1.PC5", "to": "U2.CLK", "reason": "SPI CLK"},
    ],
    "power_topology": [],
    "protocol_bindings": [],
    "checks": [],
}


class TestPhase3DesignPlan:
    """Test design plan phase."""

    @patch("circuitd.agent.create_chat")
    def test_returns_plan_when_valid_json(self, create_chat):
        mock_chat = MagicMock()
        mock_chat.send.return_value = "```json\n" + json.dumps(MINIMAL_PLAN) + "\n```"
        create_chat.return_value = mock_chat

        result = _run_phase3_design_plan(
            {"explicit_parts": []},
            {"selected_components": []},
            backend="ollama",
            model=None,
            ollama_url=None,
        )
        assert "instances" in result
        assert len(result["instances"]) == 2
        assert result["instances"][0]["component"] == "CH32V003F4P6"
        assert "nets" in result
        assert "connections" in result

    @patch("circuitd.agent.create_chat")
    def test_returns_empty_plan_when_no_valid_json(self, create_chat):
        mock_chat = MagicMock()
        mock_chat.send.return_value = "No structured output"
        create_chat.return_value = mock_chat

        result = _run_phase3_design_plan(
            {}, {},
            backend="ollama", model=None, ollama_url=None,
        )
        assert result.get("instances") == []
        assert result.get("nets") == []
