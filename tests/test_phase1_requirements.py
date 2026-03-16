"""Tests for Phase 1: Requirements extraction."""

import json
from unittest.mock import patch, MagicMock

import pytest

from circuitd.agent import _run_phase1_requirements, _extract_json


class TestExtractJson:
    """Test JSON extraction from LLM response."""

    def test_extract_from_fenced_block(self):
        text = 'Some text\n```json\n{"explicit_parts": [{"name": "X"}]}\n```'
        out = _extract_json(text)
        assert out is not None
        assert out.get("explicit_parts")[0]["name"] == "X"

    def test_extract_from_bare_braces(self):
        text = 'Before {"a": 1} after'
        assert _extract_json(text) == {"a": 1}

    def test_returns_none_for_invalid(self):
        assert _extract_json("no json here") is None
        assert _extract_json("") is None


class TestPhase1Requirements:
    """Test requirements extraction phase."""

    @pytest.fixture
    def mock_response_with_json(self):
        return '```json\n' + json.dumps({
            "functions": ["blink LED"],
            "constraints": [],
            "explicit_parts": [{"name": "ATTiny85", "reason": "MCU"}],
            "implied_blocks": [{"name": "LDO", "reason": "3.3V"}],
            "power_requirements": [],
            "interfaces": [],
            "support_components": [],
            "assumptions": [],
            "open_questions": [],
        }) + '\n```'

    @patch("circuitd.agent.create_chat")
    def test_returns_requirements_when_valid_json(self, create_chat, mock_response_with_json):
        mock_chat = MagicMock()
        mock_chat.send.return_value = mock_response_with_json
        create_chat.return_value = mock_chat

        result = _run_phase1_requirements(
            "USB LED board",
            backend="ollama",
            model=None,
            ollama_url=None,
        )
        assert "explicit_parts" in result
        assert len(result["explicit_parts"]) == 1
        assert result["explicit_parts"][0]["name"] == "ATTiny85"
        assert len(result["implied_blocks"]) == 1

    @patch("circuitd.agent.create_chat")
    def test_returns_minimal_dict_when_no_valid_json(self, create_chat):
        mock_chat = MagicMock()
        mock_chat.send.return_value = "No json here"
        create_chat.return_value = mock_chat

        result = _run_phase1_requirements(
            "foo",
            backend="ollama",
            model=None,
            ollama_url=None,
        )
        assert result.get("explicit_parts") == []
        assert result.get("implied_blocks") == []
