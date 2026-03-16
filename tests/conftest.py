"""Pytest fixtures shared across phase tests."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Minimal valid DECL component for use in tests
MINIMAL_DECL_COMPONENT = """
component Resistor {
  pins {
    1: Passive as A
    2: Passive as B
  }
  attributes {
    resistance: Resistance = 10kohm
  }
}

"""

# Minimal requirements JSON from phase 1
MINIMAL_REQUIREMENTS = {
    "functions": [],
    "constraints": [],
    "explicit_parts": [{"name": "MCU", "reason": "main controller"}],
    "implied_blocks": [],
    "power_requirements": [],
    "interfaces": [],
    "support_components": [],
    "assumptions": [],
    "open_questions": [],
}


@pytest.fixture
def mock_stdlib_path(tmp_path):
    """A temporary directory as STDLIB_PATH with components/agent."""
    agent = tmp_path / "components" / "agent"
    agent.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def sample_datasheet_json():
    """JSON string that get_part_datasheet returns on success."""
    return json.dumps({
        "url": "https://example.com/datasheet.pdf",
        "text": "Sample IC Datasheet\n\nPinout:\n1 VCC 2 GND 3 CLK 4 DATA\n\nAbsolute max 5.5V.",
    })
