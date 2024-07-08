from typing import Any

import pytest

from snapi.query_agents.tns import TNSQueryAgent


@pytest.fixture
def tns_agent() -> TNSQueryAgent:
    """TNS query agent fixture."""
    return TNSQueryAgent()


@pytest.fixture
def test_event() -> dict[str, Any]:
    """Test event fixture."""
    return {
        "id": "2023ixf",
        "ra": 210.910674,
        "dec": 54.31165,
        "redshift": 0.0008,
    }
