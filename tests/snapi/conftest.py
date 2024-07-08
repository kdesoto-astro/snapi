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


class Helpers:  # pylint: disable=too-few-public-methods
    """Helper functions for test cases."""

    @staticmethod
    def assert_query_result(query_result: Any, event: dict[str, Any]) -> None:
        """Assert query result."""
        assert query_result.objname == event["id"]
        assert query_result.coordinates is not None
        assert query_result.coordinates.ra.deg == pytest.approx(event["ra"])
        assert query_result.coordinates.dec.deg == pytest.approx(event["dec"])
        assert query_result.redshift == pytest.approx(event["redshift"])
        assert query_result.internal_names is not None
        assert query_result.light_curves is not None

        assert len(query_result.internal_names) > 0
        assert len(query_result.light_curves) > 0
        for lc in query_result.light_curves:
            assert lc.filter is not None
            assert lc.filter.band is not None
            assert len(lc.times) > 0
            assert len(lc.mags) > 0


@pytest.fixture
def helpers() -> Any:
    """Returns Helpers fixture."""
    return Helpers
