from typing import Any

import pytest

from snapi.query_agents.alerce import ALeRCEQueryAgent
from snapi.query_agents.antares import ANTARESQueryAgent
from snapi.query_agents.ghost import GHOSTQueryAgent
from snapi.query_agents.tns import TNSQueryAgent


@pytest.fixture
def tns_agent() -> TNSQueryAgent:
    """TNS query agent fixture."""
    return TNSQueryAgent()


@pytest.fixture
def alerce_agent() -> ALeRCEQueryAgent:
    """ALeRCE query agent fixture."""
    return ALeRCEQueryAgent()


@pytest.fixture
def antares_agent() -> ANTARESQueryAgent:
    """ANTARES query agent fixture."""
    return ANTARESQueryAgent()


@pytest.fixture
def ghost_agent() -> GHOSTQueryAgent:
    """GHOST query agent fixture."""
    return GHOSTQueryAgent()


@pytest.fixture
def test_event() -> dict[str, Any]:
    """Test event fixture."""
    return {
        "id": "2023ixf",
        "ra": 210.910674,
        "dec": 54.31165,
        "redshift": 0.0008,
        "ztf_id": "ZTF23aaklqou",
        "hostname": "NGC 5461",
    }


@pytest.fixture
def test_event_oqm() -> dict[str, Any]:
    """Test event fixture."""
    return {
        "id": "2022oqm",
        "ra": 227.28421256,
        "dec": 52.5347606571,
        "redshift": 0.012,
        "ztf_id": "ZTF22aasxgjp",
        "hostname": "NGC 5875",
    }


class Helpers:  # pylint: disable=too-few-public-methods
    """Helper functions for test cases."""

    @staticmethod
    def assert_query_result(
        query_result: Any, iid: str, ra: float, dec: float, z: float, phot_spec: bool = True
    ) -> None:
        """Assert query result."""
        assert query_result.objname == iid
        assert query_result.coordinates is not None
        assert query_result.coordinates.ra.deg == pytest.approx(ra, rel=1e-3)
        assert query_result.coordinates.dec.deg == pytest.approx(dec, rel=1e-3)
        assert query_result.redshift == pytest.approx(z, rel=1e-3)
        assert query_result.internal_names is not None

        if phot_spec:
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
