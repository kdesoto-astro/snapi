"""Test cases for TNS query functions."""
from typing import Any

import pytest

from snapi.query_agents.tns import TNSQueryAgent


def test_tns_query_by_name(tns_agent: TNSQueryAgent, test_event: dict[str, Any]) -> None:
    """Test TNS query by name."""
    result_list = tns_agent.query_by_name(test_event["id"])[0]
    assert len(result_list) == 1
    query_result = result_list[0]
    assert query_result.objname == test_event["id"]
    assert query_result.coordinates is not None
    assert query_result.coordinates.ra.deg == pytest.approx(test_event["ra"])
    assert query_result.coordinates.dec.deg == pytest.approx(test_event["dec"])
    assert query_result.redshift == pytest.approx(test_event["redshift"])
    assert query_result.internal_names is not None
    assert query_result.light_curves is not None

    assert len(query_result.internal_names) > 0
    assert len(query_result.light_curves) > 0
    for lc in query_result.light_curves:
        assert lc.filter is not None
        assert lc.filter.band is not None
        assert len(lc.times) > 0
        assert len(lc.mags) > 0
