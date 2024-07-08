"""Test cases for TNS query functions."""
from typing import Any

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord

from snapi.query_agents.tns import TNSQueryAgent
from snapi.transient import Transient


@pytest.mark.skip_precommit
def test_tns_query_by_name(tns_agent: TNSQueryAgent, test_event: dict[str, Any], helpers: Any) -> None:
    """Test TNS query by name."""
    result_list = tns_agent.query_by_name(test_event["id"])[0]
    assert len(result_list) == 1
    query_result = result_list[0]
    helpers.assert_query_result(query_result, test_event)


@pytest.mark.skip_precommit
def test_tns_query_by_coords(tns_agent: TNSQueryAgent, test_event: dict[str, Any], helpers: Any) -> None:
    """Test TNS query by name."""
    test_coord = SkyCoord(test_event["ra"], test_event["dec"], unit="deg")
    result_list = tns_agent.query_by_coords(test_coord)[0]
    assert len(result_list) == 1
    query_result = result_list[0]
    helpers.assert_query_result(query_result, test_event)


@pytest.mark.skip_precommit
def test_tns_query_by_transient(tns_agent: TNSQueryAgent, test_event: dict[str, Any], helpers: Any) -> None:
    """Test TNS query by name."""
    test_transient = Transient(
        iid=test_event["id"],
        ra=test_event["ra"] * u.deg,  # pylint: disable=no-member
        dec=test_event["dec"] * u.deg,  # pylint: disable=no-member
        redshift=test_event["redshift"],
    )
    result_list = tns_agent.query_transient(test_transient)[0]
    assert len(result_list) == 1
    query_result = result_list[0]
    helpers.assert_query_result(query_result, test_event)
