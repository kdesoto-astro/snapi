"""Test cases for ANTARES query functions."""
from typing import Any

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord

from snapi.query_agents.antares import ANTARESQueryAgent
from snapi.transient import Transient


@pytest.mark.skip_precommit
def test_antares_query_by_name(
    antares_agent: ANTARESQueryAgent, test_event: dict[str, Any], helpers: Any
) -> None:
    """Test ANTARES query by name."""
    result_list = antares_agent.query_by_name(test_event["ztf_id"])[0]
    assert len(result_list) == 1
    query_result = result_list[0]
    query_result.redshift = test_event["redshift"]
    helpers.assert_query_result(
        query_result, test_event["ztf_id"], test_event["ra"], test_event["dec"], test_event["redshift"]
    )


@pytest.mark.skip_precommit
def test_antares_query_by_coords(
    antares_agent: ANTARESQueryAgent, test_event: dict[str, Any], helpers: Any
) -> None:
    """Test ANTARES query by name."""
    test_coord = SkyCoord(test_event["ra"], test_event["dec"], unit="deg")
    result_list = antares_agent.query_by_coords(test_coord)[0]
    assert len(result_list) == 1
    query_result = result_list[0]
    query_result.redshift = test_event["redshift"]
    helpers.assert_query_result(
        query_result, test_event["ztf_id"], test_event["ra"], test_event["dec"], test_event["redshift"]
    )


@pytest.mark.skip_precommit
def test_antares_query_by_transient(
    antares_agent: ANTARESQueryAgent, test_event: dict[str, Any], helpers: Any
) -> None:
    """Test ANTARES query by name."""
    test_transient = Transient(
        iid=test_event["ztf_id"],
        ra=test_event["ra"] * u.deg,  # pylint: disable=no-member
        dec=test_event["dec"] * u.deg,  # pylint: disable=no-member
        redshift=test_event["redshift"],
    )
    result_list = antares_agent.query_transient(test_transient)[0]
    assert len(result_list) == 1
    query_result = result_list[0]
    query_result.redshift = test_event["redshift"]
    helpers.assert_query_result(
        query_result, test_event["ztf_id"], test_event["ra"], test_event["dec"], test_event["redshift"]
    )
