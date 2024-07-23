"""Test cases for ghost query functions."""
from typing import Any

import astropy.units as u
from astropy.coordinates import SkyCoord

from snapi.query_agents.ghost import GHOSTQueryAgent
from snapi.transient import Transient


def test_ghost_query_by_name(ghost_agent: GHOSTQueryAgent) -> None:
    """Test GHOST query by name."""
    try:
        result_list = ghost_agent.query_by_name("NGC 5493")[0]
    except RuntimeError:  # try again
        result_list = ghost_agent.query_by_name("NGC 5493")[0]
    assert len(result_list) == 1
    query_result = result_list[0]
    assert len(query_result.to_dict()["host_internal_names"]) > 0
    assert query_result.to_dict()["hostname"] == "NGC 5493"


def test_ghost_query_by_coords(ghost_agent: GHOSTQueryAgent, test_event_oqm: dict[str, Any]) -> None:
    """Test ghost query by name."""
    test_coord = SkyCoord(test_event_oqm["ra"], test_event_oqm["dec"], unit="deg")
    try:
        result_list = ghost_agent.query_by_coords(test_coord)[0]
    except RuntimeError:  # try again
        result_list = ghost_agent.query_by_coords(test_coord)[0]
    assert len(result_list) == 1
    query_result = result_list[0]
    assert len(query_result.to_dict()["host_internal_names"]) > 0
    assert query_result.to_dict()["hostname"] == test_event_oqm["hostname"]


def test_ghost_query_by_transient(ghost_agent: GHOSTQueryAgent, test_event_oqm: dict[str, Any]) -> None:
    """Test ghost query by name."""
    test_transient = Transient(
        iid=test_event_oqm["ztf_id"],
        ra=test_event_oqm["ra"] * u.deg,  # pylint: disable=no-member
        dec=test_event_oqm["dec"] * u.deg,  # pylint: disable=no-member
        redshift=test_event_oqm["redshift"],
    )
    try:
        result_list = ghost_agent.query_transient(test_transient)[0]
    except RuntimeError:  # try again
        result_list = ghost_agent.query_transient(test_transient)[0]
    assert len(result_list) == 1
    query_result = result_list[0]
    assert len(query_result.to_dict()["host_internal_names"]) > 0
    assert query_result.to_dict()["hostname"] == test_event_oqm["hostname"]
