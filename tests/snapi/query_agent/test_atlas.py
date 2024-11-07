"""Test cases for ATLAS query functions."""
from typing import Any

import astropy.units as u
from astropy.coordinates import SkyCoord

from snapi.query_agents import ATLASQueryAgent
from snapi.transient import Transient


def test_atlas_query_by_coords(
    atlas_agent: ATLASQueryAgent, test_event: dict[str, Any], helpers: Any
) -> None:
    """Test ALeRCE query by name."""
    test_coord = SkyCoord(test_event["ra"], test_event["dec"], unit="deg")
    result_list = atlas_agent.query_by_coords(test_coord)[0]
    for _ in range(10):
        if len(result_list) == 1:
            break
        result_list = atlas_agent.query_by_coords(test_coord)[0]
    assert len(result_list) == 1
    query_result = result_list[0]
    query_result.objname = test_event["ztf_id"]
    query_result.redshift = test_event["redshift"]
    query_result.internal_names = {test_event["id"], test_event["ztf_id"]}
    helpers.assert_query_result(
        query_result, test_event["ztf_id"], test_event["ra"], test_event["dec"], test_event["redshift"]
    )


def test_atlas_query_by_name(atlas_agent: ATLASQueryAgent, test_event: dict[str, Any]) -> None:
    """Test ALeRCE query by name."""
    assert not atlas_agent.query_by_name(test_event["ztf_id"])[1]


def test_atlas_query_by_transient(
    atlas_agent: ATLASQueryAgent, test_event: dict[str, Any], helpers: Any
) -> None:
    """Test ALeRCE query by name."""
    test_transient = Transient(
        iid=test_event["ztf_id"],
        ra=test_event["ra"] * u.deg,  # pylint: disable=no-member
        dec=test_event["dec"] * u.deg,  # pylint: disable=no-member
        redshift=test_event["redshift"],
    )
    result_list = atlas_agent.query_transient(test_transient)[0]
    for _ in range(10):
        if len(result_list) == 1:
            break
        result_list = atlas_agent.query_transient(test_transient)[0]
    assert len(result_list) == 1
    query_result = result_list[0]
    query_result.objname = test_event["ztf_id"]
    query_result.redshift = test_event["redshift"]
    query_result.internal_names = {test_event["id"], test_event["ztf_id"]}
    helpers.assert_query_result(
        query_result, test_event["ztf_id"], test_event["ra"], test_event["dec"], test_event["redshift"]
    )
