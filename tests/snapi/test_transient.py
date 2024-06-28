import pytest

from snapi.transient import tns_object_helper, tns_search_helper


@pytest.mark.skip_precommit
@pytest.mark.local  # local because API keys are not on Github
def test_tns_search_helper() -> None:
    """Test that the TNS search queries
    are working as intended.
    """
    # check internal name search
    test_internal = "ZTF23aaklqou"  # 2023ixf
    r = tns_search_helper(internal_name=test_internal)
    assert len(r) == 1
    assert r[0]["objname"] == "2023ixf"
    assert r[0]["prefix"] == "SN"

    # check RA/dec check
    test_ra = "14:03:38.562"  # 2023ixf
    test_dec = "+54:18:41.94"
    r = tns_search_helper(ra=test_ra, dec=test_dec, radius=1)
    assert len(r) == 1
    assert r[0]["objname"] == "2023ixf"
    assert r[0]["prefix"] == "SN"


@pytest.mark.skip_precommit
@pytest.mark.local  # local because API keys are not on Github
def test_tns_object_helper() -> None:
    """Test that the TNS object retrieval
    function is working as intended.
    """
    sample_iau_name = "2023ixf"
    r = tns_object_helper(sample_iau_name)

    assert r["objname"] == sample_iau_name
    assert len(r["photometry"]) > 0
    assert len(r["spectra"]) > 0

    fake_iau_name = "2025zzz"
    r = tns_object_helper(fake_iau_name)
    assert "objname" not in r  # no results
