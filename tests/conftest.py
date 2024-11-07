import os
from typing import Any

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from numpy.typing import NDArray

from snapi import Filter, Formatter, LightCurve, Photometry, Spectrometer, Spectroscopy, Spectrum, Transient
from snapi.query_agents import (
    ALeRCEQueryAgent,
    ANTARESQueryAgent,
    ATLASQueryAgent,
    GHOSTQueryAgent,
    TNSQueryAgent,
)

@pytest.fixture(scope="class")
def test_rng() -> np.random.Generator:
    """numpy rng for all test fixtures."""
    return np.random.default_rng()


@pytest.fixture(scope="class")
def data_dir() -> str:
    """Where data for tests are stored."""
    two_up = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(two_up, "data/tests")
    os.makedirs(path, exist_ok=True)
    return path


@pytest.fixture(scope="class")
def tns_agent() -> TNSQueryAgent:
    """TNS query agent fixture."""
    return TNSQueryAgent()


@pytest.fixture(scope="class")
def alerce_agent() -> ALeRCEQueryAgent:
    """ALeRCE query agent fixture."""
    return ALeRCEQueryAgent()


@pytest.fixture(scope="class")
def antares_agent() -> ANTARESQueryAgent:
    """ANTARES query agent fixture."""
    return ANTARESQueryAgent()


@pytest.fixture(scope="class")
def ghost_agent() -> GHOSTQueryAgent:
    """GHOST query agent fixture."""
    return GHOSTQueryAgent()


@pytest.fixture(scope="class")
def atlas_agent() -> ATLASQueryAgent:
    """ATLAS query agent fixture."""
    return ATLASQueryAgent()


@pytest.fixture(scope="class")
def test_coordinate() -> SkyCoord:
    """Test coordinate."""
    return SkyCoord(ra=100.0 * u.deg, dec=30.0 * u.deg)  # pylint: disable=no-member


@pytest.fixture(scope="class")
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


@pytest.fixture(scope="class")
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


@pytest.fixture(scope="class")
def sample_arrs() -> dict[str, Any]:
    """Arrays of times/mags/etc. for
    LC initialization."""
    info = {
        "time": np.array([-5.0, 1.0, 2.0, 3.0]),
        "mag": np.array([21, 20, 14, 18]),
        "mag_unc": np.array([0.3, 0.1, 0.2, 0.4]),
        "zpt": np.array([24.0, 24.2, 23.9, 24.1]),
    }
    info["flux"] = 10.0 ** (-1.0 * (info["mag"] - info["zpt"]) / 2.5)
    info["flux_unc"] = (np.log(10.0) / 2.5) * (info["flux"] * info["mag_unc"])
    return info


@pytest.fixture(scope="class")
def sample_filt() -> Filter:
    """Example SNAPI Filter for testing."""
    return Filter(
        band="g",
        instrument="ZTF",
        center=4741.64 * u.AA,  # pylint: disable=no-member
    )


@pytest.fixture(scope="class")
def fill_dict() -> dict[str, Any]:
    return {"time": pd.to_timedelta(1000.0, "D"), "non_detections": True, "mags": 28.0}


@pytest.fixture(scope="class")
def test_lightcurve1(sample_arrs: dict[str, NDArray[Any]], sample_filt: Filter) -> LightCurve:
    """Test lightcurve fixture."""
    return LightCurve(
        times=sample_arrs["time"],
        mags=sample_arrs["mag"],
        mag_errs=sample_arrs["mag_unc"],
        zpts=sample_arrs["zpt"],
        filt=sample_filt,
        phased=True,
    )


@pytest.fixture(scope="class")
def test_lightcurve2() -> LightCurve:
    """Test lightcurve fixture."""
    test_filter = Filter(
        band="r",
        instrument="ZTF",
        center=6173.23 * u.AA,  # pylint: disable=no-member
    )
    return LightCurve(
        times=[0.0, 3.0, 2.5, 4.0],
        mags=[21, 20, 20, 20],
        mag_errs=[0.3, 0.1, 0.2, 0.5],
        zpts=[25.0, 25.0, 25.0, 25.0],
        filt=test_filter,
        phased=True,
    )


@pytest.fixture(scope="class")
def test_spectrum_arrs(test_rng: np.random.Generator) -> dict[str, NDArray[np.float64]]:
    """Arrays to make test spectrum objects."""
    return {"flux": 10.0 * test_rng.normal(size=10), "errors": np.abs(test_rng.normal(size=10))}


@pytest.fixture(scope="class")
def test_spectrum_arrs2(test_rng: np.random.Generator) -> dict[str, NDArray[np.float64]]:
    """Arrays to make test spectrum objects."""
    return {"flux": 5.0 * test_rng.normal(size=10) + 30.0, "errors": np.abs(test_rng.normal(size=10)) / 2.0}


@pytest.fixture(scope="class")
def test_spectrometer(test_spectrum_arrs: dict[str, NDArray[np.float64]]) -> Spectrometer:
    """Test spectrometer fixture."""
    return Spectrometer(
        instrument="test_spectrometer",
        wavelength_start=4000.0 * u.AA,  # pylint: disable=no-member
        wavelength_delta=1.0 * u.AA,  # pylint: disable=no-member
        num_channels=len(test_spectrum_arrs["flux"]),
    )


@pytest.fixture(scope="class")
def test_spectrometer2(test_spectrum_arrs2: dict[str, NDArray[np.float64]]) -> Spectrometer:
    """Test spectrometer fixture."""
    return Spectrometer(
        instrument="test_spectrometer2",
        wavelength_start=5000.0 * u.AA,  # pylint: disable=no-member
        wavelength_delta=2.0 * u.AA,  # pylint: disable=no-member
        num_channels=len(test_spectrum_arrs2["flux"]),
    )


@pytest.fixture(scope="class")
def test_spectrum1(
    test_spectrum_arrs: dict[str, NDArray[np.float64]], test_spectrometer: Spectrometer
) -> Spectrum:
    """Test spectrum fixture."""
    return Spectrum(
        time=1.0 * u.day,  # pylint: disable=no-member
        fluxes=test_spectrum_arrs["flux"],
        errors=test_spectrum_arrs["errors"],
        spectrometer=test_spectrometer,
    )


@pytest.fixture(scope="class")
def test_spectrum2(
    test_spectrum_arrs2: dict[str, NDArray[np.float64]], test_spectrometer2: Spectrometer
) -> Spectrum:
    """Test spectrum fixture."""
    return Spectrum(
        time=1.0 * u.day,  # pylint: disable=no-member
        fluxes=test_spectrum_arrs2["flux"],
        errors=test_spectrum_arrs2["errors"],
        spectrometer=test_spectrometer2,
    )


@pytest.fixture(scope="class")
def test_spectroscopy(test_spectrum1: Spectrum, test_spectrum2: Spectrum) -> Spectroscopy:
    """Test spectroscopy fixture."""
    return Spectroscopy([test_spectrum1, test_spectrum2])


@pytest.fixture(scope="class")
def test_spectrum_arrs(test_rng: np.random.Generator) -> dict[str, NDArray[np.float64]]:
    """Arrays to make test spectrum objects."""
    return {"flux": 10.0 * test_rng.normal(size=10), "errors": np.abs(test_rng.normal(size=10))}


@pytest.fixture(scope="class")
def test_spectrum_arrs2(test_rng: np.random.Generator) -> dict[str, NDArray[np.float64]]:
    """Arrays to make test spectrum objects."""
    return {"flux": 5.0 * test_rng.normal(size=10) + 30.0, "errors": np.abs(test_rng.normal(size=10)) / 2.0}


@pytest.fixture(scope="class")
def test_spectrometer(test_spectrum_arrs: dict[str, NDArray[np.float64]]) -> Spectrometer:
    """Test spectrometer fixture."""
    return Spectrometer(
        instrument="test_spectrometer",
        wavelength_start=4000.0 * u.AA,  # pylint: disable=no-member
        wavelength_delta=1.0 * u.AA,  # pylint: disable=no-member
        num_channels=len(test_spectrum_arrs["flux"]),
    )


@pytest.fixture(scope="class")
def test_spectrometer2(test_spectrum_arrs2: dict[str, NDArray[np.float64]]) -> Spectrometer:
    """Test spectrometer fixture."""
    return Spectrometer(
        instrument="test_spectrometer2",
        wavelength_start=5000.0 * u.AA,  # pylint: disable=no-member
        wavelength_delta=2.0 * u.AA,  # pylint: disable=no-member
        num_channels=len(test_spectrum_arrs2["flux"]),
    )


@pytest.fixture(scope="class")
def test_spectrum1(
    test_spectrum_arrs: dict[str, NDArray[np.float64]], test_spectrometer: Spectrometer
) -> Spectrum:
    """Test spectrum fixture."""
    return Spectrum(
        time=1.0 * u.day,  # pylint: disable=no-member
        fluxes=test_spectrum_arrs["flux"],
        errors=test_spectrum_arrs["errors"],
        spectrometer=test_spectrometer,
    )


@pytest.fixture(scope="class")
def test_spectrum2(
    test_spectrum_arrs2: dict[str, NDArray[np.float64]], test_spectrometer2: Spectrometer
) -> Spectrum:
    """Test spectrum fixture."""
    return Spectrum(
        time=1.0 * u.day,  # pylint: disable=no-member
        fluxes=test_spectrum_arrs2["flux"],
        errors=test_spectrum_arrs2["errors"],
        spectrometer=test_spectrometer2,
    )


@pytest.fixture(scope="class")
def test_spectroscopy(test_spectrum1: Spectrum, test_spectrum2: Spectrum) -> Spectroscopy:
    """Test spectroscopy fixture."""
    return Spectroscopy([test_spectrum1, test_spectrum2])


@pytest.fixture(scope="class")
def test_photometry(
    test_lightcurve1: LightCurve, test_lightcurve2: LightCurve
) -> Photometry:  # pylint: disable=redefined-outer-name
    """Test photometry fixture."""
    return Photometry(lcs=[test_lightcurve1, test_lightcurve2])


@pytest.fixture(scope="class")
def test_transient(test_photometry: Photometry) -> Transient:  # pylint: disable=redefined-outer-name
    """Test transient fixture."""
    return Transient(
        iid="test_transient",
        ra=100 * u.deg,  # pylint: disable=no-member
        dec=30 * u.deg,  # pylint: disable=no-member
        redshift=0.01,
        internal_names={"transient1", "transient2"},
        photometry=test_photometry,
    )


@pytest.fixture(scope="class")
def test_formatter() -> Formatter:
    """Test formatter initialized for
    plotting checks."""
    return Formatter()


class Helpers:  # pylint: disable=too-few-public-methods
    """Helper functions for test cases."""

    @staticmethod
    def assert_query_result(  # pylint: disable=too-many-positional-arguments
        query_result: Any,
        iid: str,
        ra: float,
        dec: float,
        z: float,
        phot_spec: bool = True,
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


@pytest.fixture(scope="class")
def helpers() -> Any:
    """Returns Helpers fixture."""
    return Helpers


@pytest.fixture(scope="class")
def lightcurve_class_setup(  # pylint: disable=too-many-positional-arguments
    request: pytest.FixtureRequest,
    sample_arrs: dict[str, Any],
    sample_filt: Filter,
    test_lightcurve1: LightCurve,
    test_lightcurve2: LightCurve,
    test_coordinate: SkyCoord,
    data_dir: str,
    fill_dict: dict[str, Any],
) -> None:
    """Set up attributes for the class."""
    request.cls.sample_arrs = sample_arrs
    request.cls.sample_filt = sample_filt
    request.cls.lc = test_lightcurve1
    request.cls.other_overlap = test_lightcurve2  # t = 3 in both
    request.cls.other_no_overlap = LightCurve(
        times=request.cls.other_overlap.detections[request.cls.other_overlap.times != 3.0],
        filt=request.cls.other_overlap.filter,
    )
    request.cls.coord = test_coordinate
    request.cls.save_fn = os.path.join(data_dir, "test_lc_save.h5")
    request.cls.load_fn = os.path.join(data_dir, "test_lc_load.h5")
    request.cls.fill_dict = fill_dict


@pytest.fixture(scope="class")
def spectrum_class_setup(
    request: pytest.FixtureRequest,
    test_spectrum_arrs: dict[str, Any],
    test_spectrometer: Spectrometer,
    test_spectrometer2: Spectrometer,
    test_spectrum1: Spectrum,
    test_spectrum2: Spectrum,
    data_dir: str,
) -> None:
    """Set up attributes for the class."""
    request.cls.sample_arrs = test_spectrum_arrs
    request.cls.spectrometer = test_spectrometer
    request.cls.spectrometer2 = test_spectrometer2
    request.cls.spec = test_spectrum1
    request.cls.spec = test_spectrum2
    request.cls.save_fn = os.path.join(data_dir, "test_spectrum_save.h5")
    request.cls.load_fn = os.path.join(data_dir, "test_spectrum_load.h5")
