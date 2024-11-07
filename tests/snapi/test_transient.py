"""Test object functions for transient.py."""
import copy
import os

import astropy.units as u

# import numpy as np
# import pytest
from astropy.coordinates import SkyCoord

from snapi import LightCurve, Photometry, Spectroscopy, Spectrum, Transient


class TestTransientInit:
    """Tests valid and invalid initialization of Transient object."""

    def test_invalid_init(self) -> None:
        """Tests invalid initialization"""

    def test_empty_init(self) -> None:
        """Test initialization with no fields."""
        transient = Transient()
        assert len(transient.photometry) == 0
        assert len(transient.spectroscopy) == 0
        assert transient.id == ""
        assert transient.coordinates is None
        assert transient.redshift is None
        assert transient.spec_class is None

    def test_init_with_fields(self, test_photometry: Photometry, test_spectroscopy: Spectroscopy) -> None:
        """Test initialization with light curves."""
        transient = Transient(
            iid="2020test",
            internal_names={
                "testalias",
            },
            ra=180.0 * u.deg,  # pylint: disable=no-member
            dec=30.0 * u.deg,  # pylint: disable=no-member
            photometry=test_photometry,
            spectroscopy=test_spectroscopy,
            spec_class="SN Ia",
            redshift=0.1,
        )

        assert transient.id == "2020test"
        assert transient.redshift == 0.1
        assert transient.spec_class == "SN Ia"
        assert transient.coordinates == SkyCoord(
            ra=180.0 * u.deg, dec=30.0 * u.deg, frame="icrs"  # pylint: disable=no-member
        )
        assert "testalias" in transient.internal_names

        # TODO: photometry + spectroscopy checks

    def test_init_main_name(self) -> None:
        """Test that _choose_main_name is working behind
        the scenes."""
        transient = Transient(internal_names={"testalias", "2020test"})
        assert transient.id == "2020test"
        transient2 = Transient(
            iid="testalias",
            internal_names={
                "2020test",
            },
        )
        assert transient2.id == "2020test"


def test_add_lightcurve(test_lightcurve1: LightCurve, test_lightcurve2: LightCurve) -> None:
    """Test the add_lightcurve method of photometry.py."""
    transient = Transient()
    transient.add_lightcurve(test_lightcurve1)
    assert len(transient.photometry.detections) == 4
    assert len(transient.photometry) == 1

    transient2 = Transient()
    transient2.add_lightcurves([test_lightcurve1, test_lightcurve2])
    assert len(transient2.photometry.detections) == 8
    assert len(transient2.photometry) == 2


def test_add_spectrum(test_spectrum1: Spectrum, test_spectrum2: Spectrum) -> None:
    """Test the add_lightcurve method of photometry.py."""
    transient = Transient()
    transient.add_spectrum(test_spectrum1)
    assert len(transient.spectroscopy) == 1

    transient2 = Transient()
    transient2.add_spectra([test_spectrum1, test_spectrum2])
    assert len(transient2.spectroscopy) == 2


def test_len(test_transient: Transient) -> None:
    """Tests __len__ reflects number of LCs + spectra"""
    assert len(test_transient) == 2
    assert len(Transient()) == 0


def test_eq(test_transient: Transient, test_lightcurve1: LightCurve, test_spectrum1: Spectrum) -> None:
    """Tests __eq__ function of photometry."""
    assert copy.deepcopy(test_transient) == test_transient
    assert (
        Transient(
            spectroscopy=Spectroscopy(
                [
                    test_spectrum1,
                ]
            ),
            photometry=Photometry(
                [
                    test_lightcurve1,
                ]
            ),
        )
        == test_transient
    )


class TestIngestQueryInfo:
    """Test ingestion of query info in dictionary form."""

    def test_ingest_query_info_empty_transient() -> None:
        """Test ingest_query_info() into an empty Transient object."""

    def test_ingest_query_info_overlap() -> None:
        """Test ingest_query_info() into a Transient with pre-filled
        fields."""


class TestSaveLoad:
    """Test save/load functionalities for Transient."""

    def test_save(self, data_dir: str, test_transient: Transient) -> None:
        """Tests saving Transient to an h5 file."""
        save_fn = os.path.join(data_dir, "test_transient_save.h5")
        if os.path.exists(save_fn):
            os.remove(save_fn)  # regenerate it each time
        test_transient.save(save_fn)
        assert os.path.exists(save_fn)

    def test_load(self, data_dir: str, test_transient: Transient) -> None:
        """Tests loading Transient from h5 file."""
        load_fn = os.path.join(data_dir, "test_transient_load.h5")
        transient_loaded = Transient.load(load_fn)
        assert transient_loaded == test_transient

    def test_save_load(self, data_dir: str, test_transient: Transient) -> None:
        """Tests full save/load cycle."""
        save_fn = os.path.join(data_dir, "test_transient_save.h5")
        if os.path.exists(save_fn):
            os.remove(save_fn)  # regenerate it each time
        test_transient.save(save_fn)
        transient_loaded = Transient.load(save_fn)
        assert transient_loaded == test_transient
