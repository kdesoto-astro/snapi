"""Test object functions for photometry.py."""
import copy
import os

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest

from snapi import Filter, Formatter, LightCurve, Photometry


class TestPhotometryInit:
    """Tests valid and invalid initialization of Photometry object."""

    def test_invalid_init(self) -> None:
        """Tests invalid initialization"""
        with pytest.raises(TypeError):
            Photometry(
                [
                    "lc1",  # type: ignore
                ]
            )

    def test_empty_init(self) -> None:
        """Test initialization without light curves."""
        phot = Photometry()
        assert len(phot.light_curves) == 0

        # check attributes
        assert len(phot.times) == 0
        assert len(phot.mags) == 0
        assert len(phot.filters) == 0

    def test_init_with_lcs(self, test_lightcurve1: LightCurve, test_lightcurve2: LightCurve) -> None:
        """Test initialization with light curves."""
        phot = Photometry([test_lightcurve1, test_lightcurve2])
        assert len(phot.light_curves) == 2
        for lc in phot.light_curves:
            assert len(lc) == 4
            assert lc.filter is not None

        # check attributes
        assert len(phot.times) == 8
        assert len(phot.mags) == 8
        assert len(phot.filters) == 8


def test_filter_by_instrument(test_photometry: Photometry, test_lightcurve1: LightCurve) -> None:
    """Test filtering photometry by instrument."""
    filtered_phot1 = test_photometry.filter_by_instrument("ZTF")
    assert len(filtered_phot1) == 2

    filtered_phot_empty = test_photometry.filter_by_instrument("PS1")
    assert len(filtered_phot_empty) == 0

    # partial change
    test_lightcurve_ps1 = test_lightcurve1.copy()
    test_lightcurve_ps1.filter = Filter(
        instrument="PS1", band="r", center=5000.0 * u.AA  # pylint: disable=no-member
    )

    new_phot = Photometry([test_lightcurve1, test_lightcurve_ps1])
    filtered_phot_partial = new_phot.filter_by_instrument("ZTF")
    assert len(filtered_phot_partial) == 1


def test_filter(test_photometry: Photometry, sample_filt: Filter) -> None:
    """Test the filter() function."""
    filtered_phot = test_photometry.filter(sample_filt)
    assert len(filtered_phot) == 1

    # can also filter by string
    filtered_phot2 = test_photometry.filter("ZTF_r")
    assert len(filtered_phot2) == 1


def test_normalize(test_photometry: Photometry) -> None:
    """Test the normalize method of photometry.py."""
    phot_copy = test_photometry.normalize()
    assert np.max(phot_copy.detections["flux"]) == 1.0


def test_phase(test_photometry: Photometry) -> None:
    """Test the phase method of photometry.py."""
    goal_arr = np.array([-7.5, -2.5, -1.5, -0.5, 0.0, 0.5, 0.5, 1.5])
    phot_copy2 = test_photometry.phase(2.5, inplace=False)  # should also phase by 2.5
    assert np.all(phot_copy2.times == goal_arr)
    phot_copy = test_photometry.phase(inplace=False)  # should phase by 2.5
    assert np.all(phot_copy.times == goal_arr)
    # TODO: add test for periodic phasing


def test_calculate_period() -> None:
    """Calculate the period of a periodic multi-band signal."""


def test_plot_no_errors(test_photometry: Photometry, test_formatter: Formatter) -> None:
    """Test that plot function runs without errors."""
    _, ax = plt.subplots()
    test_photometry.plot(ax=ax)  # default formatter
    test_photometry.plot(ax=ax, formatter=test_formatter)
    test_photometry.plot(ax=ax, formatter=test_formatter, mags=False)


def test_add_lightcurve(test_lightcurve1: LightCurve, test_lightcurve2: LightCurve) -> None:
    """Test the add_lightcurve method of photometry.py."""
    phot = Photometry()
    phot.add_lightcurve(test_lightcurve1, inplace=True)
    assert len(phot.detections) == 4
    assert len(phot) == 1
    phot.add_lightcurve(test_lightcurve2, inplace=True)
    assert len(phot.detections) == 8
    assert len(phot) == 2
    phot.add_lightcurve(test_lightcurve1, inplace=True)  # test merge functionality
    assert len(phot.detections) == 8
    assert len(phot) == 2


def test_remove_lightcurve(test_photometry: Photometry, test_lightcurve1: LightCurve) -> None:
    """Tests removing light curve from Photometry."""
    phot = test_photometry.remove_lightcurve(str(test_lightcurve1.filter))
    assert len(phot) == 1
    assert phot.light_curves[0] != test_lightcurve1

"""
class TestTile:
    Tests tiling of extra light curves to Photometry.

    def test_tile_invalid(self, test_photometry: Photometry) -> None:
        Test case where n_lightcurves < len(photometry)
        with pytest.raises(ValueError):
            test_photometry.tile(1, inplace=True)

    def test_tile_valid(
        self, test_photometry: Photometry, test_lightcurve1: LightCurve, test_lightcurve2: LightCurve
    ) -> None:
        Test that tiled light curve matches one of the original
        light curves.
        phot = test_photometry.tile(3)
        assert len(phot) == 3
        for lc in phot.light_curves:
            assert lc.detections.equals(test_lightcurve1.detections) or lc.detections.equals(
                test_lightcurve2.detections
            )
"""


def test_len(test_photometry: Photometry) -> None:
    """Tests __len__ reflects number of LCs."""
    assert len(test_photometry) == 2
    assert len(Photometry()) == 0


def test_eq(test_photometry: Photometry, test_lightcurve1: LightCurve, test_lightcurve2: LightCurve) -> None:
    """Tests __eq__ function of photometry."""
    assert test_photometry.copy() == test_photometry
    assert Photometry([test_lightcurve2, test_lightcurve1]) == test_photometry


def test_dense_array(test_photometry: Photometry) -> None:
    """Test generation of dense arrays.
    TODO: add to this"""
    dense_arr = test_photometry.dense_array()
    assert dense_arr.shape == (2, 7, 6)  # num UNIQUE timestamps x (1 + 5*nfilts)


def test_absolute(test_photometry: Photometry) -> None:
    """Test conversion of photometry to absolute units
    from redshifts."""
    phot = test_photometry.phase(inplace=False)
    abs_phot = phot.absolute(redshift=0.1)
    assert np.isclose(abs_phot.times, phot.times / 1.1).all()
    assert not np.isclose(abs_phot.mags, phot.mags).any()
    assert np.isclose(abs_phot.fluxes, phot.fluxes).all()  # both zp and mag adjusted


def test_correct_extinction(test_photometry: Photometry) -> None:
    """Test extinction correct of photometry."""
    mwebv1 = 0.0
    corrected_phot: Photometry = test_photometry.correct_extinction(mwebv=mwebv1)
    assert corrected_phot == test_photometry

    mwebv2 = 1.0
    corrected_phot = test_photometry.correct_extinction(mwebv=mwebv2)
    assert (corrected_phot.mags < test_photometry.mags).all()
    assert np.allclose(corrected_phot.fluxes, test_photometry.fluxes)  # photon counts the same


class TestSaveLoad:
    """Test save/load functionalities for Photometry."""

    def test_save(self, data_dir: str, test_photometry: Photometry) -> None:
        """Tests saving Photometry to an h5 file."""
        save_fn = os.path.join(data_dir, "test_phot_save.h5")
        if os.path.exists(save_fn):
            os.remove(save_fn)  # regenerate it each time
        test_photometry.save(save_fn)
        assert os.path.exists(save_fn)

    def test_load(self, data_dir: str, test_photometry: Photometry) -> None:
        """Tests loading LightCurve from h5 file."""
        load_fn = os.path.join(data_dir, "test_phot_load.h5")
        phot_loaded = Photometry.load(load_fn)
        assert phot_loaded == test_photometry

    def test_save_load(self, data_dir: str, test_photometry: Photometry) -> None:
        """Tests full save/load cycle."""
        save_fn = os.path.join(data_dir, "test_phot_save.h5")
        if os.path.exists(save_fn):
            os.remove(save_fn)  # regenerate it each time
        test_photometry.save(save_fn)
        phot_loaded = Photometry.load(save_fn)
        assert phot_loaded == test_photometry
