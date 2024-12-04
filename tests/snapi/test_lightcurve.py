import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from numpy.typing import NDArray

from snapi import Filter, Formatter, LightCurve


@pytest.mark.usefixtures("lightcurve_class_setup")
class TestLightCurveInit:
    """Test suite for LightCurve object initialization."""

    sample_arrs: dict[str, NDArray[Any]]
    sample_filt: Filter

    def test_init_from_mag_arrs(self) -> None:
        """Tests for successful initialization from
        individual time and mag arrays."""
        lc = LightCurve.from_arrays(
            phase=self.sample_arrs["time"],
            mags=self.sample_arrs["mag"],
            mag_errs=self.sample_arrs["mag_unc"],
            zpts=self.sample_arrs["zpt"],
            filt=self.sample_filt,
        )

        assert np.allclose(self.sample_arrs["time"], lc.times)
        assert np.allclose(self.sample_arrs["mag"], lc.mags)
        assert np.allclose(self.sample_arrs["mag_unc"], lc.mag_errors)
        # check cross-calculations
        assert np.allclose(self.sample_arrs["flux"], lc.fluxes)
        assert np.allclose(self.sample_arrs["flux_unc"], lc.flux_errors)
        assert (lc.filter is not None) and (lc.filter.instrument == "ZTF")
        assert (lc.filter is not None) and (lc.filter.band == "g")

        # check sorting
        assert np.allclose(lc.times, np.sort(self.sample_arrs["time"]))
        assert np.allclose(lc.fluxes, self.sample_arrs["flux"][np.argsort(self.sample_arrs["time"])])
        assert np.allclose(lc.flux_errors, self.sample_arrs["flux_unc"][np.argsort(self.sample_arrs["time"])])

    def test_init_from_flux_arrs(self) -> None:
        """Tests for successful initialization from
        individual time and flux arrays."""
        lc = LightCurve.from_arrays(
            phase=self.sample_arrs["time"],
            fluxes=self.sample_arrs["flux"],
            flux_errs=self.sample_arrs["flux_unc"],
            zpts=self.sample_arrs["zpt"],
            filt=self.sample_filt,
        )

        assert np.allclose(self.sample_arrs["time"], lc.times)
        assert np.allclose(self.sample_arrs["flux"], lc.fluxes)
        assert np.allclose(self.sample_arrs["flux_unc"], lc.flux_errors)
        # check cross-calculations
        assert np.allclose(self.sample_arrs["mag"], lc.mags)
        assert np.allclose(self.sample_arrs["mag_unc"], lc.mag_errors)
        assert (lc.filter is not None) and (lc.filter.instrument == "ZTF")
        assert (lc.filter is not None) and (lc.filter.band == "g")

        # check sorting
        assert np.allclose(lc.times, np.sort(self.sample_arrs["time"]))
        assert np.allclose(lc.mags, self.sample_arrs["mag"][np.argsort(self.sample_arrs["time"])])
        assert np.allclose(lc.mag_errors, self.sample_arrs["mag_unc"][np.argsort(self.sample_arrs["time"])])

    def test_failed_init(self) -> None:
        """Ensure initializations fail where intended."""
        with pytest.raises(TypeError):
            # no times
            LightCurve.from_arrays(  # pylint: disable=no-value-for-parameter
                mags=self.sample_arrs["mag"],  # type: ignore
                mag_errs=self.sample_arrs["mag_unc"],
                zpts=self.sample_arrs["zpt"],
                filt=self.sample_filt,
            )

        with pytest.raises(TypeError):
            # wrong Filter
            LightCurve.from_arrays(
                times=self.sample_arrs["time"],
                mags=self.sample_arrs["mag"],
                mag_errs=self.sample_arrs["mag_unc"],
                zpts=self.sample_arrs["zpt"],
                filt="ZTF_g",  # type: ignore
            )


@pytest.mark.usefixtures("lightcurve_class_setup")
class TestSetters:
    """Test the setter functions and their checks."""

    lc: LightCurve
    other_overlap: LightCurve

    def test_successful_setters(self) -> None:
        """Tests setter calls that should work."""
        lc_copy = self.lc.copy()
        lc_copy.times = self.other_overlap.times
        assert (lc_copy.times == self.other_overlap.times).all()
        lc_copy.fluxes = self.other_overlap.fluxes
        assert (lc_copy.fluxes == self.other_overlap.fluxes).all()
        # check that fluxes changed accordingly
        assert (lc_copy.mags != self.lc.mags).any()
        assert (lc_copy.mags != self.other_overlap.mags).any()  # cause diff. zeropoints
        # now update zeropoint to match
        lc_copy.zeropoints = self.other_overlap.zeropoints
        assert (lc_copy.mags == self.other_overlap.mags).all()

        # check deeper equivalence through detections attribute
        assert lc_copy.detections[["mag", "flux"]].equals(self.other_overlap.detections[["mag", "flux"]])

    def test_wrong_length_setters(self) -> None:
        """Test setter calls that should fail because new
        column is different size compared to original
        column."""
        with pytest.raises(ValueError):
            self.lc.times = self.other_overlap.times[:-1]

        with pytest.raises(ValueError):
            self.lc.mags = self.other_overlap.mags[:-1]

    def test_wrong_type_setters(self) -> None:
        """Test setter calls that should fail because new
        column is of different type than original column."""
        with pytest.raises(ValueError):
            self.lc.times = np.array(["a", "b", "c", "d"])

        with pytest.raises(TypeError):
            self.lc.mags = np.array(["a", "b", "c", "d"])

        with pytest.raises(TypeError):
            self.lc.filter = "ZTF_r"  # type: ignore

        with pytest.raises(TypeError):
            self.lc.filter = 20.0  # type: ignore


def test_len(test_lightcurve1: LightCurve) -> None:
    """Test that __len__ function is working correctly.
    Should return the number of observations."""
    assert len(test_lightcurve1) == 4


def test_copy(test_lightcurve1: LightCurve) -> None:
    """Test copy() functionality."""
    lc_copy = test_lightcurve1.copy()
    assert test_lightcurve1.detections.equals(lc_copy.detections)
    assert test_lightcurve1.non_detections.equals(lc_copy.non_detections)
    assert test_lightcurve1.filter == lc_copy.filter


def test_eq(test_lightcurve1: LightCurve, test_lightcurve2: LightCurve) -> None:
    """Test __eq__ functionality."""
    lc_copy = test_lightcurve1.copy()
    assert lc_copy == test_lightcurve1
    assert test_lightcurve1 != test_lightcurve2


def test_peak(test_lightcurve1: LightCurve, test_lightcurve2: LightCurve) -> None:
    """Test peak of a LightCurve is calculated correctly."""
    # test normal LC (no repeated mags)
    peak = test_lightcurve1.peak
    assert peak["phase"] == 2.0
    assert peak["mag"] == 14.0
    assert peak["flux"] == 10.0 ** (-1.0 * (14.0 - 23.9) / 2.5)

    # test with repeated mags
    peak = test_lightcurve2.peak
    assert peak["phase"] == 3.0  # incorporates m_unc
    assert peak["mag"] == 20.0
    assert peak["flux"] == 10.0 ** (-1.0 * (20.0 - 25.0) / 2.5)


@pytest.mark.usefixtures("lightcurve_class_setup")
class TestPhase:
    """Tests all modes of phase() function, including periodic
    and shift phasing."""

    lc: LightCurve

    def test_phase_constant_given(self) -> None:
        """Test phase() function of LightCurve where
        no periodic phasing is done and there is a given shift."""
        lc_copy = self.lc.phase(t0=0, inplace=False)  # no shift
        assert lc_copy == self.lc

        lc_copy = self.lc.phase(t0=1, inplace=False)
        shifted_times = lc_copy.times + 1
        assert np.all(shifted_times == self.lc.times)

    def test_phase_constant_not_given(self) -> None:
        """Test phase() function of LightCurve where
        no periodic phasing is done and there is no given shift."""
        lc_copy = self.lc.phase(inplace=False)
        shifted_times = lc_copy.times + 2
        assert np.all(shifted_times == self.lc.times)

    def test_phase_periodic_given(self) -> None:
        """Test phase() function of LightCurve where periodic=True
        and period is given.
        """

    def test_phase_periodic_not_given(self) -> None:
        """Test phase() function of LightCurve where periodic=True
        and period is not given.
        """

    def test_phase_both_given(self) -> None:
        """Test phase() function of LightCurve with both
        phase and period given.
        """

    def test_phase_both_not_given(self) -> None:
        """Test phase() function of LightCurve with both
        phase and period are not given but set to True.
        """


def test_calculate_period() -> None:
    """Tests that period is calculated correctly
    for light curve."""


@pytest.mark.usefixtures("lightcurve_class_setup")
class TestTruncate:
    """Test light curve truncation between min and max times."""

    lc: LightCurve

    def test_truncate_no_limits(self) -> None:
        """Test case where min=-inf and max=+inf"""
        lc_copy = self.lc.copy()
        lc_copy.truncate()
        assert lc_copy == self.lc
        lc_copy.truncate(min_t=-np.inf, max_t=np.inf)
        assert lc_copy == self.lc

    def test_truncate_min_only(self) -> None:
        """Test truncation with only finite minimum."""
        lc_copy = self.lc.copy()
        lc_copy.truncate(min_t=-10.0)
        assert len(lc_copy) == 4
        lc_copy.truncate(min_t=0.0)
        assert len(lc_copy) == 3
        lc_copy.truncate(min_t=100.0)
        assert len(lc_copy) == 0.0

    def test_truncate_max_only(self) -> None:
        """Test truncation with only finite maximum."""
        lc_copy = self.lc.copy()
        lc_copy.truncate(max_t=1000)
        assert len(lc_copy) == 4
        lc_copy.truncate(max_t=0.0)
        assert len(lc_copy) == 1
        lc_copy.truncate(max_t=-6.0)
        assert len(lc_copy) == 0

    def test_truncate_both(self) -> None:
        """Test truncation from both sides."""
        lc_copy = self.lc.copy()
        lc_copy.truncate(min_t=-10.0, max_t=100.0)
        assert len(lc_copy) == 4
        lc_copy.truncate(min_t=0, max_t=2.5)
        assert len(lc_copy) == 2
        lc_copy.truncate(min_t=1.5, max_t=1.75)
        assert len(lc_copy) == 0

    def test_illegal_truncate(self) -> None:
        """Test truncate calls that yield errors."""
        with pytest.raises(ValueError):
            self.lc.truncate(min_t=1.0, max_t=0.0)  # min > max


@pytest.mark.usefixtures("lightcurve_class_setup")
class TestSubsample:
    """Test subsample() function."""

    lc: LightCurve

    def test_subsample_identity(self) -> None:
        """Tests that when n_points >= len(lc)
        that nothing happens."""
        lc_copy = self.lc.copy()
        lc_copy.subsample(10)
        assert lc_copy.detections.equals(self.lc.detections)
        lc_copy.subsample(4)
        assert lc_copy.detections.equals(self.lc.detections)

    def test_subsample_actual(self) -> None:
        """Tests that subsampling is working
        as intended.
        """
        lc_copy = self.lc.copy()
        lc_copy.subsample(2)
        assert len(lc_copy) == 2
        lc_copy.subsample(0)
        assert len(lc_copy) == 0

    def test_subsample_illegal(self) -> None:
        """Tests error modes of subsample."""
        with pytest.raises(ValueError):
            self.lc.subsample(-1)  # n_points < 0


def test_successful_plot(test_lightcurve1: LightCurve, test_formatter: Formatter) -> None:
    """Tests that lc.plot() runs without erroring.

    TODO: manual test case?
    """
    _, ax = plt.subplots()
    test_lightcurve1.plot(ax=ax)  # no formatter
    test_lightcurve1.plot(ax=ax, formatter=test_formatter)
    # alternate kwargs
    test_lightcurve1.plot(ax=ax, formatter=test_formatter, mags=False)


def test_merge_close_times() -> None:
    """Tests observations within eps are combined
    such that their individual and combined uncertainties
    are properly accounted for.
    """


@pytest.mark.usefixtures("lightcurve_class_setup")
class TestMerge:
    """Tests merging two light curves together."""

    lc: LightCurve
    other_no_overlap: LightCurve
    other_overlap: LightCurve

    def test_merge_no_overlap(self) -> None:
        """Tests merging two light curves
        with no temporal overlap.
        """
        other_lc_copy = self.other_no_overlap.copy()
        other_lc_copy.filter = self.lc.filter  # filters must match

        lc_merged = self.lc.merge(other_lc_copy)
        assert len(lc_merged) == 7
        merged_t = lc_merged.times
        for t in self.lc.times:
            assert t in merged_t
        for t in other_lc_copy.times:
            assert t in merged_t

    def test_merge_partial_overlap(self) -> None:
        """Tests merging two light curves
        with partial temporal overlap.
        """
        other_lc_copy = self.other_overlap.copy()
        other_lc_copy.filter = self.lc.filter  # filters must match

        lc_merged = self.lc.merge(other_lc_copy)
        assert len(lc_merged) == 7
        merged_t = lc_merged.times
        for t in self.lc.times:
            assert t in merged_t
        for t in other_lc_copy.times:
            assert t in merged_t

    def test_merge_full_overlap(self) -> None:
        """Tests merging two of the same
        light curve."""
        lc_merged = self.lc.merge(self.lc)
        assert len(lc_merged) == 4
        merged_t = lc_merged.times
        for t in self.lc.times:
            assert t in merged_t

    def test_merge_invalid(self) -> None:
        """Tests merging two LC
        with different filters."""
        with pytest.raises(ValueError):
            self.lc.merge(self.other_overlap)


@pytest.mark.usefixtures("lightcurve_class_setup")
class TestPad:
    """Tests padding of LightCurve with fill values."""

    lc: LightCurve
    fill_dict: dict[str, Any]

    def test_pad_illegal(self) -> None:
        """Tests illegal padding: n_times < 0.
        Illegal 'fill' values handled by add_observation()
        """
        with pytest.raises(ValueError):
            self.lc.pad(fill=self.fill_dict, n_times=-1)  # n_times < 0

    def test_pad_valid(self) -> None:
        """Tests valid padding."""
        lc_copy = self.lc.pad(fill=self.fill_dict, n_times=2)
        assert len(lc_copy) == 6
        lc_copy = self.lc.pad(fill=self.fill_dict, n_times=0)
        assert len(lc_copy) == 4


@pytest.mark.usefixtures("lightcurve_class_setup")
class TestResample:
    """Tests light curve resampling from
    uncertainty values."""

    lc: LightCurve

    def test_resample_single(self) -> None:
        """Tests single resampled mags are
        drawn properly from data.
        """
        resampled_mags = self.lc.resample(num=1)
        assert len(resampled_mags[0]) == len(self.lc)
        assert (
            np.all(np.abs(resampled_mags[0] - self.lc.mags) / self.lc.mag_errors) < 10
        )  # 10-sigma difference is VERY unlikely

    def test_resample_illegal(self) -> None:
        """Tests resample() in case where num < 0."""
        with pytest.raises(ValueError):
            self.lc.resample(num=-1)


def test_absolute(test_lightcurve1: LightCurve) -> None:
    """Tests conversion from apparent to absolute
    light curve."""
    lc_copy = test_lightcurve1.phase(inplace=False)
    abs_lc = lc_copy.absolute(redshift=0.1)
    assert np.isclose(abs_lc.times, lc_copy.times / 1.1).all()
    assert not np.isclose(abs_lc.mags, lc_copy.mags).any()
    assert np.isclose(abs_lc.fluxes, lc_copy.fluxes).all()  # both zp and mag adjusted


@pytest.mark.usefixtures("lightcurve_class_setup")
class TestCorrectExtinction:
    """Tests correct_extinction() function for both
    provided MWEBV or coordinates."""

    lc: LightCurve
    coord: SkyCoord

    def test_correct_extinction_mwebv(self) -> None:
        """Tests correct_extinction() when only
        mwebv is provided.
        """
        mwebv1 = 0.0
        corrected_lc: LightCurve = self.lc.correct_extinction(mwebv=mwebv1)
        assert corrected_lc == self.lc

        mwebv2 = 1.0
        corrected_lc = self.lc.correct_extinction(mwebv=mwebv2)
        assert (corrected_lc.mags < self.lc.mags).all()

    def test_correct_extinction_coords(self) -> None:
        """Tests correct_extinction when only
        coordinate is provided.
        """
        corrected_lc: LightCurve = self.lc.correct_extinction(coordinates=self.coord)
        assert (corrected_lc.mags < self.lc.mags).all()

    def test_correct_extinction_illegal(self) -> None:
        """Test when inconsistent mwebv and coordinates
        are both provided, and when neither are."""
        with pytest.raises(ValueError):
            self.lc.correct_extinction()
        with pytest.raises(ValueError):
            self.lc.correct_extinction(mwebv=0.0, coordinates=self.coord)


@pytest.mark.usefixtures("lightcurve_class_setup")
class TestConvertToImage:
    """Test functionality for converting light curves
    into 2D images for CNNs."""

    lc: LightCurve

    def test_convert_to_image_gaf(self) -> None:
        """Test Gramian angular field
        generation."""
        self.lc.convert_to_images(method="gaf", augment=False)

    def test_convert_to_image_dff_dt(self) -> None:
        """Test dff-dt image
        generation."""
        self.lc.convert_to_images(augment=False)

    def test_convert_to_image_mtf(self) -> None:
        """Test Markov Transition Field
        generation."""
        self.lc.convert_to_images(method="mtf", augment=False)

    def test_convert_to_image_recurrence(self) -> None:
        """Test recurrence plot
        generation."""
        self.lc.convert_to_images(method="recurrence", augment=False)

    def test_convert_to_image_augment(self) -> None:
        """Test convert-to-image methods but with augmentations."""
        self.lc.convert_to_images(method="dff-dt")
        self.lc.convert_to_images(method="gaf")
        self.lc.convert_to_images(method="mtf")
        self.lc.convert_to_images(method="recurrence")


@pytest.mark.usefixtures("lightcurve_class_setup")
class TestSaveLoad:
    """Test save/load functionalities for LightCurve."""

    save_fn: str
    load_fn: str
    lc: LightCurve

    def test_save(self) -> None:
        """Tests saving LightCurve to an h5 file."""
        if os.path.exists(self.save_fn):
            os.remove(self.save_fn)  # regenerate it each time
        self.lc.save(self.save_fn)
        assert os.path.exists(self.save_fn)

    def test_load(self) -> None:
        """Tests loading LightCurve from h5 file."""
        lc_loaded = LightCurve.load(self.load_fn)
        assert lc_loaded == self.lc

    def test_save_load(self) -> None:
        """Tests full save/load cycle."""
        if os.path.exists(self.save_fn):
            os.remove(self.save_fn)  # regenerate it each time
        self.lc.save(self.save_fn)
        lc_loaded = LightCurve.load(self.save_fn)
        assert lc_loaded == self.lc
