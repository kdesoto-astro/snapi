"""Test Spectrum functionality."""
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.typing import NDArray

from snapi import Formatter, Spectrometer, Spectrum


@pytest.mark.usefixtures("spectrum_class_setup")
class TestSpectrumInit:
    """Test suite for Spectrum object initialization."""

    sample_arrs: dict[str, NDArray[Any]]
    spectrometer: Spectrometer

    def test_init_from_arrs(self) -> None:
        """Tests for successful initialization from
        individual flux and error arrays."""
        spectrum = Spectrum(
            fluxes=self.sample_arrs["flux"],
            errors=self.sample_arrs["errors"],
            spectrometer=self.spectrometer,
        )

        assert np.allclose(self.sample_arrs["flux"], spectrum.fluxes)
        assert np.allclose(self.sample_arrs["errors"], spectrum.errors)
        assert (spectrum.spectrometer is not None) and (
            spectrum.spectrometer.instrument == "test_spectrometer"
        )
        assert np.allclose(spectrum.spectrometer.wavelengths, spectrum.wavelengths)

    def test_init_flux_padding(self) -> None:
        """Tests for successful initialization from
        individual flux and error arrays."""
        spectrum = Spectrum(
            fluxes=self.sample_arrs["flux"][:-2],
            errors=self.sample_arrs["errors"],
            spectrometer=self.spectrometer,
        )

        assert np.allclose(self.sample_arrs["flux"][:-2], spectrum.fluxes[:-2])
        assert np.allclose(self.sample_arrs["errors"], spectrum.errors)
        assert (spectrum.spectrometer is not None) and (
            spectrum.spectrometer.instrument == "test_spectrometer"
        )
        assert np.allclose(spectrum.spectrometer.wavelengths, spectrum.wavelengths)

        spectrum = Spectrum(
            fluxes=[],
            errors=[],
            spectrometer=self.spectrometer,
        )

        assert np.all(np.isnan(spectrum.fluxes))
        assert np.all(np.isnan(spectrum.errors))
        assert (spectrum.spectrometer is not None) and (
            spectrum.spectrometer.instrument == "test_spectrometer"
        )
        assert np.allclose(spectrum.spectrometer.wavelengths, spectrum.wavelengths)

    def test_failed_init(self) -> None:
        """Ensure initializations fail where intended."""
        with pytest.raises(ValueError):
            # array lengths
            Spectrum(
                fluxes=np.repeat(self.sample_arrs["flux"], 2),
                errors=np.repeat(self.sample_arrs["errors"], 2),
                spectrometer=self.spectrometer,
            )


def test_successful_plot(test_spectrum1: Spectrum, test_formatter: Formatter) -> None:
    """Tests that test_spectrum1.plot() runs without erroring.

    TODO: manual test case?
    """
    _, ax = plt.subplots()
    test_spectrum1.plot(ax=ax)  # no formatter
    test_spectrum1.plot(ax=ax, formatter=test_formatter)
    # alternate kwargs
    test_spectrum1.plot(ax=ax, formatter=test_formatter, normalize=False)
    test_spectrum1.plot(ax=ax, formatter=test_formatter, annotate=True)
    test_spectrum1.plot(
        ax=ax,
        formatter=test_formatter,
        overlay_lines=[
            "H",
        ],
    )
    test_spectrum1.plot(ax=ax, formatter=test_formatter, offset=1.0)
