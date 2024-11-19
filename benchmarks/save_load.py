# Benchmark save/load functionality
import numpy as np

from snapi import (
    Spectrometer,
    Spectrum,
    Filter,
    LightCurve,
    Spectroscopy,
    Photometry,
    Transient,
    TransientGroup
)

class TransientGroupSaveLoad:
    """Benchmark save/load times for a TransientGroup."""
    
    def setup_cache(self):
        """Set up the TransientGroup."""
        transients = []
        test_spectrometer = Spectrometer(
            instrument=f"test_spectrometer",
            wavelength_start=4000.0,
            wavelength_delta=1.0,
            num_channels=10,
        )
        test_spectrum = Spectrum(
            time=1.0 * u.day,  # pylint: disable=no-member
            fluxes=np.linspace(5,10,num=10),
            errors=np.linspace(0,1,num=10),
            spectrometer=test_spectrometer,
        )
        test_filter = Filter(
            band="r",
            instrument="ZTF",
            center=6173.23,
        )
        test_lightcurve = LightCurve(
            times=[0.0, 3.0, 2.5, 4.0],
            mags=[21, 20, 20, 20],
            mag_errs=[0.3, 0.1, 0.2, 0.5],
            zpts=[25.0, 25.0, 25.0, 25.0],
            filt=test_filter,
            phased=True,
        )
        for i in range(100):
            transients.append(
                Transient(
                    iid=f"test_transient_{i}",
                    ra=100,
                    dec=30,
                    redshift=0.01,
                    internal_names={},
                    photometry=Photometry([test_lightcurve,]),
                    spectroscopy=Spectroscopy([test_spectrum,]),
                )
            )
        transient_group_1 = TransientGroup(transients[:1])
        transient_group_1.save("data/test_transient_group_1.h5")
        transient_group_10 = TransientGroup(transients[:10])
        transient_group_10.save("data/test_transient_group_10.h5")
        transient_group_100 = TransientGroup(transients)
        transient_group_100.save("data/test_transient_group_100.h5")
        
    def setup(self, n):
        """Return TransientGroup of size n."""
        self.t = TransientGroup.load(f"data/test_transient_group_{n}.h5")
    
    def time_load(self, n):
        """Time transient group load for different numbers of transients."""
        TransientGroup.load(f"data/test_transient_group_{n}.h5")
    
    def time_save(self, n):
        """Time transient group save for different numbers of transients."""
        self.t.save(f"data/transient_group_save_{n}.h5")