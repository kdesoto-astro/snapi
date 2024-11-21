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
    
    params = [1,10]
    
    def setup_cache(self):
        """Set up the TransientGroup."""
        transients = []
        test_spectrometer = Spectrometer(
            instrument="test_spectrometer",
            wavelength_start=4000.0,
            wavelength_delta=1.0,
            num_channels=10,
        )
        test_spectrum = Spectrum(
            time=1.0,
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
        for i in range(10):
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
        transient_group_1.save("test_transient_group_1.h5")
        transient_group_10 = TransientGroup(transients[:10])
        transient_group_10.save("test_transient_group_10.h5")
        #transient_group_100 = TransientGroup(transients)
        #transient_group_100.save("test_transient_group_100.h5")
        
    def setup(self, n):
        """Return TransientGroup of size n."""
        self.t = TransientGroup.load(f"test_transient_group_{n}.h5")
    
    def time_load(self, n):
        """Time transient group load for different numbers of transients."""
        TransientGroup.load(f"test_transient_group_{n}.h5")
    
    def time_save(self, n):
        """Time transient group save for different numbers of transients."""
        self.t.save(f"transient_group_save_{n}.h5")
        
        
class TransientSaveLoad:
    """Benchmark save/load times for a Transient."""
    
    params = [1,10]
    
    def setup_cache(self):
        """Set up the TransientGroup."""
        transients = []
        test_lcs = []
        test_specs = []
        
        for i in range(10):
            test_spectrometer = Spectrometer(
                instrument=f"test_spectrometer_{i}",
                wavelength_start=4000.0,
                wavelength_delta=1.0,
                num_channels=10,
            )
            test_specs.append(
                Spectrum(
                    time=1.0,
                    fluxes=np.linspace(5,10,num=10),
                    errors=np.linspace(0,1,num=10),
                    spectrometer=test_spectrometer,
                )
            )
            
        
            test_filter = Filter(
                band="r",
                instrument=f"ZTF{i}",
                center=6173.23,
            )
            test_lcs.append(
                LightCurve(
                    times=[0.0, 3.0, 2.5, 4.0],
                    mags=[21, 20, 20, 20],
                    mag_errs=[0.3, 0.1, 0.2, 0.5],
                    zpts=[25.0, 25.0, 25.0, 25.0],
                    filt=test_filter,
                    phased=True,
                )
            )
        for n in [1,10]:
            transient_phot = Transient(
                iid=f"test_transient_phot_{n}",
                ra=100,
                dec=30,
                redshift=0.01,
                internal_names={},
                photometry=Photometry(test_lcs[:n]),
            )
            transient_phot.save(f"test_transient_phot_{n}.h5")
            
            transient_spec = Transient(
                iid=f"test_transient_spec_{n}",
                ra=100,
                dec=30,
                redshift=0.01,
                internal_names={},
                spectroscopy=Spectroscopy(test_specs[:n]),
            )
            transient_spec.save(f"test_transient_spec_{n}.h5")
            
            transient_both = Transient(
                iid=f"test_transient_both_{n}",
                ra=100,
                dec=30,
                redshift=0.01,
                internal_names={},
                spectroscopy=Spectroscopy(test_specs[:n]),
                photometry=Photometry(test_lcs[:n]),
            )
            transient_both.save(f"test_transient_both_{n}.h5")
        
    def setup(self, n):
        """Return TransientGroup of size n."""
        self.t_phot = Transient.load(f"test_transient_phot_{n}.h5")
        self.t_spec = Transient.load(f"test_transient_spec_{n}.h5")
        self.t_both = Transient.load(f"test_transient_both_{n}.h5")
    
    def time_phot_load(self, n):
        """Time transient group load for different numbers of transients."""
        Transient.load(f"test_transient_phot_{n}.h5")
    
    def time_phot_save(self, n):
        """Time transient group save for different numbers of transients."""
        self.t_phot.save(f"transient_phot_save_{n}.h5")
        
    def time_spec_load(self, n):
        """Time transient group load for different numbers of transients."""
        results = Transient.load(f"test_transient_spec_{n}.h5")
    
    def time_spec_save(self, n):
        """Time transient group save for different numbers of transients."""
        self.t_spec.save(f"transient_spec_save_{n}.h5")
        
    def time_both_load(self, n):
        """Time transient group load for different numbers of transients."""
        result = Transient.load(f"test_transient_spec_{n}.h5")
    
    def time_both_save(self, n):
        """Time transient group save for different numbers of transients."""
        self.t_both.save(f"transient_both_save_{n}.h5")
        
        
class LightCurveSaveLoad:
    """Benchmark save/load for a light curve as a function of number of observations.
    """
    params = [1,10,100]
    def setup_cache(self):
        """Setup LC"""
        for n_obs in [1,10,100]:
            test_filter = Filter(
                band="r",
                instrument=f"ZTF{n_obs}",
                center=6173.23,
            )
            test_lc = LightCurve(
                times=np.linspace(0.,10.,num=n_obs),
                mags=np.linspace(10.,20.,num=n_obs),
                mag_errs=np.linspace(1.,2.,num=n_obs),
                zpts=[25.,] * n_obs,
                filt=test_filter,
                phased=True,
            )
            test_lc.save(f"test_lc_{n_obs}.h5")
            
    def setup(self, n):
        """Load LC."""
        self.lc = LightCurve.load(f"test_lc_{n}.h5")
            
    def time_save(self, n):
        """Time LC save"""
        self.lc.save(f"save_lc_{n}.h5")
        
    def time_load(self, n):
        """Load LC."""
        LightCurve.load(f"test_lc_{n}.h5")
        
    