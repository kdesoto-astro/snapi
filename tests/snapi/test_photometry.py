"""Test object functions for photometry.py."""
import numpy as np

from snapi.photometry import Photometry

def test_normalize(test_photometry):
    """Test the normalize method of photometry.py."""
    phot_copy = test_photometry.copy()
    phot_copy.normalize()
    assert np.max(phot_copy.detections['flux']) == 1.0

def test_phase(test_photometry):
    """Test the phase method of photometry.py."""
    goal_arr = np.array([-7.5, -2.5, -1.5, -0.5, 0., 0.5, 0.5, 1.5])
    phot_copy2 = test_photometry.copy()
    phot_copy2.phase(2.5) # should also phase by 2.5
    assert np.all(phot_copy2.detections['time'].mjd == goal_arr)
    phot_copy = test_photometry.copy()
    phot_copy.phase() # should phase by 2.5
    assert np.all(phot_copy.detections['time'].mjd == goal_arr)
    # TODO: add test for periodic phasing

def test_add_lightcurve(test_lightcurve1, test_lightcurve2):
    """Test the add_lightcurve method of photometry.py."""
    phot = Photometry()
    phot.add_lightcurve(test_lightcurve1)
    assert len(phot.detections) == 4
    assert len(phot) == 1
    phot.add_lightcurve(test_lightcurve2)
    assert len(phot.detections) == 8
    assert len(phot) == 2
    phot.add_lightcurve(test_lightcurve1) # test merge functionality
    assert len(phot.detections) == 8
    assert len(phot) == 2