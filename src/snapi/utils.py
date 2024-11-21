"""Helper functions for the SNAPI package."""
import os
from typing import Any, Optional
import sys
import pkgutil
import importlib

import h5py
import numpy as np
from astropy.coordinates import SkyCoord
from dustmaps.config import config as dustmaps_config
from dustmaps.sfd import SFDQuery
from numpy.typing import NDArray

sfd = None

def set_dustmaps_path(path: Optional[str]=None):
    """Configure path for dustmaps. If no path given,
    use the default path for SNAPI.
    """
    if path is None:
        dustmaps_config["data_dir"] = os.path.join(
            "/".join(os.path.dirname(__file__).split("/")[:-2]), "data", "dustmaps"
        )
    else:
        dustmaps_config["data_dir"] = path


def list_datasets(hdf5_file: str) -> NDArray[np.str_]:
    """List all datasets in an HDF5 file."""
    with h5py.File(hdf5_file, "r") as file:

        def visitor(name: str, node: Any) -> None:
            if isinstance(node, h5py.Dataset):
                datasets.append(name)

        datasets: list[str] = []
        file.visititems(visitor)

    datasets_np = np.unique([os.path.dirname(x) for x in datasets])
    return datasets_np


def calc_mwebv(coordinates: SkyCoord) -> float:
    """Calculate the Milky Way E(B-V) at a given set of coordinates."""
    global sfd
    
    if sfd is None:
        sfd = SFDQuery()
        
    return float(sfd(coordinates))

class_cache = {}

def index_classes_in_package(package_name):
    # Traverse the modules in the package
    for importer, modname, ispkg in pkgutil.walk_packages(
            path=__import__(package_name).__path__, prefix=package_name + '.'):
        try:
            module = importlib.import_module(modname)
            for attr_name in dir(module):
                attr_value = getattr(module, attr_name)
                if isinstance(attr_value, type):
                    class_cache[attr_name] = attr_value
        except ImportError:
            # Handle modules that fail to import, if necessary
            pass

def str_to_class(classname):
    if not class_cache:
        # Initialize the cache if it's not done yet
        index_classes_in_package('snapi')
    try:
        return class_cache[classname]
    except KeyError:
        raise ValueError(f"class {classname} not found in the snapi package!")
