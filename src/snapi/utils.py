"""Helper functions for the SNAPI package."""
import os
from typing import Any
import sys

import h5py
import numpy as np
from astropy.coordinates import SkyCoord
from dustmaps.config import config as dustmaps_config
from dustmaps.sfd import SFDQuery
from numpy.typing import NDArray

dustmaps_config["data_dir"] = os.path.join(
    "/".join(os.path.dirname(__file__).split("/")[:-2]), "data", "dustmaps"
)
sfd = SFDQuery()


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
    return float(sfd(coordinates))

def str_to_class(classname):
    for module_name, module in sys.modules.items():
        if module_name.startswith('snapi.'):
            try:
                class_obj = getattr(module, classname)
                return class_obj
            except AttributeError:
                # The module doesn't have the class, so continue searching
                pass
    raise ValueError(f"class {classname} not found in the snapi package!")