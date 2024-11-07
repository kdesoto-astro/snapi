"""Helper functions for the SNAPI package."""
import os
from typing import Any

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


def list_datasets(hdf5_file: str, archival: bool=False) -> NDArray[np.str_]:
    """List all datasets in an HDF5 file."""
    with h5py.File(hdf5_file, "r") as file:

        def visitor(name: str, node: Any) -> None:
            if isinstance(node, h5py.Dataset):
                datasets.append(name)

        datasets: list[str] = []
        file.visititems(visitor)

    if archival:
        datasets_np = np.array([x for x in datasets if x.split(".")[-1] != "__table_column_meta__"])
    else:
        datasets_np = np.unique([os.path.dirname(x) for x in datasets])
    return datasets_np


def calc_mwebv(coordinates: SkyCoord) -> float:
    """Calculate the Milky Way E(B-V) at a given set of coordinates."""
    return float(sfd(coordinates))
