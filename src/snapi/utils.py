"""Helper functions for the SNAPI package."""
import os
from typing import Any

import h5py
from astropy.coordinates import SkyCoord
from dustmaps.config import config as dustmaps_config
from dustmaps.sfd import SFDQuery

dustmaps_config["data_dir"] = os.path.join(
    "/".join(os.path.dirname(__file__).split("/")[:-2]), "data", "dustmaps"
)
sfd = SFDQuery()


def list_datasets(hdf5_file: str) -> list[str]:
    """List all datasets in an HDF5 file."""
    with h5py.File(hdf5_file, "r") as file:

        def visitor(name: str, node: Any) -> None:
            if isinstance(node, h5py.Dataset):
                datasets.append(name)

        datasets: list[str] = []
        file.visititems(visitor)
    return datasets


def calc_mwebv(coordinates: SkyCoord) -> float:
    """Calculate the Milky Way E(B-V) at a given set of coordinates."""
    return float(sfd(coordinates))