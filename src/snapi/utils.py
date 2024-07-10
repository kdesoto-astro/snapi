"""Helper functions for the SNAPI package."""
from typing import Any

import h5py


def list_datasets(hdf5_file: str) -> list[str]:
    """List all datasets in an HDF5 file."""
    with h5py.File(hdf5_file, "r") as file:

        def visitor(name: str, node: Any) -> None:
            if isinstance(node, h5py.Dataset):
                datasets.append(name)

        datasets: list[str] = []
        file.visititems(visitor)
    return datasets
