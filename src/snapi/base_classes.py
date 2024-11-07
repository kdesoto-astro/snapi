from abc import ABC, abstractmethod
from typing import TypeVar, Optional, Any
import copy
from matplotlib.axes import Axes
from astropy.io.misc import hdf5
import pandas as pd

from .utils import list_datasets


MeasT = TypeVar("MeasT", bound="MeasurementSet")
BaseT = TypeVar("BaseT", bound="Base")

class Base(ABC):
    """Base class which all objects will inherit from.

    Currently empty.
    """
    _class_str = "base"
    _arr_attrs: set[str] = set()
    _meta_attrs: set[str] = set()
    _associated_objects: dict[str, object] = {}

    @abstractmethod
    def __init__(self) -> None:
        self._id: str = ""

    def copy(self: BaseT) -> BaseT:
        """Return a deep copy of the object."""
        return copy.deepcopy(self)

    @property
    def id(self) -> str:
        """Object identifier."""
        return self._id

    @id.setter
    def id(self, iid: object) -> None:
        try:
            hash(iid)
        except Exception as exc:
            raise ValueError(f"Input {iid} is not hashable!") from exc
        try:
            self._id = str(iid)
        except Exception as exc:
            raise ValueError(f"Input {iid} could not be casted to a string!") from exc
        
    def save(self, file_name: str, path: Optional[str] = None, append: bool = False) -> None:
        """Save LightCurve object as an HDF5 file.

        Parameters
        ----------
        file_name : str
            Name of file to save.
        path : str
            HDF5 path to save Measurement.
        append : bool
            Whether to append to existing file.
        """
        if path is None:
            path = "/" + self._class_str

        mode = "a" if append else "w"

        # Save DataFrame and attributes to HDF5
        with pd.HDFStore(file_name, mode=mode) as store:  # type: ignore
            for arr_attr in self._arr_attrs:
                store.put(path + f"/{arr_attr}", getattr(self, arr_attr))

        # Save any meta attrs
        with pd.HDFStore(file_name, mode=mode) as store:  # type: ignore
            for meta_attr in self._meta_attrs:
                setattr(store.get_storer(path).attrs, "instrument", getattr(self, meta_attr))

        # Save associated objects
        for assoc_obj in self._associated_objects:
            getattr(self, assoc_obj).save(file_name = file_name, path = path + f"/{assoc_obj}", append = True)

    @classmethod
    def load(
        cls: Any,
        file_name: str,
        path: Optional[str] = None,
        archival: bool = False,
    ) -> Any:
        """Load LightCurve from saved HDF5 table. Automatically
        extracts feature information.
        """
        new_obj = cls()

        if path is None:
            paths = list_datasets(file_name, archival)
            if len(paths) > 1:
                raise ValueError("Multiple datasets found in file. Please specify path.")
            path = paths[0]

        with pd.HDFStore(file_name) as store:
            for arr_a in cls._arr_attrs:
                setattr(new_obj, arr_a, store[path+f"/{arr_a}"])
            for assoc_obj in cls._associated_objects: #TODO: make many-to-one compatible
                setattr(new_obj, assoc_obj, cls._associated_objects[assoc_obj])
            
            return new_obj


class Plottable(Base):
    """Class for objects that can be plotted."""

    @abstractmethod
    def plot(self, ax: Axes) -> Axes:
        """Adds plot of object in-place to
        'ax' object. Returns ax.
        """
        pass


class Measurement(Base):
    """Base class for storing single measurement
    modality, such as a spectrum or light curve."""

    def __init__(self) -> None:
        self._observer = None
        self._arr_attrs = set()

    

class MeasurementSet(Base):
    """Base class for storing collection
    of measurements, potentially from different
    instruments and taken at different times.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def filter_by_instrument(self: MeasT, instrument: str) -> MeasT:
        """Return MeasurementSet with only measurements
        from instrument named 'instrument.'
        """
        pass


class Observer(Base):
    """Class that holds observing facility information."""

    def __init__(
        self,
        instrument: str,
    ) -> None:
        self._instrument = instrument

    def __eq__(self, value: object) -> bool:
        """Check if two filters are equal."""
        return str(self) == str(value)

    @property
    def instrument(self) -> str:
        """Return instrument of filter."""
        return self._instrument
