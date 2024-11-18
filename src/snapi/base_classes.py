from abc import ABC, abstractmethod
import dill
from typing import TypeVar, Optional, Any
import copy
from matplotlib.axes import Axes
from astropy.io.misc import hdf5
import pandas as pd

from .utils import list_datasets, str_to_class


MeasT = TypeVar("MeasT", bound="MeasurementSet")
BaseT = TypeVar("BaseT", bound="Base")

class Base(ABC):
    """Base class which all objects will inherit from.

    Currently empty.
    """

    @abstractmethod
    def __init__(self) -> None:
        self._id: str = ""
        self.associated_objects: dict[str, object] = {}
        self.arr_attrs: list[str] = []
        self.meta_attrs: list[str] = []

    def copy(self: BaseT) -> BaseT:
        """Return a deep copy of the object."""
        return copy.deepcopy(self)
    
    def update(self) -> None:
        """Update steps needed upon modifying child attributes."""
        pass

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
            path = "/" + type(self).__name__
            
        mode = "a" if append else "w"

        # Save DataFrame and attributes to HDF5
        with pd.HDFStore(file_name, mode=mode) as store:  # type: ignore
            store.put(path, pd.Series([]))
            # first store info on array + meta attrs and assoc. objects
            store.put(path + "/arr_attrs", pd.Series(self.arr_attrs))
            store.put(path + "/meta_attrs", pd.Series(self.meta_attrs))
            obj_keys = []
            obj_values = []
            for (k, v) in self.associated_objects.items():
                obj_keys.append(k)
                obj_values.append(v)
            store.put(path + "/assoc_keys", pd.Series(obj_keys))
            store.put(path + "/assoc_types", pd.Series(obj_values))
                    
            for arr_attr in self.arr_attrs:
                attr = getattr(self, arr_attr)
                if isinstance(attr, pd.DataFrame):
                    store.put(path + f"/{arr_attr}", attr)
                else:
                    store.put(path + f"/{arr_attr}", pd.Series(attr))
                                            
        # Save any meta attrs
        with pd.HDFStore(file_name, mode='a') as store:  # type: ignore
            for meta_attr in self.meta_attrs:
                a = getattr(self, meta_attr)
                try:
                    setattr(store.get_storer(path).attrs, meta_attr, a)
                except:
                    a_enc = dill.dumps(self._cols)
                    setattr(store.get_storer(path).attrs, meta_attr, a_enc)

        # Save associated objects
        for assoc_obj in self.associated_objects:
            getattr(self, assoc_obj).save(file_name = file_name, path = path + f"/{assoc_obj}", append=True)

    @classmethod
    def load(
        cls: Any,
        file_name: str,
        path: Optional[str] = None,
    ) -> Any:
        """Load LightCurve from saved HDF5 table. Automatically
        extracts feature information.
        """
        new_obj = cls()

        if path is None:
            path = "/" + cls.__name__
            with pd.HDFStore(file_name) as store:
                try:
                    store[path]
                except:
                    raise ValueError(f"Default path {path} does not exist in file, please manually set path.")
            
        with pd.HDFStore(file_name) as store:
            # unload attributes first
            attr_dict = store.get_storer(path).attrs.__dict__  # type: ignore
            
            # get info about meta, array attributes, and associated objects
            new_obj.arr_attrs = list(store[path+'/arr_attrs'])
            new_obj.meta_attrs = list(store[path + '/meta_attrs'])
            assoc_obj_keys = store[path + '/assoc_keys']
            assoc_obj_types = store[path + '/assoc_types']
            new_obj.associated_objects = {k: t for (k,t) in zip(assoc_obj_keys, assoc_obj_types)}
            
            # extract meta values
            for a_key in new_obj.meta_attrs:
                setattr(new_obj, a_key, attr_dict[a_key])
            for attr_name in new_obj.arr_attrs: # array attribute
                attr = store[f"{path}/{attr_name}"]
                if isinstance(attr, pd.DataFrame):
                    setattr(new_obj, attr_name, attr)
                else:
                    setattr(new_obj, attr_name, attr.to_numpy())
            for attr_name in new_obj.associated_objects: # associated object load
                subtype = str_to_class(new_obj.associated_objects[attr_name])
                setattr(new_obj, attr_name, subtype.load(file_name, f"{path}/{attr_name}"))
                    
            new_obj.update()
            
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
        super().__init__()

class MeasurementSet(Base):
    """Base class for storing collection
    of measurements, potentially from different
    instruments and taken at different times.
    """

    def __init__(self) -> None:
        super().__init__()

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
        super().__init__()
        self._instrument = instrument
        self.meta_attrs.append("_instrument")

    def __eq__(self, value: object) -> bool:
        """Check if two filters are equal."""
        return str(self) == str(value)

    @property
    def instrument(self) -> str:
        """Return instrument of filter."""
        return self._instrument
