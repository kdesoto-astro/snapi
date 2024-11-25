from abc import ABC, abstractmethod
import dill
import time
from typing import TypeVar, Optional, Any
import copy
from matplotlib.axes import Axes
from astropy.io.misc import hdf5
import pandas as pd

from .utils import list_datasets, str_to_class


BaseT = TypeVar("BaseT", bound="Base")

class Base(ABC):
    """Base class which all objects will inherit from.

    Currently empty.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        self._id: str = ""
        self.associated_objects: pd.DataFrame = pd.DataFrame([], columns=['type',], dtype=str)
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
            store.put(path, self.associated_objects)
                    
            for arr_attr in self.arr_attrs:
                attr = getattr(self, arr_attr)
                if isinstance(attr, pd.DataFrame):
                    store.put(f"{path}/{arr_attr}", attr)
                else:
                    store.put(f"{path}/{arr_attr}", pd.Series(attr))
                                            
        # Save any meta attrs
        with pd.HDFStore(file_name, mode='a') as store:  # type: ignore
            attrs = store.get_storer(path).attrs
            for meta_attr in self.meta_attrs:
                a = getattr(self, meta_attr)
                try:
                    setattr(attrs, meta_attr, a)
                except:
                    a_enc = dill.dumps(self._cols)
                    setattr(attrs, meta_attr, a_enc)
                    
            # store attributes
            setattr(attrs, 'arr_attrs', self.arr_attrs)
            setattr(attrs, 'meta_attrs', self.meta_attrs)
            

        # Save associated objects
        for assoc_name in self.associated_objects.index:
            getattr(self, assoc_name).save(file_name = file_name, path = path + f"/{assoc_name}", append=True)

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
            new_obj.associated_objects = store[path]
            # unload attributes first
            attr_dict = store.get_storer(path).attrs.__dict__  # type: ignore
            
            # get info about meta, array attributes, and associated objects
            new_obj.arr_attrs = attr_dict['arr_attrs']
            new_obj.meta_attrs = attr_dict['meta_attrs']
            
            # extract meta values
            for a_key in new_obj.meta_attrs:
                setattr(new_obj, a_key, attr_dict[a_key])
            for attr_name in new_obj.arr_attrs: # array attribute
                attr = store[f"{path}/{attr_name}"]
                if isinstance(attr, pd.DataFrame):
                    setattr(new_obj, attr_name, attr)
                else:
                    setattr(new_obj, attr_name, attr.to_numpy())
            for i, obj_row in new_obj.associated_objects.iterrows(): # associated object load
                subtype = str_to_class(obj_row['type'])
                setattr(new_obj, obj_row.name, subtype.load(file_name, f"{path}/{obj_row.name}"))
            new_obj.update()
            
            return new_obj


class Plottable(Base):
    """Class for objects that can be plotted."""
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def plot(self, ax: Axes) -> Axes:
        """Adds plot of object in-place to
        'ax' object. Returns ax.
        """
        pass


class Measurement(Base):
    """Base class for storing single measurement
    modality, such as a spectrum or light curve."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def _validate_observer(self, observer):
        """Validate associated observer."""
        if (observer is not None) and (not isinstance(observer, Observer)):
            raise TypeError("filt must be None or an Observer subclass object!")
        self._observer = observer
        if self._observer is not None:
            self.associated_objects['_observer'] = observer.__class__.__name__


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
