from abc import ABC, abstractmethod
import os
import time
import dill
from typing import TypeVar, Optional, Any
import copy
from matplotlib.axes import Axes
import pandas as pd
import pyarrow.feather as feather
#import msgpack

from .utils import list_datasets, str_to_class


BaseT = TypeVar("BaseT", bound="Base")

class Base(ABC):
    """Base class which all objects will inherit from.

    Currently empty.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        self._id: str = ""
        self.arr_attrs: list[str] = []
        self.meta_attrs: list[str] = []
        self.associated_objects = None
        
    def _initialize_assoc_objects(self) -> None:
        if self.associated_objects is None:
            self.associated_objects = pd.DataFrame(
                columns=['type',],
                index=pd.Index([], dtype='string'),
                dtype='string'
            )

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
        
    def save(self, file_name: str, append: bool = False) -> None:
        """Save LightCurve object using Apache Arrow.

        Parameters
        ----------
        file_name : str
            Name of file to save (will be treated as a directory).
        path : str
            Path within the Arrow structure to save Measurement.
        append : bool
            Whether to append to existing file (handled through directory structure).
        """

        # Ensure the directory exists
        os.makedirs(file_name, exist_ok=True)
        
        if self.associated_objects is not None:
            feather.write_feather(
                self.associated_objects,
                f"{file_name}/associated_objects.feather",
                compression='uncompressed',
            )

        # Save array attributes
        for arr_attr in self.arr_attrs:
            attr = getattr(self, arr_attr)
            if not isinstance(attr, pd.DataFrame):
                attr = pd.Series(attr).to_frame()
            feather.write_feather(
                attr,
                f"{file_name}/{arr_attr}.feather",
                compression='uncompressed',
            )

        # Save meta attributes using msgpack
        meta_data = {attr: getattr(self, attr) for attr in self.meta_attrs}
        
        if len(self.arr_attrs) > 0:
            meta_data['arr_attrs'] = self.arr_attrs
        if len(self.meta_attrs) > 0:
            meta_data['meta_attrs'] = self.meta_attrs

        with open(f"{file_name}/meta.dill", 'wb') as f:
            dill.dump(meta_data, f)

        # Save associated objects
        if self.associated_objects is not None:
            for assoc_name in self.associated_objects.index:
                getattr(self, assoc_name).save(file_name=f"{file_name}/{assoc_name}", append=True)
            
    @classmethod
    def load(cls: Any, file_name: str) -> Any:
        """Load LightCurve from saved Arrow structure."""
        #t_overall = time.perf_counter()
        new_obj = cls()
        #print(new_obj.__class__.__name__, "init", time.perf_counter() - t_overall)

        try:
            #t1 = time.perf_counter()
            # Extract associated_objects
            if os.path.exists(f"{file_name}/associated_objects.feather"):
                new_obj.associated_objects = feather.read_feather(f"{file_name}/associated_objects.feather")
            else:
                new_obj.associated_objects = None
            #print(new_obj.__class__.__name__, "assoc", time.perf_counter() - t1)

            #t1 = time.perf_counter()
            # Load meta attributes
            with open(f"{file_name}/meta.dill", 'rb') as f:
                meta_data = dill.load(f)

            for attr, value in meta_data.items():
                setattr(new_obj, attr, value)
            #print(new_obj.__class__.__name__, "meta", time.perf_counter() - t1)
            
            #t1 = time.perf_counter()
            for arr_attr in new_obj.arr_attrs:
                attr = feather.read_feather(f"{file_name}/{arr_attr}.feather")
                if attr.ndim == 1:
                    setattr(new_obj, arr_attr, attr.to_numpy())
                else:
                    setattr(new_obj, arr_attr, attr)
            #print(new_obj.__class__.__name__, "arr", time.perf_counter() - t1)
            
            # Load associated objects
            if new_obj.associated_objects is not None:
                for i, obj_row in new_obj.associated_objects.iterrows():
                    #t1 = time.perf_counter()
                    subtype = str_to_class(obj_row['type'])
                    #print(new_obj.__class__.__name__, "subtype", time.perf_counter() - t1)
                    setattr(new_obj, obj_row.name, subtype.load(f"{file_name}/{obj_row.name}"))

            #t1 = time.perf_counter()
            new_obj.update()
            #print(new_obj.__class__.__name__, "update", time.perf_counter() - t1)

        except FileNotFoundError:
            raise ValueError(f"Path {file_name}/{path} does not exist.")
        
        #print(time.perf_counter() - t_overall)
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
