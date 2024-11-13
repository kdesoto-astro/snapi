from typing import Iterable, Optional, Callable, Any
from .transient import Transient

class Group(Base):
    """Metaclass to store information about a set of objects.
    """
    def __init__(
        self,
        objs: Optional[Iterable[Base]] = None
    ):
        if objs is not None:
            for t in objs:
                setattr(self, "_"+t.id, t) # will also check uniqueness
                self.associated_objects.append(t.id)
            
        self.update()
        
        self.arr_attrs.append("_meta")
        self.meta_attrs.append("_cols")
        
        
    def update(self):
        """Update meta-dataframe."""
        # keep pandas df for metadata
        self.associated_objects = sorted(self.associated_objects)
        self._meta = pd.DataFrame(
            [],
            columns=self._cols
        )
        self._meta.index.name = 'id'
        
        for t_id in self.associated_objects:
            extracted_dict = self._extract_meta(getattr(self, t_id))
            self._meta.append(extracted_dict)
        
    def __get__(self, obj_id: str):
        return getattr(self, "_"+obj_id).copy()
    
    def __set__(self, obj_id: str, obj: Base):
        extracted_dict = self._extract_meta(obj)
        if not hasattr(self, "_"+obj_id):
            self._meta.append(extracted_dict)
            self._meta.sort_index(inplace=True)
        else:
            self._meta.loc[obj_id,:] = pd.Series(extracted_dict)
            
        setattr(self, "_"+obj_id, obj)
 
    def __iter__(self):
        """Iterates through transients."""
        for obj_id in self.associated_objects:
            yield self[obj_id]
            
    def add_col(self, col_name: str, attribute: str, attr_manipulation):
        """Add column to meta dataframe. Takes in name for column and
        function to apply to the object to retrieve the attribute.
        """
        if col_name in self._cols:
            raise ValueError("column name already in metadata")
        self._cols[col_name] = (attribute, attr_manipulation)
        self.update()
        
    def _extract_meta(self, obj):
        """Extract metadata from object.
        Returns dictionary.
        """
        meta_dict = {'id': obj.id}
        for c, v in self._cols.items():
            altered_attr = v(obj)
            meta_dict[c] = altered_attr
        return meta_dict
    
    @property
    def metadata(self):
        """Return metadata DataFrame.
        """
        return self._meta.copy()
    
    def filter(self, ids: Iterable[str], inplace: bool=False) -> Any:
        """Return copy of object but only with ids in 'ids'.
        """
        if inplace:
            for obj_id in self.associated_objects:
                if obj_id not in ids:
                    attr = getattr(self, "_"+obj_id)
                    del attr
                    self.associated_objects.remove(obj_id)
                    
            self.update()
            return self
            
        else:
            group = self.__class__()
            for obj_id in self.associated_objects:
                if (obj_id in ids) and hasattr(self, "_"+obj_id):
                    group[obj_id] = getattr(self, "_"+obj_id)
                    
        return group
                
            

class TransientGroup(Base):
    """Stores information about set of transient objects, with pointers to individual objects."""
    def __init__(
        self,
        transients: Optional[Iterable[Transient]] = None,
        col_defs: Optional[dict[str, Callable]] = None
    ) -> None:
        
        super().__init__(transients)
            
        if col_defs is None:
            # DEFAULT COLS
            self._cols = {
                'ra': lambda x: x.ra,
                'dec': lambda x: x.dec,
                'internal_names': lambda x: ', '.join(x.internal_names),
                'spec_class': lambda x: x.spec_class,
                'redshift': lambda x: x.redshift,
            }
        else:
            self._cols = col_defs
            
    @classmethod
    def from_directory(cls, dir_path: str, names: Optional[Iterable[str]] = None):
        """Imports transient list from directory. If names is provided, only
        grabs subset from within names.
        """
        all_fns = glob.glob(
            os.path.join(dir_path, "*.h5")
        )
        for fn in all_fns:
            t = Transient.load(fn)
            if (names is None) or (t.id in names):
                if hasattr(self, t.id):
                    continue
                self[t.id] = t
                
        self.update()
        
    def add_binary_class(self, target_label: str, class_attr: str = 'spec_class'):
        """Convert spec_class to a binary classification
        problem."""
        self._cols[f"binary_class_{target_label}"] = lambda x: getattr(x, class_attr) == target_label
        
    def canonicalize_classes(self, canonicalize_func: Callable, class_attr: str = 'spec_class'):
        """Convert labels to canon labels.
        """
        self._cols[f"canonical_class"] = lambda x: canonicalize_func(getattr(x, class_attr))

        
class SamplerResultGroup(Base):
    """Container for multiple SamplerResult objects
    that extracts + organizes metadata and performs
    group-level data augmentation.
    """
    def __init__(
        self,
        sampler_results: Optional[Iterable[SamplerResult]] = None
        param_names: Optional[list[str]] = None
    ) -> None:
        
        super().__init__(sampler_results)
        self._cols = {}
        if param_names is None and (sampler_results is not None):
            # union of all fits
            for sr in sampler_results:
                for fp in sr.fit_parameters:
                    if fp in self_cols:
                        continue
                    self._cols[f"{fp}_median"] = lambda x: np.nanmedian(x.fit_parameters[fp])
        else:
            for fp in param_names:
                self._cols[f"{fp}_median"] = lambda x: np.nanmedian(x.fit_parameters[fp])
                
        self._cols['score'] = lambda x: x.score
        self._cols['sampler'] = lambda x: x.sampler
        
        
    @classmethod
    def from_directory(cls, dir_path: str, names: Optional[Iterable[str]] = None):
        """Imports transient list from directory. If names is provided, only
        grabs subset from within names.
        """
        all_fns = glob.glob(
            os.path.join(dir_path, "*.h5")
        )
        for fn in all_fns:
            t = SamplerResult.load(fn)
            if (names is None) or (t.id in names):
                if hasattr(self, t.id):
                    continue
                self[t.id] = t
                
        self.update()
        
        
    def set_samples_per_event(self, num_samples: int):
        """Set the number of samples to keep per event.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be greater than zero")
        
        for sr_id in self.associated_objects:
            new_sr = self[sr_id]
            new_sr.fit_parameters = new_sr.fit_parameters.iloc[:num_samples,:]
            self[sr_id] = new_sr
            
            
    def balance_classes(self, classes: dict[str, str], majority_samples=1):
        """Balance classes by keeping more samples for
        rarer classes.
        
        Parameters
        ----------
        classes: dict[str, str]
            Mapping from id to class
            
        Returns
        -------
        tuple of np.ndarray
            Tuple containing oversampled features and labels.
        """
        classes_filtered = {c: v for (c,v) in classes if c in self.associated_objects}
        labels_unique, counts = np.unique(
            classes_filtered.values(), return_counts=True
        )
        samples_per_class = {
            l: majority_samples * round(max(counts) / c) for (l, c) in zip(labels_unique, counts)
        }
        
        for sr_id in self.associated_objects:
            new_sr = self[sr_id]
            sr_class = classes[sr_id]
            num_samples = samples_per_class[sr_class]
            
            new_sr.fit_parameters = new_sr.fit_parameters.iloc[:num_samples,:]
            self[sr_id] = new_sr
    
    
    def oversample_smote(self, classes: dict[str, str]):
        """
        Uses SMOTE to oversample data from rarer classes.
        Returns features and labels.
        """
        oversample = SMOTE()
        classes_arr = self._meta.index.map(classes)
        med_cols = [x for x in self._cols if x[:-6] == 'median']
        features_smote, labels_smote = oversample.fit_resample(
            self._meta.loc[:, med_cols],
            classes_arr
        )
        return features_smote, labels_smote
    
    @property
    def all_samples(self) -> pd.DataFrame:
        """Return all samples from all objects as a dataframe."""
        
        combined_df = None
        meta_cols = [x for x in self._cols if x[:-6] != 'median']
        
        for sr_id in self.associated_objects:
            sr = self[sr_id]
            df = sr.fit_parameters
            for m in meta_cols:
                df[m] = self._meta[sr_id, m]
            df.set_index(sr_id, inplace=True)
            if combined_df is None:
                combined_df = df
                df.index.name = 'id'
            else:
                combined_df = pd.concat([combined_df, df])
        
        return combined_df
            
        
    
                