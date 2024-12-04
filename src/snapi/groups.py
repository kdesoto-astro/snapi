from typing import Iterable, Optional, Callable, Any
import glob
import os

import pandas as pd
import numpy as np

from .analysis.sampler import SamplerResult
from .base_classes import Base
from .formatter import Formatter
from .transient import Transient

class Group(Base):
    """Metaclass to store information about a set of objects.
    """
    def __init__(
        self,
        objs: Optional[Iterable[Base]] = None
    ):
        super().__init__()
        self._initialize_assoc_objects()
                
        if objs is not None:
            for t in objs:
                if hasattr(self, "_"+t.id):
                    continue
                setattr(self, "_"+t.id, t) # will also check uniqueness
                self.associated_objects.loc["_"+t.id] = {'type': t.__class__.__name__}
            
        self.update()
        
        #self.arr_attrs.append("_meta")
        self.meta_attrs.append("_cols")
        
        
    def update(self):
        """Update meta-dataframe."""    
        # keep pandas df for metadata        
        extracted_dicts = []
        for t_id in self.associated_objects.index:
            extracted_dicts.append(self._extract_meta(getattr(self, t_id)))
            
        if len(extracted_dicts) == 0:
            self._meta = pd.DataFrame([], columns=['id', *self._cols.keys()])
        else:
            self._meta = pd.DataFrame(extracted_dicts)
            
        self._meta.set_index('id', inplace=True)
        self._meta.sort_index(inplace=True)
        
    def __getitem__(self, obj_id: str):
        return getattr(self, "_"+obj_id).copy()
    
    def __setitem__(self, obj_id: str, obj: Base):
        extracted_dict = self._extract_meta(obj)
        if not hasattr(self, "_"+obj_id):
            extracted_df = pd.DataFrame(extracted_dict)
            extracted_df.set_index('id', inplace=True)
            self._meta = pd.concat([self._meta, extracted_df])
            self._meta.sort_index(inplace=True)
        else:
            self._meta.loc[obj_id,:] = pd.Series(extracted_dict)
            
        setattr(self, "_"+obj_id, obj)
        
    def __len__(self):
        return len(self.associated_objects)
 
    def __iter__(self):
        """Iterates through transients."""
        for obj_id in self.associated_objects.index:
            yield self[obj_id[1:]]
            
    def add_col(self, col_name: str, attr_manipulation):
        """Add column to meta dataframe. Takes in name for column and
        function to apply to the object to retrieve the attribute.
        """
        if col_name in self._cols:
            raise ValueError("column name already in metadata")
        self._cols[col_name] = attr_manipulation
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
            for obj_id in self.associated_objects.index:
                if obj_id[1:] not in ids:
                    attr = getattr(self, obj_id)
                    del attr
                    self.associated_objects.drop(index=obj_id, inplace=True)
                    
            self.update()
            return self
            
        else:
            group = self.__class__()
            for obj_id in self.associated_objects.index:
                if (obj_id[1:] in ids) and hasattr(self, obj_id):
                    group_obj = getattr(self, obj_id)
                    setattr(group, obj_id, group_obj) # will also check uniqueness
                    group.associated_objects.loc[obj_id] = {'type': group_obj.__class__.__name__}
                    
        group.update()
                    
        return group
                
            

class TransientGroup(Group):
    """Stores information about set of transient objects, with pointers to individual objects."""
    def __init__(
        self,
        transients: Optional[Iterable[Transient]] = None,
        col_defs: Optional[dict[str, Callable]] = None
    ) -> None:
            
        if col_defs is None:
            # DEFAULT COLS
            self._cols = {
                'ra': lambda x: x._ra,
                'dec': lambda x: x._dec,
                'internal_names': lambda x: ', '.join(x.internal_names),
                'spec_class': lambda x: x.spec_class,
                'redshift': lambda x: x.redshift,
            }
        else:
            self._cols = col_defs
            
        super().__init__(transients)
                        
    @classmethod
    def from_directory(cls, dir_path: str, names: Optional[Iterable[str]] = None):
        """Imports transient list from directory. If names is provided, only
        grabs subset from within names.
        """
        all_fns = glob.glob(
            os.path.join(dir_path, "*/")
        )

        new_obj = cls()
        for i, fn in enumerate(all_fns):
            if i % 250 == 0:
                print(f"Added transient {i} out of {len(all_fns)}")
            try:
                t = Transient.load(fn)
            except:
                print(f"{fn.split('/')[-1]} skipped: unable to load")
                continue
            if (names is None) or (t.id in names):
                if hasattr(new_obj, "_"+t.id):
                    continue
                setattr(new_obj, "_"+t.id, t) # will also check uniqueness
                new_obj.associated_objects.loc["_"+t.id] = {'type': Transient.__name__}
                                
        new_obj.update()
        return new_obj
        
    def add_binary_class(self, target_label: str, class_attr: str = 'spec_class'):
        """Convert spec_class to a binary classification
        problem."""
        self.add_col(f"binary_class_{target_label}", lambda x: getattr(x, class_attr) == target_label)
        
    def canonicalize_classes(self, canonicalize_func: Callable, class_attr: str = 'spec_class'):
        """Convert labels to canon labels.
        """
        self.add_col("canonical_class", lambda x: canonicalize_func(getattr(x, class_attr)))

        
class SamplerResultGroup(Group):
    """Container for multiple SamplerResult objects
    that extracts + organizes metadata and performs
    group-level data augmentation.
    """
    def __init__(
        self,
        sampler_results: Optional[Iterable[SamplerResult]] = None,
        param_names: Optional[list[str]] = None
    ) -> None:
        
        self._cols = {}
            
        if param_names is None and (sampler_results is not None):
            # union of all fits
            for sr in sampler_results:
                for fp in sr.fit_parameters:
                    if fp in self._cols:
                        continue
                    self._cols[f"{fp}_median"] = lambda x, col=fp: x.fit_parameters[col].dropna().median()
                    
        elif param_names is not None:
            for fp in param_names:
                self._cols[f"{fp}_median"] = lambda x, col=fp: x.fit_parameters[col].dropna().median()
                
        self._cols['score_median'] = lambda x: np.nanmedian(x.score)
        self._cols['sampler'] = lambda x: x.sampler
        
        super().__init__(sampler_results)
        
    @classmethod
    def from_directory(cls, dir_path: str, names: Optional[Iterable[str]] = None):
        """Imports transient list from directory. If names is provided, only
        grabs subset from within names.
        """
        new_obj = cls()
        all_fns = glob.glob(
            os.path.join(dir_path, "*")
        )
        for fn in all_fns:
            t = SamplerResult.load(fn)
            if hasattr(new_obj, "_"+t.id):
                continue
            setattr(new_obj, "_"+t.id, t) # will also check uniqueness
            new_obj.associated_objects["_"+t.id] = SamplerResult.__name__
                
        new_obj.update()
        return new_obj
        
        
    def set_samples_per_event(self, num_samples: int):
        """Set the number of samples to keep per event.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be greater than zero")
        
        for sr_id in self.associated_objects.index:
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
        classes_filtered = {c: v for (c,v) in classes.items() if "_"+c in self.associated_objects.index}
        labels_unique, counts = np.unique(
            list(classes_filtered.values()), return_counts=True
        )
        samples_per_class = {
            l: majority_samples * round(max(counts) / c) for (l, c) in zip(labels_unique, counts)
        }
        
        for sr_id in self.associated_objects.index:
            new_sr = self[sr_id[1:]]
            sr_class = classes_filtered[sr_id[1:]]
            num_samples = samples_per_class[sr_class]
            
            new_sr.fit_parameters = new_sr.fit_parameters.iloc[:num_samples,:]
            new_sr.score = new_sr.score[:num_samples]
            self[sr_id[1:]] = new_sr
    
    
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
        
        for sr_id in self.associated_objects.index:
            sr = self[sr_id[1:]]
            df = sr.fit_parameters
            for m in meta_cols:
                df[m] = self._meta.loc[sr_id[1:], m]
            df['score'] = sr.score
            df['id'] = sr_id[1:]
            df.set_index('id', inplace=True)
            if combined_df is None:
                combined_df = df
            else:
                combined_df = pd.concat([combined_df, df])
        
        return combined_df
    
    def umap(self, ax, classes: Optional[pd.DataFrame] = None, formatter=Formatter()):
        """Plot 2D UMAP of sampling features.
        """
        import umap
        import umap.plot

        features = self.all_samples
        if classes:
            classes_ordered = classes[features.index]
        features = features.to_numpy()
        
        # add jitter
        for i in range(features.shape[1]):
            features[:,i] += np.random.normal(scale=np.std(features) / 1e3, size=len(features))
        nan_features = np.any(np.isnan(features), axis=1)
        
        mapper = umap.UMAP().fit(features[~nan_features], force_all_finite=False)
        
        if classes:
            ax = umap.plot.points(
                mapper,
                labels=classes_ordered[~nan_features],
                color_key_cmap=formatter.categorical_cmap,
                ax=ax
            )
        else:
            ax = umap.plot.points(
                mapper,
                cmap=formatter.cmap,
                ax=ax
            )
        return ax


    def pacmap(self, ax, classes: Optional[pd.DataFrame] = None, formatter=Formatter()):
        """Plot 2D PACMAP of sampling features.
        """
        import pacmap
        
        features = self.all_samples
        if classes:
            classes_ordered = classes[features.index]
        features = features.to_numpy()
        
        # add jitter
        for i in range(features.shape[1]):
            features[:,i] += np.random.normal(scale=np.nanstd(features) / 100, size=len(features))
        nan_features = np.any(np.isnan(features), axis=1)
        
        embedding = pacmap.PaCMAP(n_components=2)
        X_transformed = embedding.fit_transform(features[~nan_features], init="pca")
        
        if classes:
            labels = classes_ordered[~nan_features]
        else:
            labels = np.array(["none",] * len(X_transformed))
        for l in np.unique(labels):
            # visualize the embedding
            ax.scatter(
                X_transformed[labels == l, 0],
                X_transformed[labels == l, 1],
                s=formatter.marker_size,
                color=formatter.edge_color,
                marker=formatter.marker_style,
                alpha=0.3,
                label=l
            )
            formatter.rotate_colors()
            formatter.rotate_markers()
            
        formatter.reset_colors()
        formatter.reset_markers()
        
        return ax


    
                