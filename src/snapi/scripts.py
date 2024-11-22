# Scripts that bridge the transition between classes
from typing import Optional
from functools import partial
import copy
import multiprocess as mp
#import dill
#from pathos.multiprocessing import ProcessPool
import os
import numpy as np

from .groups import TransientGroup, SamplerResultGroup
from .analysis import Sampler, SamplerResult
from .photometry import Photometry

"""
__STATE = _ProcessPool__STATE = _CustomPool__STATE = {}

class CustomPool(ProcessPool):
    def __init__(self, *args, **kwds):
        self.__nodes = kwds.pop('ncpus', cpu_count())
        super().__init__(*args, **kwds)
        
    def _serve(self, nodes=None): #XXX: should be STATE method; use id
        
        if nodes is None: nodes = self.__nodes
        _pool = __STATE.get(self._id, None)
        if not _pool or nodes != _pool.__nodes or self._kwds != _pool._kwds:
            self._clear()
            ctx = mp.get_context('spawn')
            _pool = ctx.Pool(nodes, **self._kwds)
            _pool.__nodes = nodes
            _pool._kwds = self._kwds
            __STATE[self._id] = _pool
        return _pool
"""
    
def single_worker_fit(transient_batch, sampler, pad):
    """Single worker's script to run in parallel."""
    sampler_results = []
    for t in transient_batch:
        print(t)
        sampler_results.append(
            single_transient_fit(t, sampler, pad)
        )
    print("worker done")
    return sampler_results
        
    
def single_transient_fit(transient, sampler, pad):
    """Fit function for a single Transient object."""
    sampler.reset() # play it safe
    if pad:
        padded_lcs = set()
        fill = {'phase': 1000., 'flux': 0.1, 'flux_error': 1000., 'zeropoint': 23.90, 'upper_limit': False}

        padded_lcs = []
        orig_size = len(transient.photometry.detections)
        num_pad = int(2**np.ceil(np.log2(orig_size)))
        for lc in transient.photometry.light_curves:
            padded_lc=lc.pad(fill, num_pad - len(lc.detections))
            padded_lcs.append(padded_lc)
            
        padded_photometry = Photometry.from_light_curves(padded_lcs)
        sampler.fit_photometry(padded_photometry, orig_num_times=orig_size)
    else:
        sampler.fit_photometry(transient.photometry)
        
    r = sampler.result
    r.id = transient.id
        
    return r

def fit_transient_group(
    transient_group: TransientGroup,
    sampler: Sampler,
    parallelize: bool = True,
    n_parallel: Optional[int] = None,
    checkpoint_fn: Optional[str] = None,
    checkpoint_freq: int = 100,
    pad: bool=False,
) -> SamplerResultGroup:
    """Fit all transients in a group by the same Sampler
    with the same SamplerPrior. If parallelize, run
    simultaneously across n_parallel cores. If checkpoint_fn,
    save checkpointed SamplerResultGroup every checkpoint_freq
    fits.
    """
    if checkpoint_fn and os.path.exists(checkpoint_fn): # first try loading checkpoint
        sr_group = SamplerResultGroup.load(checkpoint_fn)
        sampler_results = [x for x in sr_group]
        # ignore transients already sampled
        sampled_names = sr_group.metadata.index
        transients = [x for x in transient_group if x.id not in sampled_names]
        
    else:
        sampler_results = []
        transients = [x for x in transient_group]
    
    if parallelize:
        single_worker_fit_static = partial(
            single_worker_fit, pad=pad
        )
        samplers = [copy.deepcopy(sampler) for i in range(n_parallel)]
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(n_parallel)
        
        if checkpoint_fn:
            checkpoint_batches = [transients[i::checkpoint_freq] for i in range(checkpoint_freq)]
            for i, cb in enumerate(checkpoint_batches):
                transient_batches = [cb[i::n_parallel] for i in range(n_parallel)]
                result = pool.starmap(single_worker_fit_static, zip(transient_batches, samplers))
                flattened_result = list(itertools.chain(*result))
                sampler_results.extend(flattened_result)
                SamplerResultGroup(sampler_results).save(checkpoint_fn)
                print(f"Finished checkpoint {i} of {len(checkpoint_batches)}")
                
        else:
            transient_batches = [transients[i::n_parallel] for i in range(n_parallel)]
            results = pool.starmap(single_transient_fit_static, zip(transient_batches, samplers))
            sampler_results.extend(results)
            
    else:
        single_transient_fit_static = partial(
            single_transient_fit,
            pad=pad,
        )
        for i, t in enumerate(transients):
            sampler_results.append(single_transient_fit_static(t, sampler))
            if checkpoint_fn and ((i+1) % checkpoint_freq == 0):
                print(f"Checkpointing...{i+1} of {len(transients)}")
                SamplerResultGroup(sampler_results).save(checkpoint_fn)
                
    return sampler_results
        
        