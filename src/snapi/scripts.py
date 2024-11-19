# Scripts that bridge the transition between classes
from typing import Optional
from functools import partial
import copy
import multiprocessing

from .groups import TransientGroup, SamplerResultGroup
from .analysis import Sampler, SamplerResult

def single_transient_fit(transient, sampler, priors):
    """Fit function for a single Transient object."""
    sampler_copy = copy.deepcopy(sampler)
    sampler_copy.fit_photometry(transient.photometry)
    return sampler_copy.result
    
def fit_transient_group(
    transient_group: TransientGroup,
    sampler: Sampler,
    parallelize: bool = True,
    n_parallel: Optional[int] = None,
    checkpoint_fn: Optional[str] = None,
    checkpoint_freq: int = 100,
) -> SamplerResultGroup
    """Fit all transients in a group by the same Sampler
    with the same SamplerPrior. If parallelize, run
    simultaneously across n_parallel cores. If checkpoint_fn,
    save checkpointed SamplerResultGroup every checkpoint_freq
    fits.
    """
    single_transient_fit_static = partial(
        single_transient_fit,
        sampler=sampler,
    )
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
        pool = multiprocessing.Pool(n_parallel)
        
        if checkpoint_fn:
            batches = [transients[i::checkpoint_freq] for i in range(checkpoint_freq)]
            for b in batches:
                results = pool.map(single_transient_fit_static, b)
                sampler_results.extend(results)
                SamplerResultGroup(sampler_results).save(checkpoint_fn)
                
        else:
            results = pool.map(single_transient_fit_static, transients)
            sampler_results.extend(results)
            
    else:
        for i, t in enumerate(transients):
            sampler_results.append(single_transient_fit_static(t))
            if checkpoint_fn and ((i+1) % checkpoint_freq == 0):
                print(f"Checkpointing...{i+1} of {len(transients)}")
                SamplerResultGroup(sampler_results).save(checkpoint_fn)
                
    return sampler_results
        
        