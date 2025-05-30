# Scripts that bridge the transition between classes
import os

#os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
# Set JAX to use CPU
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
# Disable GPU visibility
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# Set TensorFlow logging (which affects some JAX logging)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from typing import Optional
from functools import partial
import itertools
import copy
import multiprocess as mp
import logging
import numpy as np
from jax import config, device_count

from .groups import TransientGroup, SamplerResultGroup
from .analysis import Sampler
from .photometry import Photometry
    
config.update('jax_platform_name', 'cpu')
#print(device_count('cpu'))

import numpyro
numpyro.enable_validation()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(processName)s - %(message)s',
    level=logging.INFO
)

def single_worker_fit(transient_batch, sampler, pad):
    """Single worker's script to run in parallel."""
    sampler_results = []
    for t in transient_batch:
        sampler_results.append(
            single_transient_fit(t, sampler, pad)
        )
    logging.info("Worker done")
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
    if r is None:
        return None
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
    overwrite: bool=False
) -> SamplerResultGroup:
    """Fit all transients in a group by the same Sampler
    with the same SamplerPrior. If parallelize, run
    simultaneously across n_parallel cores. If checkpoint_fn,
    save checkpointed SamplerResultGroup every checkpoint_freq
    fits.
    """
    if (not overwrite) and checkpoint_fn and os.path.exists(checkpoint_fn): # first try loading checkpoint
        sr_group = SamplerResultGroup.load(checkpoint_fn)
        sampler_results = [x for x in sr_group if x.id in transient_group.metadata.index]
        # ignore transients already sampled
        sampled_names = sr_group.metadata.index
        transients = [x for x in transient_group if x.id not in sampled_names]
        
    else:
        sampler_results = []
        transients = [x for x in transient_group]
    
    if parallelize:
        mp.log_to_stderr(logging.INFO)
        single_worker_fit_static = partial(
            single_worker_fit, pad=pad
        )
        samplers = [copy.deepcopy(sampler) for i in range(n_parallel)]
        ctx = mp.get_context('spawn')
        
        if checkpoint_fn:
            num_checkpoints = len(transients) // checkpoint_freq
            checkpoint_batches = [transients[i::num_checkpoints] for i in range(num_checkpoints)]
            for i, cb in enumerate(checkpoint_batches):
                transient_batches = [cb[i::n_parallel] for i in range(n_parallel)]
                
                with ctx.Pool(n_parallel) as pool:
                    result = pool.starmap(single_worker_fit_static, zip(transient_batches, samplers))
                    
                flattened_result = list(itertools.chain(*result))
                sampler_results.extend(flattened_result)
                sampler_results = [x for x in sampler_results if x is not None]
                SamplerResultGroup(sampler_results).save(checkpoint_fn)
                print(f"Finished checkpoint {i+1} of {num_checkpoints}")
                
        else:
            transient_batches = [transients[i::n_parallel] for i in range(n_parallel)]
            results = pool.starmap(single_transient_fit_static, zip(transient_batches, samplers))
            sampler_results.extend(results)
            sampler_results = [x for x in sampler_results if x is not None]
            
    else:
        single_transient_fit_static = partial(
            single_transient_fit,
            pad=pad,
        )
        for i, t in enumerate(transients):
            sampler_results.append(single_transient_fit_static(t, sampler))
            if checkpoint_fn and ((i+1) % checkpoint_freq == 0):
                print(f"Checkpointing...{i+1} of {len(transients)}")
                sampler_results = [x for x in sampler_results if x is not None]
                SamplerResultGroup(sampler_results).save(checkpoint_fn)
                
    sampler_results = [x for x in sampler_results if x is not None]
    return sampler_results


def fit_many_hierarchical(
    transient_group: TransientGroup,
    num_events: int,
    sampler: Sampler,
    pad: bool=False,
):
    """Use hierarchical Bayesian sampling to jointly fit many transients.
    """
    all_ids = [t.id for t in transient_group][-1*num_events:]

    if pad:
        all_phot_lens = [len(t.photometry.detections) for t in transient_group]
        num_pad = np.max(all_phot_lens)

        all_padded_phots = []
        orig_sizes = []

        for i, t in enumerate(transient_group):
            if i >= num_events:
                break
            padded_lcs = set()
            fill = {'phase': 1000., 'flux': 0.1, 'flux_error': 1000., 'zeropoint': 23.90, 'upper_limit': False}

            padded_lcs = []
            orig_size = len(t.photometry.detections)
            for lc in t.photometry.light_curves:
                padded_lc=lc.pad(fill, num_pad - len(lc.detections))
                padded_lcs.append(padded_lc)
                
            padded_photometry = Photometry.from_light_curves(padded_lcs)
            all_padded_phots.append(padded_photometry)
            orig_sizes.append(orig_size)
    else:
        all_padded_phots = [t.photometry for t in transient_group][-1*num_events:]
        orig_sizes = [len(phot.detections) for phot in all_padded_phots]
        print(np.max(orig_sizes))

    print("Starting hierarchical fit")
    sampler.fit_photometry_hierarchical(all_padded_phots, orig_num_times=orig_sizes)
    results = sampler.result

    indiv_results = results[2:]
    prior_results = results[0:2]

    for i, r in enumerate(indiv_results):
        t_id = all_ids[i]
        r.id = t_id

    return prior_results, indiv_results



        
        