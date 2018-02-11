"""Collects the results of compare_near_eq_and_exact_parallel_over_outer_loop.py
which generates a single .pkl file per outer-loop sample.
"""

from pickle import load
from glob import glob
from simtk import unit
from tqdm import tqdm
import numpy as np
from benchmark.evaluation.compare_near_eq_and_exact import estimate_from_work_samples

def experiment_tuple_to_key(experiment):
    """One of the experiment descriptors is unit'd -- strip unit."""
    (scheme, dt, marginal, testsystem) = experiment
    return (scheme, dt / unit.femtosecond, marginal, testsystem)

def key_to_experiment_tuple(key):
    """Re-add unit"""
    (scheme, dt, marginal, testsystem) = key
    return (scheme, dt * unit.femtosecond, marginal, testsystem)

fnames = glob('*.pkl')

print('reading all files')
results = dict()
for fname in tqdm(fnames):
    with open(fname, 'rb') as f:
        l = load(f)

        (experiment, result) = l
        Ws = list(result['Ws'])

        key = experiment_tuple_to_key(experiment)
        if key not in results:
            results[key] = Ws
        else:
            results[key].extend(Ws)

print(results.keys())

from pickle import dump
print('saving all summaries')
for i, key in enumerate(tqdm(results.keys())):
    Ws = np.array(results[key])
    new_estimate = np.mean([estimate_from_work_samples(w) for w in Ws])
    result = {'Ws': Ws, 'new_estimate': new_estimate}

    with open('summary/summary_{}.pkl'.format(i), 'wb') as f:
        dump((key_to_experiment_tuple(key), result), f)