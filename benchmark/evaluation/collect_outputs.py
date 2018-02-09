"""Collects the results of compare_near_eq_and_exact_parallel_over_outer_loop.py
which generates a single .pkl file per outer-loop sample.
"""

from pickle import load
from glob import glob
from simtk import unit

def experiment_tuple_to_key(experiment):
    (scheme, dt, marginal, testsystem) = experiment
    return (scheme, dt / unit.femtosecond, marginal, testsystem)

def key_to_experiment_tuple(key):
    (scheme, dt, marginal, testsystem) = key
    return (scheme, dt * unit.femtosecond, marginal, testsystem)

fnames = glob('*.pkl')

results = dict()
for fname in fnames:
    with open(fname, 'rb') as f:

        (experiment, result) = load(f)

        key = experiment_tuple_to_key(experiment)
        if key not in results:
            results[key] = result
        else:
            results[key].append(result)

print(results.keys())

from pickle import dump

for i, key in enumerate(results.keys()):
    with open('summary_{}.pkl'.format(i), 'wb') as f:
        dump((key_to_experiment_tuple(key), results[key]), f)