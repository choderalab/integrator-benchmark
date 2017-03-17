# Estimate entropy
import numpy as np
from sklearn.neighbors import KDTree
from scipy.special import digamma

def get_nearest_neighbor_distances(X, k=1):
    """For each point in X, get the distance to its k-th nearest neighbor"""
    tree = KDTree(X, p=np.inf)
    dist, ind = tree.query(X, k=k+1)
    return dist[:,-1]

def estimate_entropy(X, k=1):
    """Estimate the differential entropy, using the distribution of k-nearest-neighbor
    distances

    Reference:
        https://github.com/gregversteeg/NPEET/blob/master/entropy_estimators.py"""
    if len(X.shape) == 1:
        X = X.reshape((len(X), 1))
    n, dimension = X.shape
    const = digamma(n) - digamma(k) + dimension * np.log(2)
    nearest_neighbor_distances = get_nearest_neighbor_distances(X, k)
    eps = np.random.rand(n) * 1e-10
    return (const + dimension * np.mean(np.log(nearest_neighbor_distances + eps)))

def estimate_marginal_entropies(X, k=1):
    """Estimate the entropy of each marginal of X."""
    marginal_entropies = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        marginal_entropies[i] = estimate_entropy(X[:,i], k)
    return marginal_entropies

def get_nearest_neighbor_distances_by_sorting(x):
    """1D is easy because we can sort:
    sort the array, then for each point, get its distance to the points immediately before
    or after it. Take the minimum of these two...
    """

    sorted_x = np.array(sorted(x))

    # distance from x_i to x_{i+1} for i in [1, N-1]
    distance_to_next = np.abs(sorted_x[2:] - sorted_x[1:-1])

    # distance from x_i to x_{i-1} for i in [1, N-1]
    distance_to_previous = np.abs(sorted_x[1:-1] - sorted_x[:-2])

    return np.minimum(distance_to_next, distance_to_previous)