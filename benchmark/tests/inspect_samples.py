# want to double check that the WaterBox samples have equilibrated volume, for example

from benchmark import DATA_PATH, FIGURE_PATH
import numpy as np
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


# TODO: import automatic equilibration detection scheme from pymbar

filenames = ["flexible_waterbox_samples.npy",
             "waterbox_constrained_samples.npy"]

def compute_volume(x):
    """Compute volume of axis-aligned bounding box"""
    return np.prod(x.max(0) - x.min(0))

for filename in filenames:
    full_path = os.path.join(DATA_PATH, filename)
    if os.path.exists(full_path):
        X = np.load(full_path)
        Vs = [compute_volume(x) for x in tqdm(X)]

        plt.plot(Vs, label=filename)

plt.legend(loc="best", fancybox=True)
plt.savefig(os.path.join(FIGURE_PATH, "volume.png"), dpi=300)
plt.close()
