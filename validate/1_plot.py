import numpy as np
from pickle import load
import matplotlib.pyplot as plt

path = "/Users/joshuafass/Documents/MSKCC/Chodera Lab/standalone_quartic_verification/1_wshad_sanity_check.pkl"
with open(path, "rb") as f:
    results = load(f)

schemes = sorted(list(set([key[0] for key in results.keys()])))
dt_range = sorted(list(set([key[1] for key in results.keys()])))

path = "1_wshad_sanity_check.pkl"
with open(path, "rb") as f:
    results = load(f)

plt.figure(figsize=(6, 4))
for scheme in schemes:
    y = []
    dy = []
    for dt in dt_range:
        exp_neg_work = np.exp(-results[(scheme, dt)])

        y.append(np.mean(exp_neg_work))
        dy.append(1.96 * np.std(exp_neg_work) / np.sqrt(len(exp_neg_work)))

    y = np.array(y)
    dy = np.array(dy)

    # style A: line + translucent shaded regions
    plt.plot(dt_range, y, label=scheme)
    plt.fill_between(dt_range, y - dy, y + dy, alpha=0.4)

    # style B: line with error bars
    # plt.errorbar(dt_range, y, dy, label=scheme)

plt.ylim(0.5, 1.5)
plt.legend(loc="best")
plt.hlines(1, min(dt_range), max(dt_range), linestyles='--')
plt.xlabel(r"$\Delta t$")
plt.ylabel(r"$\widehat{\langle exp(-W_{shad}) \rangle}$")
plt.title(r"Sanity check 1: $\langle exp(-W_{shad}) \rangle = 1$?")
plt.savefig("sanity_check_1.jpg", dpi=300)