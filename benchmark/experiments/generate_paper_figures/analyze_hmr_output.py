from pickle import load

with open("hmr_output.pkl", "rb") as f:
    results = load(f)

import numpy as np
import matplotlib.pyplot as plt

# results = {
#     "configuration": {
#         "W_F": W_F_c,
#         "W_R": W_R_c
#     },
#     "full" : {
#         "W_F": W_F_c,
#         "W_R": W_R_c,
#         "potential_traces": potential_traces
#     }
# }



from benchmark.evaluation import estimate_nonequilibrium_free_energy

W_F, W_R = results["configuration"]["W_F"], results["configuration"]["W_R"]
dF, sq_ddF = estimate_nonequilibrium_free_energy(W_F, W_R)
print("Configuration DeltaF_neq: {:.3f} +/- {:.3f}".format(dF, np.sqrt(sq_ddF)))

W_F, W_R = results["full"]["W_F"], results["full"]["W_R"]
dF, sq_ddF = estimate_nonequilibrium_free_energy(W_F, W_R)
print("Phase-space DeltaF_neq: {:.3f} +/- {:.3f}".format(dF, np.sqrt(sq_ddF)))

potential_traces = results["full"]["potential_traces"]


from statsmodels.tsa.stattools import acf


#print(potential_traces[0])
#plt.figure()
#for p in potential_traces:
#    plt.plot(p)
#plt.show()

plt.figure()
autocorr_functions = []
for p in potential_traces:
    autocorr_function, conf_int = acf(p, unbiased=False, nlags=len(p)-1, alpha=0.05)
    autocorr_functions.append(autocorr_function)
    plt.plot(autocorr_function, linewidth=0.1, color='blue')


plt.plot(np.mean(autocorr_functions, 0), linewidth=2)
plt.show()