import numpy as np

def estimate_nonequilibrium_free_energy(W_shads_F, W_shads_R, N_eff=None):
    """Estimate nonequilibrium free energy (and uncertainty) from work measurements.

    Reference: eqs. A1, A2 in http://threeplusone.com/Sivak2013a.pdf
    """
    if N_eff == None: N_eff = len(W_shads_F)
    DeltaF_neq = 0.5 * (np.mean(W_shads_F) - np.mean(W_shads_R))
    # squared uncertainty = [var(W_0->M) + var(W_M->2M) - 2 cov(W_0->M, W_M->2M)] / (4 N_eff)
    sq_uncertainty = (np.var(W_shads_F) + np.var(W_shads_R) - 2 * np.cov(W_shads_F, W_shads_R)[0,1]) / (4 * N_eff)
    return DeltaF_neq, sq_uncertainty