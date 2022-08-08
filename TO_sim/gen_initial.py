import numpy as np
import scipy.stats as scs


def Lorentzian(N, mean=0, sigma=1):
    """return theta, omega, Kc"""
    init_theta = np.random.uniform(-np.pi, np.pi, size=N)
    init_omega = scs.cauchy.rvs(mean, sigma, N)
    Kc = 2 / (np.pi * scs.cauchy.pdf(mean, mean, sigma))
    return init_theta, init_omega, Kc


def Normal(N, mean=0, sigma=1):
    """return theta, omega, Kc"""
    init_theta = np.random.uniform(-np.pi, np.pi, size=N)
    init_omega = scs.norm.rvs(mean, sigma, N)
    Kc = 2 / (np.pi * scs.norm.pdf(mean, mean, sigma))
    return init_theta, init_omega, Kc
