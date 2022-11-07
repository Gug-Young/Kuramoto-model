import numpy as np
import scipy.stats as scs
from scipy.special import erfinv

def _make_init_theta_(seed,N):
    if type(seed)==str:
        init_theta= np.linspace(-np.pi,np.pi,N,endpoint=False)
    else:
        np.random.seed(seed)
        init_theta = np.random.uniform(-np.pi, np.pi, size=N)
    return init_theta

def Lorentzian(N, mean=0, sigma=1,seed=None):
    """return theta, omega, Kc"""
    init_theta = _make_init_theta_(seed,N)
    if type(seed)==str:seed=None
    else: seed=seed
    init_omega = scs.cauchy.rvs(mean, sigma, N,random_state = seed)
    Kc = 2 / (np.pi * scs.cauchy.pdf(mean, mean, sigma))
    return init_theta, init_omega, Kc

def Quantile_Lorentzian(N, mean=0, sigma=1,seed=None):
    """return theta, omega, Kc"""
    init_theta = _make_init_theta_(seed,N)
    init_omega = np.array([sigma*np.tan(np.pi/2 * (2*i - N - 1)/(N+1)) + mean for i in range(1,N+1)])
    Kc = 2 / (np.pi * scs.cauchy.pdf(mean, mean, sigma))
    return init_theta, init_omega, Kc

def Normal(N, mean=0, sigma=1,seed=None):
    """return theta, omega, Kc"""
    init_theta = _make_init_theta_(seed,N)
    if type(seed)==str:seed=None
    else: seed=seed
    init_omega = scs.norm.rvs(mean, sigma, N,random_state = seed)
    Kc = 2 / (np.pi * scs.norm.pdf(mean, mean, sigma))
    return init_theta, init_omega, Kc

def Quantile_Normal(N, mean=0, sigma=1,seed=None):
    """return theta, omega, Kc"""
    init_theta = _make_init_theta_(seed,N)
    init_omega = np.array([mean +sigma*(2**0.5)*erfinv((2*i - N - 1)/(N+1)) for i in range(1,N+1)])
    Kc = 2 / (np.pi * scs.cauchy.pdf(mean, mean, sigma))
    return init_theta, init_omega, Kc

def Lorentzian(N, mean=0, sigma=1,seed=None):
    """return theta, omega, Kc"""
    init_theta = _make_init_theta_(seed,N)
    if type(seed)==str:seed=None
    else: seed=seed
    init_omega = scs.cauchy.rvs(mean, sigma, N,random_state = seed)
    Kc = 2 / (np.pi * scs.cauchy.pdf(mean, mean, sigma))
    return init_theta, init_omega, Kc