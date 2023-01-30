from TO_sim.Sol_Kuramoto import *
from TO_sim.gen_Distribution import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
from TO_sim.Private import MY_slack_sender

def Hysteresis(
    m, N=500, K_span=(0.1, 12.5), dK=0.2, dt=0.1, t_end=1000, dist="Quantile Lorentzian"
):
    """_summary_

    Args:
        m (float): mass of oscillator.
        N (int, optional): Number of oscillator. Defaults to 500.
        K_span (tuple, optional): Simulation K start and end. Defaults to (0.2, 12.5).
        dK (float, optional): Step to increse or decrease K. Defaults to 0.2.
        dt (float, optional): Time step. Defaults to 0.1.
        t_end (int, optional): End of simulation time. Defaults to 1000.
        dist (str, optional): Type of distribution for omega(Lorentzian, Normal, Quantile or Random sampled ). Defaults to "Quantile Lorentzian".

    Returns:
        Ks : Simulationed K list
        t_dic : simulationed t dictionary(Forward)
        rs_dic : simulationed r dictionary(Forward)
        t_r_dic : simulationed t dictionary(Backward)
        rs_r_dic : simulationed r dictionary(Backward)
    """
    K_start, K_end = K_span
    Ks = np.arange(K_start, K_end + dK, dK)
    dtheta_init = np.zeros(N)
    if dist == "Normal":
        theta_init, omega_init, Kc = Normal(N, 0, 1, seed=0)
    elif dist == "Lorentzian":
        theta_init, omega_init, Kc = Lorentzian(N, 0, 1, seed=0)
    elif dist == "Quantile Lorentzian":
        theta_init, omega_init, Kc = Quantile_Lorentzian(N, 0, 1, seed=0)
    elif dist == "Quantile Normal":
        theta_init, omega_init, Kc = Quantile_Normal(0, 1, N)

    rs_dic = {}
    t_dic = {}

    num = 0
    for K in Ks:
        if num == 0:
            theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=theta_init,
                p_dtheta=dtheta_init,
                p_omega=omega_init,
                distribution="Normal",
            )
        else:
            theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=theta_s[-1],
                p_dtheta=dtheta_s[-1],
                p_omega=omega,
                distribution="Normal",
            )
        rs_dic[Ks[num]] = rs
        t_dic[num] = t + num * t_end
        num += 1

    theta_r_init, dtheta_r_init = theta_s[-1], dtheta_s[-1]

    rs_r_dic = {}
    t_r_dic = {}
    dKr = -dK
    Ksr = np.arange(K_end, K_start + dKr, dKr)

    num_r = 0
    for K in Ksr:
        if num_r == 0:
            theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=theta_r_init,
                p_dtheta=dtheta_r_init,
                p_omega=omega_init,
                distribution=dist,
            )
        else:
            theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=theta_s[-1],
                p_dtheta=dtheta_s[-1],
                p_omega=omega,
                distribution=dist,
            )
        rs_r_dic[Ksr[num_r]] = rs
        t_r_dic[num_r] = t + num_r * t_end
        num_r += 1

    return Ks, t_dic, rs_dic, t_r_dic, rs_r_dic

@MY_slack_sender
def Hysteresis_pd(
    m, N=500, K_span=(0.1, 12.5), dK=0.2, dt=0.1, t_end=1000, dist="Quantile Lorentzian",seed=None, shuffle = False, shuffle_seed = None
):
    r"""_summary_

    Args:
        m (float): mass of oscillator.
        N (int, optional): Number of oscillator. Defaults to 500.
        K_span (tuple, optional): Simulation K start and end. Defaults to (0.2, 12.5).
        dK (float, optional): Step to increse or decrease K. Defaults to 0.2.
        dt (float, optional): Time step. Defaults to 0.1.
        t_end (int, optional): End of simulation time. Defaults to 1000.
        dist (str, optional): Type of distribution for omega(Lorentzian, Normal, Quantile or Random sampled ). Defaults to "Quantile Lorentzian".
        seed (float,optional): To make random distribution of $\theta$ and $\omega$, 'uniform' make uniform distribution of $\theta$
        shuffle (bool,optional): To shuffle phsae distribtuion, defult False
        shuffle_seed (float,optional): To shuffle phsae distribtuion, default None
    Returns:
        Ksdf : (Forward)Simulationed data(pandas data frame), order parameter r, times, theta, dtheta and omega
        Ksrdf : (Backward)simulationed data(pandas data frame), order parameter r, times, theta, dtheta and omega
    """
    K_start, K_end = K_span
    Ks = np.round(np.arange(K_start, K_end + dK/2, dK),2)
    dtheta_init = np.zeros(N)
    if dist == "Normal":
        theta_init, omega_init, Kc = Normal(N, 0, 1, seed=seed)
    elif dist == "Lorentzian":
        theta_init, omega_init, Kc = Lorentzian(N, 0, 1, seed=seed)
    elif dist == "Quantile Lorentzian":
        theta_init, omega_init, Kc = Quantile_Lorentzian(N, 0, 1, seed=seed)
    elif dist == "Quantile Normal":
        theta_init, omega_init, Kc = Quantile_Normal(N, mean=0, sigma=1,seed=seed)
    Ksdf = pd.DataFrame({"Omega":Ks,"theta_s":Ks,"dtheta_s":Ks,"rs":Ks,"ts":Ks},index=Ks,dtype=object)
    num = 0
    if shuffle:
        np.random.seed(shuffle_seed)
        np.random.shuffle(theta_init)
    for K in tqdm(Ks):
        if num == 0:
            theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=theta_init,
                p_dtheta=dtheta_init,
                p_omega=omega_init,
                distribution="Normal",
            )
        else:
            theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=theta_s[-1],
                p_dtheta=dtheta_s[-1],
                p_omega=omega,
                distribution="Normal",
            )
        Ksdf["rs"][K] = rs
        Ksdf["theta_s"][K] = theta_s
        Ksdf["dtheta_s"][K] = dtheta_s
        Ksdf["Omega"][K] = omega
        Ksdf["ts"][K] = t + num * t_end
        num += 1

    theta_r_init, dtheta_r_init = theta_s[-1], dtheta_s[-1]

    dKr = -dK
    Ksr = np.round(np.arange(K_end, K_start + dKr/2, dKr),2)
    Ksrdf = pd.DataFrame({"Omega":Ksr,"theta_s":Ksr,"dtheta_s":Ksr,"rs":Ksr,"ts":Ksr},index=Ksr,dtype=object)
    num_r = 0
    for K in tqdm(Ksr):
        if num_r == 0:
            theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=theta_r_init,
                p_dtheta=dtheta_r_init,
                p_omega=omega_init,
                distribution=dist,
            )
        else:
            theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=theta_s[-1],
                p_dtheta=dtheta_s[-1],
                p_omega=omega,
                distribution=dist,
            )
        Ksrdf["rs"][K] = rs
        Ksrdf["theta_s"][K] = theta_s
        Ksrdf["dtheta_s"][K] = dtheta_s
        Ksrdf["Omega"][K] = omega
        Ksrdf["ts"][K] = t + num_r * t_end

        num_r += 1
    if m==0:
        diff_osc = lambda x,dt: np.diff(x/dt,axis=0)
        Ksdf["dtheta_s"] = Ksdf["theta_s"].apply(diff_osc,dt=dt)
        Ksrdf["dtheta_s"] = Ksrdf["theta_s"].apply(diff_osc,dt=dt)
    return Ksdf,Ksrdf


def Hysteresis_pd_without_aram(
    m, N=500, K_span=(0.1, 12.5), dK=0.2, dt=0.1, t_end=1000, dist="Quantile Lorentzian",seed=None, shuffle = False, shuffle_seed = None
):
    r"""_summary_

    Args:
        m (float): mass of oscillator.
        N (int, optional): Number of oscillator. Defaults to 500.
        K_span (tuple, optional): Simulation K start and end. Defaults to (0.2, 12.5).
        dK (float, optional): Step to increse or decrease K. Defaults to 0.2.
        dt (float, optional): Time step. Defaults to 0.1.
        t_end (int, optional): End of simulation time. Defaults to 1000.
        dist (str, optional): Type of distribution for omega(Lorentzian, Normal, Quantile or Random sampled ). Defaults to "Quantile Lorentzian".
        seed (float,optional): To make random distribution of $\theta$ and $\omega$, 'uniform' make uniform distribution of $\theta$
        shuffle (bool,optional): To shuffle phsae distribtuion, defult False
        shuffle_seed (float,optional): To shuffle phsae distribtuion, default None
    Returns:
        Ksdf : (Forward)Simulationed data(pandas data frame), order parameter r, times, theta, dtheta and omega
        Ksrdf : (Backward)simulationed data(pandas data frame), order parameter r, times, theta, dtheta and omega
    """
    K_start, K_end = K_span
    Ks = np.round(np.arange(K_start, K_end + dK/2, dK),2)
    dtheta_init = np.zeros(N)
    if dist == "Normal":
        theta_init, omega_init, Kc = Normal(N, 0, 1, seed=seed)
    elif dist == "Lorentzian":
        theta_init, omega_init, Kc = Lorentzian(N, 0, 1, seed=seed)
    elif dist == "Quantile Lorentzian":
        theta_init, omega_init, Kc = Quantile_Lorentzian(N, 0, 1, seed=seed)
    elif dist == "Quantile Normal":
        theta_init, omega_init, Kc = Quantile_Normal(N, mean=0, sigma=1,seed=seed)
    Ksdf = pd.DataFrame({"Omega":Ks,"theta_s":Ks,"dtheta_s":Ks,"rs":Ks,"ts":Ks},index=Ks,dtype=object)
    num = 0
    omega_init = np.sort(omega_init)
    if shuffle:
        np.random.seed(shuffle_seed)
        np.random.shuffle(theta_init)
    for K in tqdm(Ks):
        if num == 0:
            theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=theta_init,
                p_dtheta=dtheta_init,
                p_omega=omega_init,
                distribution="Normal",
            )
        else:
            theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=theta_s[-1],
                p_dtheta=dtheta_s[-1],
                p_omega=omega,
                distribution="Normal",
            )
        Ksdf["rs"][K] = rs
        Ksdf["theta_s"][K] = theta_s
        Ksdf["dtheta_s"][K] = dtheta_s
        Ksdf["Omega"][K] = omega
        Ksdf["ts"][K] = t + num * t_end
        num += 1

    theta_r_init, dtheta_r_init = theta_s[-1], dtheta_s[-1]

    dKr = -dK
    Ksr = np.round(np.arange(K_end, K_start + dKr/2, dKr),2)
    Ksrdf = pd.DataFrame({"Omega":Ksr,"theta_s":Ksr,"dtheta_s":Ksr,"rs":Ksr,"ts":Ksr},index=Ksr,dtype=object)
    num_r = 0
    for K in tqdm(Ksr):
        if num_r == 0:
            theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=theta_r_init,
                p_dtheta=dtheta_r_init,
                p_omega=omega_init,
                distribution=dist,
            )
        else:
            theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=theta_s[-1],
                p_dtheta=dtheta_s[-1],
                p_omega=omega,
                distribution=dist,
            )
        Ksrdf["rs"][K] = rs
        Ksrdf["theta_s"][K] = theta_s
        Ksrdf["dtheta_s"][K] = dtheta_s
        Ksrdf["Omega"][K] = omega
        Ksrdf["ts"][K] = t + num_r * t_end

        num_r += 1
    if m==0:
        diff_osc = lambda x,dt: np.diff(x/dt,axis=0)
        Ksdf["dtheta_s"] = Ksdf["theta_s"].apply(diff_osc,dt=dt)
        Ksrdf["dtheta_s"] = Ksrdf["theta_s"].apply(diff_osc,dt=dt)
    return Ksdf,Ksrdf

def Hysteresis_pd_perturbation(
    m, N=500, K_span=(0.1, 12.5), dK=0.2, dt=0.1, t_end=1000, dist="Quantile Lorentzian",seed=None, shuffle = False, shuffle_seed = None
):
    r"""_summary_

    Args:
        m (float): mass of oscillator.
        N (int, optional): Number of oscillator. Defaults to 500.
        K_span (tuple, optional): Simulation K start and end. Defaults to (0.2, 12.5).
        dK (float, optional): Step to increse or decrease K. Defaults to 0.2.
        dt (float, optional): Time step. Defaults to 0.1.
        t_end (int, optional): End of simulation time. Defaults to 1000.
        dist (str, optional): Type of distribution for omega(Lorentzian, Normal, Quantile or Random sampled ). Defaults to "Quantile Lorentzian".
        seed (float,optional): To make random distribution of $\theta$ and $\omega$, 'uniform' make uniform distribution of $\theta$
        shuffle (bool,optional): To shuffle phsae distribtuion, defult False
        shuffle_seed (float,optional): To shuffle phsae distribtuion, default None
    Returns:
        Ksdf : (Forward)Simulationed data(pandas data frame), order parameter r, times, theta, dtheta and omega
        Ksrdf : (Backward)simulationed data(pandas data frame), order parameter r, times, theta, dtheta and omega
    """
    K_start, K_end = K_span
    Ks = np.round(np.arange(K_start, K_end + dK/2, dK),2)
    
    dtheta_init = np.zeros(N)
    if dist == "Normal":
        theta_init, omega_init, Kc = Normal(N, 0, 1, seed=seed)
    elif dist == "Lorentzian":
        theta_init, omega_init, Kc = Lorentzian(N, 0, 1, seed=seed)
    elif dist == "Quantile Lorentzian":
        theta_init, omega_init, Kc = Quantile_Lorentzian(N, 0, 1, seed=seed)
    elif dist == "Quantile Normal":
        theta_init, omega_init, Kc = Quantile_Normal(N, mean=0, sigma=1,seeds=seed)
    elif dist == "Identical":
        theta_init, omega_init, Kc = Identical(N, 0 , seed=seed)
    Ksdf = pd.DataFrame({"Omega":Ks,"theta_s":Ks,"dtheta_s":Ks,"rs":Ks,"ts":Ks},index=Ks,dtype=object)
    num = 0
    omega_init = np.sort(omega_init)
    pert_index = np.where(abs(omega_init)<=0.3)
    theta_init[pert_index] = 0
    if shuffle:
        np.random.seed(shuffle_seed)
        np.random.shuffle(theta_init)
    for K in tqdm(Ks):
        
        if num == 0:
            theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=theta_init,
                p_dtheta=dtheta_init,
                p_omega=omega_init,
                distribution="Normal",
            )
        else:
            theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=Next_theta,
                p_dtheta=dtheta_s[-1],
                p_omega=omega,
                distribution="Normal",
            )
        Next_theta = theta_s[-1]
        Next_theta[pert_index] = 0
        Ksdf["rs"][K] = rs
        Ksdf["theta_s"][K] = theta_s
        Ksdf["dtheta_s"][K] = dtheta_s
        Ksdf["Omega"][K] = omega
        Ksdf["ts"][K] = t + num * t_end
        num += 1

    theta_r_init, dtheta_r_init = theta_s[-1], dtheta_s[-1]

    dKr = -dK
    Ksr = np.round(np.arange(K_end, K_start + dKr/2, dKr),2)
    Ksrdf = pd.DataFrame({"Omega":Ksr,"theta_s":Ksr,"dtheta_s":Ksr,"rs":Ksr,"ts":Ksr},index=Ksr,dtype=object)
    num_r = 0
    for K in tqdm(Ksr):
        if num_r == 0:
            theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=theta_r_init,
                p_dtheta=dtheta_r_init,
                p_omega=omega_init,
                distribution=dist,
            )
        else:
            theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=theta_s[-1],
                p_dtheta=dtheta_s[-1],
                p_omega=omega,
                distribution=dist,
            )
        Ksrdf["rs"][K] = rs
        Ksrdf["theta_s"][K] = theta_s
        Ksrdf["dtheta_s"][K] = dtheta_s
        Ksrdf["Omega"][K] = omega
        Ksrdf["ts"][K] = t + num_r * t_end

        num_r += 1
    if m==0:
        diff_osc = lambda x,dt: np.diff(x/dt,axis=0)
        Ksdf["dtheta_s"] = Ksdf["theta_s"].apply(diff_osc,dt=dt)
        Ksrdf["dtheta_s"] = Ksrdf["theta_s"].apply(diff_osc,dt=dt)
    return Ksdf,Ksrdf


def Hysteresis_pd_sim(
    m, N=500, K_span=(0.1, 12.5), dK=0.2, dt=0.1, t_end=1000, dist="Quantile Lorentzian",seed=None,
):
    r"""_summary_

    Args:
        m (float): mass of oscillator.
        N (int, optional): Number of oscillator. Defaults to 500.
        K_span (tuple, optional): Simulation K start and end. Defaults to (0.2, 12.5).
        dK (float, optional): Step to increse or decrease K. Defaults to 0.2.
        dt (float, optional): Time step. Defaults to 0.1.
        t_end (int, optional): End of simulation time. Defaults to 1000.
        dist (str, optional): Type of distribution for omega(Lorentzian, Normal, Quantile or Random sampled ). Defaults to "Quantile Lorentzian".
        seed (float,optional): To make random distribution of $\theta$ and $\omega$, 'uniform' make uniform distribution of $\theta$
        shuffle (bool,optional): To shuffle phsae distribtuion, defult False
        shuffle_seed (float,optional): To shuffle phsae distribtuion, default None
    Returns:
        Ksdf : (Forward)Simulationed data(pandas data frame), order parameter r, times, theta, dtheta and omega
        Ksrdf : (Backward)simulationed data(pandas data frame), order parameter r, times, theta, dtheta and omega
    """
    K_start, K_end = K_span
    Ks = np.round(np.arange(K_start, K_end + dK/2, dK),2)
    dtheta_init = np.zeros(N)
    if dist == "Normal":
        theta_init, omega_init, Kc = Normal(N, 0, 1, seed=seed)
    elif dist == "Lorentzian":
        theta_init, omega_init, Kc = Lorentzian(N, 0, 1, seed=seed)
    elif dist == "Quantile Lorentzian":
        theta_init, omega_init, Kc = Quantile_Lorentzian(N, 0, 1, seed=seed)
    elif dist == "Quantile Normal":
        theta_init, omega_init, Kc = Quantile_Normal(N, mean=0, sigma=1,seed=seed)
    Ksdf = pd.DataFrame({"rs":Ks},index=Ks,dtype=object)
    num = 0
    for K in tqdm(Ks):
        if num == 0:
            theta_s, dtheta_s, _, rs, _ = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=theta_init,
                p_dtheta=dtheta_init,
                p_omega=omega_init,
                distribution="Normal",
            )
        else:
            theta_s, dtheta_s, _, rs, _ = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=theta_s[-1],
                p_dtheta=dtheta_s[-1],
                p_omega=omega_init,
                distribution="Normal",
            )
        Ksdf["rs"][K] = rs
        num += 1

    theta_r_init, dtheta_r_init = theta_s[-1], dtheta_s[-1]

    dKr = -dK
    Ksr = np.round(np.arange(K_end, K_start + dKr/2, dKr),2)
    Ksrdf = pd.DataFrame({"rs":Ksr,"ts":Ksr},index=Ksr,dtype=object)
    num_r = 0
    for K in tqdm(Ksr):
        if num_r == 0:
            theta_s, dtheta_s, _, rs, _ = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=theta_r_init,
                p_dtheta=dtheta_r_init,
                p_omega=omega_init,
                distribution=dist,
            )
        else:
            theta_s, dtheta_s, _, rs, _ = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=theta_s[-1],
                p_dtheta=dtheta_s[-1],
                p_omega=omega_init,
                distribution=dist,
            )
        Ksrdf["rs"][K] = rs
        # Ksrdf["ts"][K] = t + num_r * t_end

        num_r += 1
    return Ksdf,Ksrdf


def Hysteresis_pd_init_pvel(
    m, N=500, K_span=(0.1, 12.5), dK=0.2, dt=0.1, t_end=1000, dist="Quantile Lorentzian",seed=None, shuffle = False, shuffle_seed = None, Init_dtheta=True,Init_dtheta_seed = None
):
    r"""_summary_

    Args:
        m (float): mass of oscillator.
        N (int, optional): Number of oscillator. Defaults to 500.
        K_span (tuple, optional): Simulation K start and end. Defaults to (0.2, 12.5).
        dK (float, optional): Step to increse or decrease K. Defaults to 0.2.
        dt (float, optional): Time step. Defaults to 0.1.
        t_end (int, optional): End of simulation time. Defaults to 1000.
        dist (str, optional): Type of distribution for omega(Lorentzian, Normal, Quantile or Random sampled ). Defaults to "Quantile Lorentzian".
        seed (float,optional): To make random distribution of $\theta$ and $\omega$, 'uniform' make uniform distribution of $\theta$
        shuffle (bool,optional): To shuffle phsae distribtuion, defult False
        shuffle_seed (float,optional): To shuffle phsae distribtuion, default None
        Init_dtheta (bool,optional): Give initial phase velocity by omega distribution
    Returns:
        Ksdf : (Forward)Simulationed data(pandas data frame), order parameter r, times, theta, dtheta and omega
        Ksrdf : (Backward)simulationed data(pandas data frame), order parameter r, times, theta, dtheta and omega
    """
    K_start, K_end = K_span
    Ks = np.round(np.arange(K_start, K_end + dK/2, dK),2)
    dtheta_init = np.zeros(N)
    if dist == "Normal":
        theta_init, omega_init, Kc = Normal(N, 0, 1, seed=seed)
    elif dist == "Lorentzian":
        theta_init, omega_init, Kc = Lorentzian(N, 0, 1, seed=seed)
    elif dist == "Quantile Lorentzian":
        theta_init, omega_init, Kc = Quantile_Lorentzian(N, 0, 1, seed=seed)
    elif dist == "Quantile Normal":
        theta_init, omega_init, Kc = Quantile_Normal(N, mean=0, sigma=1,seed=seed)
    Ksdf = pd.DataFrame({"Omega":Ks,"theta_s":Ks,"dtheta_s":Ks,"rs":Ks,"ts":Ks},index=Ks,dtype=object)
    num = 0
    omega_init = np.sort(omega_init)
    if Init_dtheta:

        dtheta_init = scs.cauchy.rvs(0, 1, N,random_state = Init_dtheta_seed)
        dtheta_init = scs.cauchy.rvs(0, 1, N,random_state = Init_dtheta_seed)
    if shuffle:
        np.random.seed(shuffle_seed)
        np.random.shuffle(theta_init)
    for K in Ks:
        if num == 0:
            theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=theta_init,
                p_dtheta=dtheta_init,
                p_omega=omega_init,
                distribution="Normal",
            )
        else:
            theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=theta_s[-1],
                p_dtheta=dtheta_s[-1],
                p_omega=omega,
                distribution="Normal",
            )
        Ksdf["rs"][K] = rs
        Ksdf["theta_s"][K] = theta_s
        Ksdf["dtheta_s"][K] = dtheta_s
        Ksdf["Omega"][K] = omega
        Ksdf["ts"][K] = t + num * t_end
        num += 1

    theta_r_init, dtheta_r_init = theta_s[-1], dtheta_s[-1]

    dKr = -dK
    Ksr = np.round(np.arange(K_end, K_start + dKr/2, dKr),2)
    Ksrdf = pd.DataFrame({"Omega":Ksr,"theta_s":Ksr,"dtheta_s":Ksr,"rs":Ksr,"ts":Ksr},index=Ksr,dtype=object)
    num_r = 0
    for K in Ksr:
        if num_r == 0:
            theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=theta_r_init,
                p_dtheta=dtheta_r_init,
                p_omega=omega_init,
                distribution=dist,
            )
        else:
            theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=theta_s[-1],
                p_dtheta=dtheta_s[-1],
                p_omega=omega,
                distribution=dist,
            )
        Ksrdf["rs"][K] = rs
        Ksrdf["theta_s"][K] = theta_s
        Ksrdf["dtheta_s"][K] = dtheta_s
        Ksrdf["Omega"][K] = omega
        Ksrdf["ts"][K] = t + num_r * t_end

        num_r += 1
    if m==0:
        diff_osc = lambda x,dt: np.diff(x/dt,axis=0)
        Ksdf["dtheta_s"] = Ksdf["theta_s"].apply(diff_osc,dt=dt)
        Ksrdf["dtheta_s"] = Ksrdf["theta_s"].apply(diff_osc,dt=dt)
    return Ksdf,Ksrdf




def Hysteresis_pd_init_pvel_sim(
    m, N=500, K_span=(0.1, 12.5), dK=0.2, dt=0.1, t_end=1000, dist="Quantile Lorentzian",seed=None, shuffle = False, shuffle_seed = None, Init_dtheta=True,Init_dtheta_seed = None
):
    r"""_summary_

    Args:
        m (float): mass of oscillator.
        N (int, optional): Number of oscillator. Defaults to 500.
        K_span (tuple, optional): Simulation K start and end. Defaults to (0.2, 12.5).
        dK (float, optional): Step to increse or decrease K. Defaults to 0.2.
        dt (float, optional): Time step. Defaults to 0.1.
        t_end (int, optional): End of simulation time. Defaults to 1000.
        dist (str, optional): Type of distribution for omega(Lorentzian, Normal, Quantile or Random sampled ). Defaults to "Quantile Lorentzian".
        seed (float,optional): To make random distribution of $\theta$ and $\omega$, 'uniform' make uniform distribution of $\theta$
        shuffle (bool,optional): To shuffle phsae distribtuion, defult False
        shuffle_seed (float,optional): To shuffle phsae distribtuion, default None
        Init_dtheta (bool,optional): Give initial phase velocity by omega distribution
    Returns:
        Ksdf : (Forward)Simulationed data(pandas data frame), order parameter r, times, theta, dtheta and omega
        Ksrdf : (Backward)simulationed data(pandas data frame), order parameter r, times, theta, dtheta and omega
    """
    K_start, K_end = K_span
    Ks = np.round(np.arange(K_start, K_end + dK/2, dK),2)
    dtheta_init = np.zeros(N)
    if dist == "Normal":
        theta_init, omega_init, Kc = Normal(N, 0, 1, seed=seed)
    elif dist == "Lorentzian":
        theta_init, omega_init, Kc = Lorentzian(N, 0, 1, seed=seed)
    elif dist == "Quantile Lorentzian":
        theta_init, omega_init, Kc = Quantile_Lorentzian(N, 0, 1, seed=seed)
    elif dist == "Quantile Normal":
        theta_init, omega_init, Kc = Quantile_Normal(N, mean=0, sigma=1,seed=seed)
    Ksdf = pd.DataFrame({"rs":Ks},index=Ks,dtype=object)
    num = 0
    omega_init = np.sort(omega_init)
    if Init_dtheta:

        dtheta_init = scs.cauchy.rvs(0, 1, N,random_state = Init_dtheta_seed)
    if shuffle:
        np.random.seed(shuffle_seed)
        np.random.shuffle(theta_init)
    for K in tqdm(Ks):
        if num == 0:
            theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=theta_init,
                p_dtheta=dtheta_init,
                p_omega=omega_init,
                distribution="Normal",
            )
        else:
            theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=theta_s[-1],
                p_dtheta=dtheta_s[-1],
                p_omega=omega,
                distribution="Normal",
            )
        Ksdf["rs"][K] = rs
        num += 1

    theta_r_init, dtheta_r_init = theta_s[-1], dtheta_s[-1]

    dKr = -dK
    Ksr = np.round(np.arange(K_end, K_start + dKr/2, dKr),2)
    Ksrdf = pd.DataFrame({"rs":Ks},index=Ks,dtype=object)
    num_r = 0
    for K in tqdm(Ksr):
        if num_r == 0:
            theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=theta_r_init,
                p_dtheta=dtheta_r_init,
                p_omega=omega_init,
                distribution=dist,
            )
        else:
            theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=theta_s[-1],
                p_dtheta=dtheta_s[-1],
                p_omega=omega,
                distribution=dist,
            )
        Ksrdf["rs"][K] = rs

        num_r += 1
    if m==0:
        diff_osc = lambda x,dt: np.diff(x/dt,axis=0)

    return Ksdf,Ksrdf
