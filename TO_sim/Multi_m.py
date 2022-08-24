from TO_sim.Sol_Kuramoto import *
from TO_sim.gen_Distribution import *
import matplotlib.pyplot as plt
import parmap
import numpy as np

# from tqdm import tqdm_notebook
from tqdm.notebook import tqdm

Rand_P = lambda N, dinv: np.arange(1, N + 1) / (N + 1) + (1 / (N + 1)) * (1 / dinv) * (
    np.random.rand(N) - 0.5
)
from TO_sim.To_Draw import *
import os


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory." + directory)


def normalization(arr):
    min_, max_ = np.min(arr), np.max(arr)
    N_arr = (arr - min_) / (max_ - min_)
    return N_arr


def normalization_del(arr):
    min_, max_ = np.min(arr), np.max(arr)
    N_arr = (arr - min_) / (max_ - min_)
    min_idx, max_idx = np.argmin(arr), np.argmax(arr)
    Nd_arr = np.delete(arr, (min_idx, max_idx))
    return Nd_arr


def Noise_grow(N, Noise, length):
    arr = np.arange(1, N + 1) / (N + 1)
    Noise_ = []
    Noise_.append(arr)
    for i in range(length):
        arr = arr + (1 / (N + 1)) * (Noise) * (np.random.rand(N) - 0.5)
        Noise_.append(arr)
    N_Noise = normalization(np.array(Noise_))
    return N_Noise


Q_L = lambda mu, d, p: d * np.tan(np.pi / 2 * (2 * p - 1)) + mu
Q_L_ALL = lambda mu, d, N: Q_L(mu, d, np.random.rand(N))
Rand_P = lambda N, noise: np.arange(1, N + 1) / (N + 1) + (1 / (N + 1)) * (noise) * (
    np.random.rand(N) - 0.5
)
Normal_P = lambda N, dinv: np.arange(1, N + 1) / (N + 1)
Rand_P_NEW = lambda N, noise: normalization_del(Rand_P(N + 2, noise))
Rand_P_Sort = lambda N, noise: np.sort(normalization(Rand_P(N + 2, noise)))[1:-1]
Make_noise = lambda arr, D: arr + D * (2 * np.random.rand(len(arr)) - 1)


def Make_Omega(p_method, N, Noise):
    if p_method == "Sorted random":
        return np.sort(Q_L_ALL(0, 1, N))
    elif p_method == "Random":
        return Q_L_ALL(0, 1, N)
    elif p_method == "Normalize sorted perturbation":
        return Q_L(0, 1, Rand_P_Sort(N, Noise))
    elif p_method == "Normalize perturbation":
        return Q_L(0, 1, Rand_P_NEW(N, Noise))
    elif p_method == "Perturbation":
        return np.sort(Q_L(0, 1, Rand_P(N, Noise)))
    elif p_method == "Sorted perturbation":
        return Q_L(0, 1, Rand_P(N, Noise))


def Sim_Multi_mset(Noise, p_method="Normalize sorted perturbation"):
    """_summary_

    Args:
        Noise (float): Give noise at P_array to make perturbation at omega distribtion
        p_method (str, optional): "Sorted random", "Random", "Normalize sorted perturbation", "Normalize perturbation", "Perturbation" ,"Sorted perturbation". Defaults to "Normalize sorted perturbation".
    """
    for m in [0.95, 2, 6]:
        N = 500
        K_start = 0.1
        K_end = 12.5
        dK = 0.2
        Ks = np.arange(K_start, K_end + dK, dK)
        # m = 0.95
        dtheta_init = np.zeros(N)
        t_end = 1000
        Noise_inv = 1 / Noise
        dt = 0.1
        dist = "Quantile Lorentzian"
        if dist == "Normal":
            theta_init, omega_init, Kc = Normal(N, 0, 1, seed=0)
        elif dist == "Lorentzian":
            theta_init, omega_init, Kc = Lorentzian(N, 0, 1, seed=0)
        elif dist == "Quantile Lorentzian":
            theta_init, omega_init, Kc = Quantile_Lorentzian(N, 0, 1, seed=0)

        theta_dic = {}
        dtheta_dic = {}
        rs_dic = {}
        omega_dic = {}
        t_dic = {}

        num = 0
        for K in tqdm(Ks):
            if num == 0:
                omega = Make_Omega(p_method, N, Noise)
                theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                    N,
                    K,
                    m,
                    (0, t_end),
                    dt=dt,
                    p_theta=theta_init,
                    p_dtheta=dtheta_init,
                    p_omega=omega,
                    distribution="Normal",
                )
            else:
                omega = Make_Omega(p_method, N, Noise)
                theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                    N,
                    K,
                    m,
                    (0, t_end),
                    dt=dt,
                    p_theta=theta_dic[num - 1][-1],
                    p_dtheta=dtheta_dic[num - 1][-1],
                    p_omega=omega,
                    distribution="Normal",
                )
                # theta_s,dtheta_s,omega,rs,t = Sol_Kuramoto_mf(N,K,m,(0,t_end),dt=dt,p_theta=theta_dic[num-1][-1],p_dtheta=dtheta_init,p_omega=omega_dic[num-1],distribution="Normal")
            theta_dic[num] = theta_s
            dtheta_dic[num] = dtheta_s
            rs_dic[num] = rs
            omega_dic[num] = omega
            t_dic[num] = t + num * t_end
            num += 1
        theta_r_init, dtheta_r_init = theta_dic[num - 1][-1], dtheta_dic[num - 1][-1]

        theta_r_dic = {}
        dtheta_r_dic = {}
        rs_r_dic = {}
        omega_r_dic = {}
        t_r_dic = {}
        dKr = -0.2
        Ksr = np.arange(K_end, K_start + dKr, dKr)

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
                # theta_s,dtheta_s,omega,rs,t = Sol_Kuramoto_mf(N,K,m,(0,t_end),dt=dt,p_theta=theta_r_init,p_dtheta=dtheta_init,p_omega=omega_init,distribution=dist)
            else:
                theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                    N,
                    K,
                    m,
                    (0, t_end),
                    dt=dt,
                    p_theta=theta_r_dic[num_r - 1][-1],
                    p_dtheta=dtheta_r_dic[num_r - 1][-1],
                    p_omega=omega_r_dic[num_r - 1],
                    distribution=dist,
                )
                # theta_s,dtheta_s,omega,rs,t = Sol_Kuramoto_mf(N,K,m,(0,t_end),dt=dt,p_theta=theta_r_dic[num_r-1][-1],p_dtheta=dtheta_init,p_omega=omega_r_dic[num_r-1],distribution=dist)
            theta_r_dic[num_r] = theta_s
            dtheta_r_dic[num_r] = dtheta_s
            rs_r_dic[num_r] = rs
            omega_r_dic[num_r] = omega
            t_r_dic[num_r] = t + num_r * t_end
            num_r += 1

        createFolder(f"Review2/Control_P/{p_method}/Noise = {Noise}")
        Time_R(
            Ks,
            t_dic,
            rs_dic,
            t_r_dic,
            rs_r_dic,
            dK,
            dt,
            t_end,
            N,
            m,
            mean_time=50,
            save=True,
            Folder_name=f"Review2/Control_P/{p_method}/Noise = {Noise}",
        )


def Sim_Multi_m_giveNoise(m, Noise, p_method="Normalize sorted perturbation"):
    """_summary_

    Args:
        Noise (float): Give noise at P_array to make perturbation at omega distribtion
        p_method (str, optional): "Sorted random", "Random", "Normalize sorted perturbation", "Normalize perturbation", "Perturbation" ,"Sorted perturbation". Defaults to "Normalize sorted perturbation".
    """

    N = 500
    K_start = 0.1
    K_end = 12.5
    dK = 0.2
    Ks = np.arange(K_start, K_end + dK, dK)
    # m = 0.95
    dtheta_init = np.zeros(N)
    t_end = 1000
    Noise_inv = 1 / Noise
    dt = 0.1
    dist = "Quantile Lorentzian"
    if dist == "Normal":
        theta_init, omega_init, Kc = Normal(N, 0, 1, seed=0)
    elif dist == "Lorentzian":
        theta_init, omega_init, Kc = Lorentzian(N, 0, 1, seed=0)
    elif dist == "Quantile Lorentzian":
        theta_init, omega_init, Kc = Quantile_Lorentzian(N, 0, 1, seed=0)

    theta_dic = {}
    dtheta_dic = {}
    rs_dic = {}
    omega_dic = {}
    t_dic = {}

    num = 0
    for K in tqdm(Ks):
        if num == 0:
            omega = Make_Omega(p_method, N, Noise)
            theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=theta_init,
                p_dtheta=dtheta_init,
                p_omega=omega,
                distribution="Normal",
            )
        else:
            omega = Make_Omega(p_method, N, Noise)
            theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=theta_dic[num - 1][-1],
                p_dtheta=dtheta_dic[num - 1][-1],
                p_omega=omega,
                distribution="Normal",
            )
            # theta_s,dtheta_s,omega,rs,t = Sol_Kuramoto_mf(N,K,m,(0,t_end),dt=dt,p_theta=theta_dic[num-1][-1],p_dtheta=dtheta_init,p_omega=omega_dic[num-1],distribution="Normal")
        theta_dic[num] = theta_s
        dtheta_dic[num] = dtheta_s
        rs_dic[num] = rs
        omega_dic[num] = omega
        t_dic[num] = t + num * t_end
        num += 1
    theta_r_init, dtheta_r_init = theta_dic[num - 1][-1], dtheta_dic[num - 1][-1]

    theta_r_dic = {}
    dtheta_r_dic = {}
    rs_r_dic = {}
    omega_r_dic = {}
    t_r_dic = {}
    dKr = -0.2
    Ksr = np.arange(K_end, K_start + dKr, dKr)

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
            # theta_s,dtheta_s,omega,rs,t = Sol_Kuramoto_mf(N,K,m,(0,t_end),dt=dt,p_theta=theta_r_init,p_dtheta=dtheta_init,p_omega=omega_init,distribution=dist)
        else:
            theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                N,
                K,
                m,
                (0, t_end),
                dt=dt,
                p_theta=theta_r_dic[num_r - 1][-1],
                p_dtheta=dtheta_r_dic[num_r - 1][-1],
                p_omega=omega_r_dic[num_r - 1],
                distribution=dist,
            )
            # theta_s,dtheta_s,omega,rs,t = Sol_Kuramoto_mf(N,K,m,(0,t_end),dt=dt,p_theta=theta_r_dic[num_r-1][-1],p_dtheta=dtheta_init,p_omega=omega_r_dic[num_r-1],distribution=dist)
        theta_r_dic[num_r] = theta_s
        dtheta_r_dic[num_r] = dtheta_s
        rs_r_dic[num_r] = rs
        omega_r_dic[num_r] = omega
        t_r_dic[num_r] = t + num_r * t_end
        num_r += 1

    createFolder(f"Review2/Control_P/{p_method}/Noise = {Noise}")
    Time_R(
        Ks,
        t_dic,
        rs_dic,
        t_r_dic,
        rs_r_dic,
        dK,
        dt,
        t_end,
        N,
        m,
        mean_time=50,
        save=True,
        Folder_name=f"Review2/Control_P/{p_method}/Noise = {Noise}",
    )


def Sim_Multi_mset_NOT_TQDM(Noise, p_method="Normalize sorted perturbation"):
    """_summary_

    Args:
        Noise (float): Give noise at P_array to make perturbation at omega distribtion
        p_method (str, optional): "Sorted random", "Random", "Normalize sorted perturbation", "Normalize perturbation", "Perturbation" ,"Sorted perturbation". Defaults to "Normalize sorted perturbation".
    """
    for m in [0.95, 2, 6]:
        N = 500
        K_start = 0.1
        K_end = 12.5
        dK = 0.2
        Ks = np.arange(K_start, K_end + dK, dK)
        # m = 0.95
        dtheta_init = np.zeros(N)
        t_end = 1000
        Noise_inv = 1 / Noise
        dt = 0.1
        dist = "Quantile Lorentzian"
        if dist == "Normal":
            theta_init, omega_init, Kc = Normal(N, 0, 1, seed=0)
        elif dist == "Lorentzian":
            theta_init, omega_init, Kc = Lorentzian(N, 0, 1, seed=0)
        elif dist == "Quantile Lorentzian":
            theta_init, omega_init, Kc = Quantile_Lorentzian(N, 0, 1, seed=0)

        theta_dic = {}
        dtheta_dic = {}
        rs_dic = {}
        omega_dic = {}
        t_dic = {}

        num = 0
        for K in Ks:
            if num == 0:
                omega = Make_Omega(p_method, N, Noise)
                theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                    N,
                    K,
                    m,
                    (0, t_end),
                    dt=dt,
                    p_theta=theta_init,
                    p_dtheta=dtheta_init,
                    p_omega=omega,
                    distribution="Normal",
                )
            else:
                omega = Make_Omega(p_method, N, Noise)
                theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                    N,
                    K,
                    m,
                    (0, t_end),
                    dt=dt,
                    p_theta=theta_dic[num - 1][-1],
                    p_dtheta=dtheta_dic[num - 1][-1],
                    p_omega=omega,
                    distribution="Normal",
                )
                # theta_s,dtheta_s,omega,rs,t = Sol_Kuramoto_mf(N,K,m,(0,t_end),dt=dt,p_theta=theta_dic[num-1][-1],p_dtheta=dtheta_init,p_omega=omega_dic[num-1],distribution="Normal")
            theta_dic[num] = theta_s
            dtheta_dic[num] = dtheta_s
            rs_dic[num] = rs
            omega_dic[num] = omega
            t_dic[num] = t + num * t_end
            num += 1
        theta_r_init, dtheta_r_init = theta_dic[num - 1][-1], dtheta_dic[num - 1][-1]

        theta_r_dic = {}
        dtheta_r_dic = {}
        rs_r_dic = {}
        omega_r_dic = {}
        t_r_dic = {}
        dKr = -0.2
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
                # theta_s,dtheta_s,omega,rs,t = Sol_Kuramoto_mf(N,K,m,(0,t_end),dt=dt,p_theta=theta_r_init,p_dtheta=dtheta_init,p_omega=omega_init,distribution=dist)
            else:
                theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                    N,
                    K,
                    m,
                    (0, t_end),
                    dt=dt,
                    p_theta=theta_r_dic[num_r - 1][-1],
                    p_dtheta=dtheta_r_dic[num_r - 1][-1],
                    p_omega=omega_r_dic[num_r - 1],
                    distribution=dist,
                )
                # theta_s,dtheta_s,omega,rs,t = Sol_Kuramoto_mf(N,K,m,(0,t_end),dt=dt,p_theta=theta_r_dic[num_r-1][-1],p_dtheta=dtheta_init,p_omega=omega_r_dic[num_r-1],distribution=dist)
            theta_r_dic[num_r] = theta_s
            dtheta_r_dic[num_r] = dtheta_s
            rs_r_dic[num_r] = rs
            omega_r_dic[num_r] = omega
            t_r_dic[num_r] = t + num_r * t_end
            num_r += 1

        createFolder(f"Review2/Control_P/{p_method}/Noise = {Noise}")
        Time_R(
            Ks,
            t_dic,
            rs_dic,
            t_r_dic,
            rs_r_dic,
            dK,
            dt,
            t_end,
            N,
            m,
            mean_time=50,
            save=True,
            Folder_name=f"Review2/Control_P/{p_method}/Noise = {Noise}",
        )


if __name__ == "__main__":
    Noise_list = list(map(int, input("Typing the Noise list")).split())
    print(Noise_list)
    for Noise in Noise_list:
        Sim_Multi_mset(Noise)


def Sim_Multi_mset_Noise(Noise):
    """_summary_

    Args:
        Noise (float): Give noise at P_array to make perturbation at omega distribtion
    """
    for m in [0.95, 2, 6]:
        N = 500
        K_start = 0.1
        K_end = 12.5
        dK = 0.2
        Ks = np.arange(K_start, K_end + dK, dK)
        # m = 0.95
        dtheta_init = np.zeros(N)
        t_end = 1000
        Noise_inv = 1 / Noise
        dt = 0.1
        dist = "Quantile Lorentzian"
        if dist == "Normal":
            theta_init, omega_init, Kc = Normal(N, 0, 1, seed=0)
        elif dist == "Lorentzian":
            theta_init, omega_init, Kc = Lorentzian(N, 0, 1, seed=0)
        elif dist == "Quantile Lorentzian":
            theta_init, omega_init, Kc = Quantile_Lorentzian(N, 0, 1, seed=0)

        theta_dic = {}
        dtheta_dic = {}
        rs_dic = {}
        omega_dic = {}
        t_dic = {}

        num = 0
        for K in tqdm(Ks):
            if num == 0:
                omega = Make_noise(omega_init, Noise)
                theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                    N,
                    K,
                    m,
                    (0, t_end),
                    dt=dt,
                    p_theta=theta_init,
                    p_dtheta=dtheta_init,
                    p_omega=omega,
                    distribution="Normal",
                )
            else:
                omega = Make_noise(omega, Noise)
                theta = Make_noise(theta_s[-1], Noise)
                theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                    N,
                    K,
                    m,
                    (0, t_end),
                    dt=dt,
                    p_theta=theta_dic[num - 1][-1],
                    p_dtheta=dtheta_dic[num - 1][-1],
                    p_omega=omega,
                    distribution="Normal",
                )
                # theta_s,dtheta_s,omega,rs,t = Sol_Kuramoto_mf(N,K,m,(0,t_end),dt=dt,p_theta=theta_dic[num-1][-1],p_dtheta=dtheta_init,p_omega=omega_dic[num-1],distribution="Normal")
            theta_dic[num] = theta_s
            dtheta_dic[num] = dtheta_s
            rs_dic[num] = rs
            omega_dic[num] = omega
            t_dic[num] = t + num * t_end
            num += 1
        theta_r_init, dtheta_r_init = theta_dic[num - 1][-1], dtheta_dic[num - 1][-1]

        theta_r_dic = {}
        dtheta_r_dic = {}
        rs_r_dic = {}
        omega_r_dic = {}
        t_r_dic = {}
        dKr = -0.2
        Ksr = np.arange(K_end, K_start + dKr, dKr)

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
                # theta_s,dtheta_s,omega,rs,t = Sol_Kuramoto_mf(N,K,m,(0,t_end),dt=dt,p_theta=theta_r_init,p_dtheta=dtheta_init,p_omega=omega_init,distribution=dist)
            else:
                theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                    N,
                    K,
                    m,
                    (0, t_end),
                    dt=dt,
                    p_theta=theta_r_dic[num_r - 1][-1],
                    p_dtheta=dtheta_r_dic[num_r - 1][-1],
                    p_omega=omega_r_dic[num_r - 1],
                    distribution=dist,
                )
                # theta_s,dtheta_s,omega,rs,t = Sol_Kuramoto_mf(N,K,m,(0,t_end),dt=dt,p_theta=theta_r_dic[num_r-1][-1],p_dtheta=dtheta_init,p_omega=omega_r_dic[num_r-1],distribution=dist)
            theta_r_dic[num_r] = theta_s
            dtheta_r_dic[num_r] = dtheta_s
            rs_r_dic[num_r] = rs
            omega_r_dic[num_r] = omega
            t_r_dic[num_r] = t + num_r * t_end
            num_r += 1

        createFolder(f"Review2/Noise/Noise = {Noise}")
        Time_R(
            Ks,
            t_dic,
            rs_dic,
            t_r_dic,
            rs_r_dic,
            dK,
            dt,
            t_end,
            N,
            m,
            mean_time=50,
            save=True,
            Folder_name=f"Review2/Noise/Noise = {Noise}",
        )

def Sim_Multi_mset_thetaNoise(Noise):
    """_summary_

    Args:
        Noise (float): Give noise at P_array to make perturbation at omega distribtion
    """
    for m in [0.95, 2, 6]:
        N = 500
        K_start = 0.1
        K_end = 12.5
        dK = 0.2
        Ks = np.arange(K_start, K_end + dK, dK)
        # m = 0.95
        dtheta_init = np.zeros(N)
        t_end = 1000
        Noise_inv = 1 / Noise
        dt = 0.1
        dist = "Quantile Lorentzian"
        if dist == "Normal":
            theta_init, omega_init, Kc = Normal(N, 0, 1, seed=0)
        elif dist == "Lorentzian":
            theta_init, omega_init, Kc = Lorentzian(N, 0, 1, seed=0)
        elif dist == "Quantile Lorentzian":
            theta_init, omega_init, Kc = Quantile_Lorentzian(N, 0, 1, seed=0)

        theta_dic = {}
        dtheta_dic = {}
        rs_dic = {}
        omega_dic = {}
        t_dic = {}

        num = 0
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
                theta = Make_noise(theta_s[-1], Noise)
                theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                    N,
                    K,
                    m,
                    (0, t_end),
                    dt=dt,
                    p_theta=theta,
                    p_dtheta=dtheta_dic[num - 1][-1],
                    p_omega=omega_init,
                    distribution="Normal",
                )
                # theta_s,dtheta_s,omega,rs,t = Sol_Kuramoto_mf(N,K,m,(0,t_end),dt=dt,p_theta=theta_dic[num-1][-1],p_dtheta=dtheta_init,p_omega=omega_dic[num-1],distribution="Normal")
            theta_dic[num] = theta_s
            dtheta_dic[num] = dtheta_s
            rs_dic[num] = rs
            omega_dic[num] = omega_init
            t_dic[num] = t + num * t_end
            num += 1
        theta_r_init, dtheta_r_init = theta_dic[num - 1][-1], dtheta_dic[num - 1][-1]

        theta_r_dic = {}
        dtheta_r_dic = {}
        rs_r_dic = {}
        omega_r_dic = {}
        t_r_dic = {}
        dKr = -0.2
        Ksr = np.arange(K_end, K_start + dKr, dKr)

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
                # theta_s,dtheta_s,omega,rs,t = Sol_Kuramoto_mf(N,K,m,(0,t_end),dt=dt,p_theta=theta_r_init,p_dtheta=dtheta_init,p_omega=omega_init,distribution=dist)
            else:
                theta_s, dtheta_s, omega, rs, t = Sol_Kuramoto_mf(
                    N,
                    K,
                    m,
                    (0, t_end),
                    dt=dt,
                    p_theta=theta_r_dic[num_r - 1][-1],
                    p_dtheta=dtheta_r_dic[num_r - 1][-1],
                    p_omega=omega_r_dic[num_r - 1],
                    distribution=dist,
                )
                # theta_s,dtheta_s,omega,rs,t = Sol_Kuramoto_mf(N,K,m,(0,t_end),dt=dt,p_theta=theta_r_dic[num_r-1][-1],p_dtheta=dtheta_init,p_omega=omega_r_dic[num_r-1],distribution=dist)
            theta_r_dic[num_r] = theta_s
            dtheta_r_dic[num_r] = dtheta_s
            rs_r_dic[num_r] = rs
            omega_r_dic[num_r] = omega
            t_r_dic[num_r] = t + num_r * t_end
            num_r += 1

        createFolder(f"Review2/Noise_theta/Noise = {Noise}")
        Time_R(
            Ks,
            t_dic,
            rs_dic,
            t_r_dic,
            rs_r_dic,
            dK,
            dt,
            t_end,
            N,
            m,
            mean_time=50,
            save=True,
            Folder_name=f"Review2/Noise_theta/Noise = {Noise}",
        )