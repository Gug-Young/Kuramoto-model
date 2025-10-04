import numpy as np
import numba


@numba.jit(nopython=True)
def RK4(f, y0, t, args=()):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0

    for i in range(n - 1):
        h = t[i + 1] - t[i]
        k1 = f(y[i], t[i], *args)
        k2 = f(y[i] + k1 * h / 2.0, t[i] + h / 2.0, *args)
        k3 = f(y[i] + k2 * h / 2.0, t[i] + h / 2.0, *args)
        k4 = f(y[i] + k3 * h, t[i] + h, *args)
        y[i + 1] = y[i] + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y

@numba.jit(nopython=True)
def RK4_short(f, y0, t, args=(),result_time = 2010):
    n = len(t) - result_time
    h = t[1] - t[0]
    if h <= 0.01:
        n_save = n//10 + 1
    else:
        n_save = n
    y = np.zeros((n_save, len(y0)))
    y[0] = y0
    y_ = y0
    for i in range(result_time):
        k1 = f(y_, t, *args)
        k2 = f(y_ + k1 * h / 2.0, t + h / 2.0, *args)
        k3 = f(y_ + k2 * h / 2.0, t + h / 2.0, *args)
        k4 = f(y_ + k3 * h, t + h, *args)
        y_ = y_ + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    num = 0
    y[0] = y_
    if h <= 0.01:
        for i in range(n - 1):
            k1= f(y_, t[i], *args)
            k2= f(y_ + k1 * h / 2.0, t[i] + h / 2.0, *args)
            k3= f(y_ + k2 * h / 2.0, t[i] + h / 2.0, *args)
            k4= f(y_ + k3 * h, t[i] + h, *args)
            y_ =y_ + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if i%10 == 0:
                num+= 1
                y[num] = y_
    else:
        for i in range(n - 1):
            k1= f(y[i], t[i], *args)
            k2= f(y[i] + k1 * h / 2.0, t[i] + h / 2.0, *args)
            k3= f(y[i] + k2 * h / 2.0, t[i] + h / 2.0, *args)
            k4= f(y[i] + k3 * h, t[i] + h, *args)
            y[i + 1] = y[i] + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


@numba.jit(nopython=True)
def RK4_short_theta(f, y0, t, args=(),sum_range = 2010):
    save_time = sum_range
    integtime = len(t) - save_time
    h = t[1] - t[0]
    if h <= 0.01:
        n_save = sum_range*10
    else:
        n_save = sum_range
    N = len(y0)//2
    y = np.zeros((n_save, N))
    y_ = y0
    for i in range(integtime):
        k1 = f(y_, t, *args)
        k2 = f(y_ + k1 * h / 2.0, t + h / 2.0, *args)
        k3 = f(y_ + k2 * h / 2.0, t + h / 2.0, *args)
        k4 = f(y_ + k3 * h, t + h, *args)
        y_ = y_ + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    num = 0
    y[0] = y_
    if h <= 0.01:
        for i in range(n_save):
            k1= f(y_, t[i], *args)
            k2= f(y_ + k1 * h / 2.0, t[i] + h / 2.0, *args)
            k3= f(y_ + k2 * h / 2.0, t[i] + h / 2.0, *args)
            k4= f(y_ + k3 * h, t[i] + h, *args)
            y_ =y_ + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if i%10 == 0:
                num+= 1
                y[num] = y_[:N]
    else:
        for i in range(n_save):
            k1= f(y_, t[i], *args)
            k2= f(y_ + k1 * h / 2.0, t[i] + h / 2.0, *args)
            k3= f(y_ + k2 * h / 2.0, t[i] + h / 2.0, *args)
            k4= f(y_ + k3 * h, t[i] + h, *args)
            y_ =y_ + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            y[i] = y_[:N]
    return y

@numba.jit(nopython=True)
def RK4_sets(f, y0, t, args=(),result_time = 0):
    n = len(t) - result_time
    h = t[1] - t[0]
    if h <= 0.01:
        n_save = n//10 + 1
    else:
        n_save = n
    y = np.zeros((n_save, *y0.shape))
    y[0] = y0
    y_ = y0
    for i in range(result_time):
        k1 = f(y_, t, *args)
        k2 = f(y_ + k1 * h / 2.0, t + h / 2.0, *args)
        k3 = f(y_ + k2 * h / 2.0, t + h / 2.0, *args)
        k4 = f(y_ + k3 * h, t + h, *args)
        y_ = y_ + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    y[0] = y_
    num = 0

    if h <= 0.01:
        for i in range(n - 1):
            k1= f(y_, t[i], *args)
            k2= f(y_ + k1 * h / 2.0, t[i] + h / 2.0, *args)
            k3= f(y_ + k2 * h / 2.0, t[i] + h / 2.0, *args)
            k4= f(y_ + k3 * h, t[i] + h, *args)
            y_ =y_ + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if i%10 == 0:
                num+= 1
                y[num] = y_
    else:
        for i in range(n - 1):
            k1= f(y[i], t[i], *args)
            k2= f(y[i] + k1 * h / 2.0, t[i] + h / 2.0, *args)
            k3= f(y[i] + k2 * h / 2.0, t[i] + h / 2.0, *args)
            k4= f(y[i] + k3 * h, t[i] + h, *args)
            y[i + 1] = y[i] + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


@numba.jit(nopython=True)
def get_order_parameter(theta,N):
    ''' get theta and return r and theta'''
    rpsi = 1/N*np.sum(np.exp(1j*theta))
    r = np.abs(rpsi)
    psi = np.angle(rpsi)
    return r,psi

@numba.jit(nopython=True)
def Kuramoto_1st_mf(Theta,t,omega,N,m,K):
    # print("Case m = 0")
    Theta = Theta.copy()
    theta = Theta[:N]
    r,psi = get_order_parameter(theta,N)
    dtheta = omega + K*r*np.sin(psi - theta)
    Theta[:N] = dtheta
    Theta[N:2*N] = dtheta
    return Theta





# def RK4_r_sets(f, y0, t, args=(),result_time = 0):
#     n = len(t) - result_time
#     h = t[1] - t[0]
#     if h <= 0.01:
#         n_save = n//10 + 1
#     else:
#         n_save = n
    
#     y = cp.zeros((n_save, *y0.shape))
#     N_set = len(y0)
#     _,N,_,_ = args
#     rs = np.zeros((n+result_time,N_set,1))
#     y[0] = y0
#     rs[0] = abs(1/N*np.sum(np.exp(1j*y0[:,:N]),axis=1)).reshape((-1,1))
#     y_ = y0
#     j = 0
#     for i in range(result_time):
#         k1,r = f(y_, t, *args)
#         k2,_ = f(y_ + k1 * h / 2.0, t + h / 2.0, *args)
#         k3,_ = f(y_ + k2 * h / 2.0, t + h / 2.0, *args)
#         k4,_ = f(y_ + k3 * h, t + h, *args)
#         y_ = y_ + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
#         rs[j+1] = r
#         j+=1
#     y[0] = y_
#     num = 0
#     if h <= 0.01:
    
#         for i in range(n - 1):
#             k1,r = f(y_, t[i], *args)
#             k2,_ = f(y_ + k1 * h / 2.0, t[i] + h / 2.0, *args)
#             k3,_ = f(y_ + k2 * h / 2.0, t[i] + h / 2.0, *args)
#             k4,_ = f(y_ + k3 * h, t[i] + h, *args)
#             y_ = y_ + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
#             rs[j+1] = r
#             j+=1
#             if i%10 == 0:
#                 num+= 1
#                 y[num] = y_
#     else:
#         for i in range(n - 1):
#             k1,r = f(y[i], t[i], *args)
#             k2,_ = f(y[i] + k1 * h / 2.0, t[i] + h / 2.0, *args)
#             k3,_ = f(y[i] + k2 * h / 2.0, t[i] + h / 2.0, *args)
#             k4,_ = f(y[i] + k3 * h, t[i] + h, *args)
#             y[i + 1] = y[i] + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
#             rs[j+1] = r
#             j+=1
#     return y,rs