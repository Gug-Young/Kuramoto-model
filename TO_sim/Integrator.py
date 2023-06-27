import numpy as np


def Euler(f, y0, t, args=()):
    n = len(t)
    h = t[1] - t[0]
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        y[i + 1] = y[i] + h * f(y[i], t[i], *args)
    return y


def RK2(f, y0, t, args=()):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        k1 = f(y[i], t[i], *args)
        k2 = f(y[i] + k1 * h / 2.0, t[i] + h / 2.0, *args)
        y[i + 1] = y[i] + k2 * h
    return y


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

def RK4_r(f, y0, t, args=(),result_time = 0):
    n = len(t) - result_time
    y = np.zeros((n, len(y0)))
    rs = np.zeros(n+result_time)
    y[0] = y0
    _,N,_,_ = args
    h = t[1] - t[0]
    rs[0] = abs(1/N*np.sum(np.exp(1j*y0[:N])))
    y_ = y0
    j = 0
    for i in range(result_time):
        k1,r = f(y_, t, *args)
        k2,_ = f(y_ + k1 * h / 2.0, t + h / 2.0, *args)
        k3,_ = f(y_ + k2 * h / 2.0, t + h / 2.0, *args)
        k4,_ = f(y_ + k3 * h, t + h, *args)
        y_ = y_ + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        rs[j+1] = r
        j+=1
    y[0] = y_
    for i in range(n - 1):
        k1,r = f(y[i], t[i], *args)
        k2,_ = f(y[i] + k1 * h / 2.0, t[i] + h / 2.0, *args)
        k3,_ = f(y[i] + k2 * h / 2.0, t[i] + h / 2.0, *args)
        k4,_ = f(y[i] + k3 * h, t[i] + h, *args)
        y[i + 1] = y[i] + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        rs[j+1] = r
        j+=1
    return y,rs

def RK4_sampling(f, y0, t,t_sample_idx, args=()):
    n = len(t)
    n_sample = len(t_sample_idx)
    y_sample = np.zeros((n_sample, len(y0)))
    num = 0
    h = t[1] - t[0]
    y = y0
    for i in range(n - 1):
        k1 = f(y, t, *args)
        k2 = f(y + k1 * h / 2.0, t + h / 2.0, *args)
        k3 = f(y + k2 * h / 2.0, t + h / 2.0, *args)
        k4 = f(y + k3 * h, t + h, *args)
        
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if i in t_sample_idx:
            y_sample[num] = y
            num +=1
    return y_sample



def Modified_Euler(f, y0, t, args=()):
    h = t[1] - t[0]
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        k1 = h * f(y[i], t[i], *args)
        k2 = h * f(y[i] + k1, t[i] + h, *args)
        y[i + 1] = y[i] + (k1 + k2) * 0.5
    return y


def leapfrog(f, df, y0, v0, t, args=()):
    h = t[1] - t[0]
    n = len(t)
    y = np.zeros((n, len(y0)))
    v = np.zeros((n + 1, len(v0)))
    y[0] = y0
    v[0] = v0
    v[1] = v[0] + 0.5 * h * df(y0, t[0])  # velocity at 1/2 delta t
    for i in range(1, n):
        y[i] = y[i - 1] + h * f(v[i], t[i] + h / 2)  # pos at t+ delta t
        v[i + 1] = v[i] + h * df(y[i], t[i])  # velocity at
    return y, v[1:, :]


def Error(origin, method, f, y0, t, args=()):
    origin_arr = np.array(origin(t, *args))
    method_arr = np.array(method(f, y0, t, args))
    if origin_arr.shape == method_arr.shape:
        Error_arr = np.fabs(origin_arr - method_arr)
    else:
        Error_arr = np.fabs(origin_arr - method_arr.T)
    max_Error = np.max(Error_arr)
    return max_Error, Error_arr


def RK4_r_sets(f, y0, t, args=(),result_time = 0):
    n = len(t) - result_time
    h = t[1] - t[0]
    if h <= 0.01:
        n_save = n//10 
    else:
        n_save = n
    
    y = np.zeros((n_save, *y0.shape))
    N_set = len(y0)
    _,N,_,_ = args
    rs = np.zeros((n+result_time,N_set,1))
    y[0] = y0
    rs[0] = abs(1/N*np.sum(np.exp(1j*y0[:,:N]),axis=1)).reshape((-1,1))
    y_ = y0
    j = 0
    for i in range(result_time):
        k1,r = f(y_, t, *args)
        k2,_ = f(y_ + k1 * h / 2.0, t + h / 2.0, *args)
        k3,_ = f(y_ + k2 * h / 2.0, t + h / 2.0, *args)
        k4,_ = f(y_ + k3 * h, t + h, *args)
        y_ = y_ + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        rs[j+1] = r
        j+=1
    y[0] = y_
    num = 0
    for i in range(n - 1):
        k1,r = f(y_, t[i], *args)
        k2,_ = f(y_ + k1 * h / 2.0, t[i] + h / 2.0, *args)
        k3,_ = f(y_ + k2 * h / 2.0, t[i] + h / 2.0, *args)
        k4,_ = f(y_ + k3 * h, t[i] + h, *args)
        y_ = y_ + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        rs[j+1] = r
        j+=1
        if i%10 == 0:
            num+= 1
            y[num] = y_
    return y,rs