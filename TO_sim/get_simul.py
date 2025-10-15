import numba as nb
import numpy as np

# --- r, psi (복소수 없이) ---
@nb.njit(fastmath=True, cache=True)
def _order_param(theta):
    c = 0.0
    s = 0.0
    for i in range(theta.size):
        c += np.cos(theta[i])
        s += np.sin(theta[i])
    invN = 1.0 / theta.size
    c *= invN; s *= invN
    r = np.sqrt(c*c + s*s)
    psi = np.arctan2(s, c)
    return r, psi

# --- 커라모토 RHS: state와 같은 길이 반환 ---
@nb.njit(fastmath=True, cache=True)
def Kuramoto_mf_rhs(Theta, t_scalar, omega, N, m, K):
    # Theta: len N (m=0) 또는 len 2N (m>0)
    out = np.empty_like(Theta)
    if m == 0.0:
        theta = Theta[:N]
        r, psi = _order_param(theta)
        for i in range(N):
            out[i] = omega[i] + K * r * np.sin(psi - theta[i])
    else:
        theta = Theta[:N]
        dtheta = Theta[N:2*N]
        r, psi = _order_param(theta)
        invm = 1.0/m
        # θ' = dθ,  dθ' = ( -dθ + ω + K r sin(ψ-θ) ) / m
        for i in range(N):
            out[i] = dtheta[i]
        for i in range(N):
            out[N+i] = invm * (-dtheta[i] + omega[i] + K * r * np.sin(psi - theta[i]))
    return out

# --- RK4 + r 저장: 결과 shape = (n_save(eff), N+1) ---
@nb.njit(fastmath=True, cache=True)
def RK4_short_theta_store_r(y0, t, omega, N, m, K, sum_range=2010):
    save_time = sum_range
    integ_steps = t.size - save_time
    h = t[1] - t[0]

    # 저장 간격
    save_stride = 10 if h <= 0.01 else 1
    n_save_raw = save_time if h > 0.01 else save_time * 10
    n_save_eff = n_save_raw // save_stride

    # 출력: [r, theta...]
    out = np.empty((n_save_eff, N + 1), dtype=np.float64)

    # 상태 길이 확인
    state_len = y0.size
    twoN = 2 * N
    if m == 0.0 and state_len != N:
        # 1차인데 2N 형태로 들어왔으면 앞의 N만 사용
        y_ = y0[:N].copy()
    else:
        # 2차면 길이 2N 가정
        y_ = y0.copy()

    # 워밍업
    t0 = t[0]
    for k in range(integ_steps):
        tk = t0 + k*h
        k1 = Kuramoto_mf_rhs(y_, tk,      omega, N, m, K)
        k2 = Kuramoto_mf_rhs(y_ + 0.5*h*k1, tk+0.5*h, omega, N, m, K)
        k3 = Kuramoto_mf_rhs(y_ + 0.5*h*k2, tk+0.5*h, omega, N, m, K)
        k4 = Kuramoto_mf_rhs(y_ + h*k3,     tk+h,     omega, N, m, K)
        y_ = y_ + (h/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
        # 필요 시 래핑 (속도↑ 위해 주기적으로만 해도 됨)
        # if (k & 31) == 0:
        #     _wrap_angles_inplace(y_, N, m)

    # 저장 루프
    idx = 0
    for i in range(n_save_raw):
        ti = t0 + (integ_steps + i)*h
        k1 = Kuramoto_mf_rhs(y_, ti,      omega, N, m, K)
        k2 = Kuramoto_mf_rhs(y_ + 0.5*h*k1, ti+0.5*h, omega, N, m, K)
        k3 = Kuramoto_mf_rhs(y_ + 0.5*h*k2, ti+0.5*h, omega, N, m, K)
        k4 = Kuramoto_mf_rhs(y_ + h*k3,     ti+h,     omega, N, m, K)
        y_ = y_ + (h/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)

        if (i % save_stride) == 0:
            # r 계산 후 [r, theta] 저장
            if m == 0.0:
                theta = y_
            else:
                theta = y_[:N]
            r, _ = _order_param(theta)
            out[idx, 0] = r
            for j in range(N):
                out[idx, 1 + j] = theta[j]
            idx += 1

    return out


@nb.njit(fastmath=True, cache=True)
def RK4_short_theta_store_r_y(y0, t, omega, N, m, K, sum_range=2010):
    save_time = sum_range
    integ_steps = t.size - save_time
    h = t[1] - t[0]

    # 저장 간격
    save_stride = 10 if h <= 0.01 else 1
    n_save_raw = save_time if h > 0.01 else save_time * 10
    n_save_eff = n_save_raw // save_stride

    # 출력: [r, theta...]
    out = np.empty((n_save_eff, N + 1), dtype=np.float64)

    # 상태 길이 확인
    state_len = y0.size
    twoN = 2 * N
    if m == 0.0 and state_len != N:
        # 1차인데 2N 형태로 들어왔으면 앞의 N만 사용
        y_ = y0[:N].copy()
    else:
        # 2차면 길이 2N 가정
        y_ = y0.copy()

    # 워밍업
    t0 = t[0]
    for k in range(integ_steps):
        tk = t0 + k*h
        k1 = Kuramoto_mf_rhs(y_, tk,      omega, N, m, K)
        k2 = Kuramoto_mf_rhs(y_ + 0.5*h*k1, tk+0.5*h, omega, N, m, K)
        k3 = Kuramoto_mf_rhs(y_ + 0.5*h*k2, tk+0.5*h, omega, N, m, K)
        k4 = Kuramoto_mf_rhs(y_ + h*k3,     tk+h,     omega, N, m, K)
        y_ = y_ + (h/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
        # 필요 시 래핑 (속도↑ 위해 주기적으로만 해도 됨)
        # if (k & 31) == 0:
        #     _wrap_angles_inplace(y_, N, m)

    # 저장 루프
    idx = 0
    for i in range(n_save_raw):
        ti = t0 + (integ_steps + i)*h
        k1 = Kuramoto_mf_rhs(y_, ti,      omega, N, m, K)
        k2 = Kuramoto_mf_rhs(y_ + 0.5*h*k1, ti+0.5*h, omega, N, m, K)
        k3 = Kuramoto_mf_rhs(y_ + 0.5*h*k2, ti+0.5*h, omega, N, m, K)
        k4 = Kuramoto_mf_rhs(y_ + h*k3,     ti+h,     omega, N, m, K)
        y_ = y_ + (h/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)

        if (i % save_stride) == 0:
            # r 계산 후 [r, theta] 저장
            if m == 0.0:
                theta = y_
            else:
                theta = y_[:N]
            r, _ = _order_param(theta)
            out[idx, 0] = r
            for j in range(N):
                out[idx, 1 + j] = theta[j]
            idx += 1

    return out,y_



def get_r(K,m,Y,omega,N):
    dt = 0.1
    t = np.arange(0,10100,dt)

    T1 = RK4_short_theta_store_r(y0=Y, t=t, omega=omega, N=N, m=m, K=K, sum_range=1001)
    rs = T1[:,0]
    theta = T1[:,1:]
    avg_theta = (theta[1000] - theta[0])/(t[1000] - t[0])
    v_t = avg_theta
    expj = np.exp(1j*theta[-500:])

    con0 = np.abs(v_t - v_t[N//2]) < 0.15
    c_0, = np.where(con0)
    c_p, = np.where(np.logical_and(~con0, omega > 0))
    c_m, = np.where(np.logical_and(~con0, omega < 0))
    if len(c_p) > 0:
        conp = np.abs(v_t - np.min(v_t[c_p])) < 0.02
        c_p0, = np.where(conp)
        rp = np.mean(np.abs(np.sum(expj[:,c_p],axis=1)/N))
        if len(c_p0) > 0:
            rpl = np.mean(np.abs(np.sum(expj[:,c_p0],axis=1)/N))
        else:
            rpl = 0
        NP = len(c_p0)
    else:
        rp = 0
        rpl = 0
        NP = 0



    if len(c_m) > 0:
        conm = np.abs(v_t - np.max(v_t[c_m])) < 0.02
        c_m0, = np.where(conm)
        rm = np.mean(np.abs(np.sum(expj[:,c_m],axis=1)/N))
        if len(c_m0) > 0:
            rml = np.mean(np.abs(np.sum(expj[:,c_m0],axis=1)/N))
        else:
            rml = 0
        NM = len(c_m0)
    else:
        rm = 0
        rml = 0
        NM = 0
    r0 = np.mean(np.abs(np.sum(expj[:,c_0],axis=1)/N))
    rstd = np.std(rs[-500:])
    rs = np.mean(rs[-500:])
    N0 = len(c_0)
    return r0,rp,rm,rs,rpl,rml,N0,NP,NM,rstd



def get_r_y(K,m,Y,omega,N,tend=10100):
    dt = 0.1
    c_s = {}
    t = np.arange(0,tend,dt)

    T1,y_ = RK4_short_theta_store_r_y(y0=Y, t=t, omega=omega, N=N, m=m, K=K, sum_range=1001)
    rs = T1[:,0]
    theta = T1[:,1:]
    avg_theta = (theta[1000] - theta[0])/(t[1000] - t[0])
    v_t = avg_theta
    expj = np.exp(1j*theta[-500:])

    con0 = np.abs(v_t - v_t[N//2]) < 0.15
    c_0, = np.where(con0)
    c_s['0'] = c_0
    c_p, = np.where(np.logical_and(~con0, omega > 0))
    c_m, = np.where(np.logical_and(~con0, omega < 0))
    if len(c_p) > 0:
        conp = np.abs(v_t - np.min(v_t[c_p])) < 0.02
        c_p0, = np.where(conp)
        c_s['+'] = c_p
        rp = np.mean(np.abs(np.sum(expj[:,c_p],axis=1)/N))
        if len(c_p0) > 0:
            c_s['+l'] = c_p0
            rpl = np.mean(np.abs(np.sum(expj[:,c_p0],axis=1)/N))
        else:
            rpl = 0
        NP = len(c_p0)
    else:
        rp = 0
        rpl = 0
        NP = 0



    if len(c_m) > 0:
        conm = np.abs(v_t - np.max(v_t[c_m])) < 0.02
        c_m0, = np.where(conm)
        rm = np.mean(np.abs(np.sum(expj[:,c_m],axis=1)/N))
        c_s['-'] = c_m
        if len(c_m0) > 0:
            c_s['-l'] = c_m0
            rml = np.mean(np.abs(np.sum(expj[:,c_m0],axis=1)/N))
        else:
            rml = 0
        NM = len(c_m0)
    else:
        rm = 0
        rml = 0
        NM = 0
    r0 = np.mean(np.abs(np.sum(expj[:,c_0],axis=1)/N))
    rstd = np.std(rs[-500:])
    rs = np.mean(rs[-500:])
    N0 = len(c_0)
    return r0,rp,rm,rs,rpl,rml,N0,NP,NM,rstd,y_,avg_theta,c_s