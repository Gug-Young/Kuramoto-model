import numpy as np


from scipy.signal import find_peaks


def cluster(avg_dtheta, N):
    def to_cluster_barg(idx, peaks_new):
        C = idx, idx + 1
        arg_C = peaks_new[C[0]], peaks_new[C[1]]
        cluster = np.arange(arg_C[0], arg_C[1])
        return cluster

    def to_cstability(x, diff_dtheta):
        try:
            return np.mean(diff_dtheta[x[1:]])
        except IndexError:
            return np.nan

    def to_mean_avg_d_o(x, avg_dtheta, index):
        try:
            return np.mean(avg_dtheta[index][x])
        except IndexError:
            return np.nan

    iter_time = 200 * 5
    num = 0
    for index in range(-iter_time, 0):
        arg = np.argsort(avg_dtheta[index])
        SD = avg_dtheta[index][arg]
        diff_dtheta = np.diff([SD[0], *SD, SD[-1]])
        peaks, P = find_peaks(diff_dtheta, height=0.01)

        # peaks = peaks[np.where((peaks<N)&(peaks>1))]

        try:
            peaks_new = np.array([peaks[0], *peaks])
        except IndexError:
            peaks_new = np.array([0, N])

        psize = np.diff(peaks_new)
        arg_psize = np.argsort(psize)[::-1]
        sort_psize = np.sort(psize)[::-1]
        clusters = np.array(
            [to_cluster_barg(arg, peaks_new) for arg in arg_psize], dtype=object
        )[:10]
        if len(clusters) == 1:
            clusters = np.array([np.arange(peaks_new[0], N)])
        c_stability = np.array(
            list(map(to_cstability, clusters, [diff_dtheta] * len(clusters)))
        )
        mean_omega = np.array(
            list(
                map(
                    to_mean_avg_d_o,
                    clusters,
                    [avg_dtheta] * len(clusters),
                    [index] * len(clusters),
                )
            )
        )
        if num == 0:
            psize_array = sort_psize[:10]
            cluster_array = clusters
            c_stability_array = c_stability
            mean_omega_array = mean_omega
            arg_array = arg

            num += 1
        else:
            cluster_array = np.c_[cluster_array, clusters]
            psize_array = np.c_[psize_array, sort_psize[:10]]
            c_stability_array = np.c_[c_stability_array, c_stability]
            mean_omega_array = np.c_[mean_omega_array, mean_omega]
            arg_array = np.c_[arg_array, arg]

    Is_group = np.where((np.std(psize_array, axis=1) == 0) & (psize_array[:, -1] > 10))

    CM_S = np.mean(psize_array[Is_group], axis=1)
    CM_O = np.mean(mean_omega_array[Is_group], axis=1)

    return CM_S, CM_O
