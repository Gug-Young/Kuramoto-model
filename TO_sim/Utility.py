import os
import numpy as np

def Create_Folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory." + directory)


def Check_PM_idx_omega(omega,CHECK):
    P = np.searchsorted(omega,CHECK)
    M = np.searchsorted(omega,-CHECK)
    return (P,M)
    