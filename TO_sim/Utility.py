import os
import numpy as np
from knockknock import slack_sender
from knockknock import desktop_sender
from functools import wraps
from time import time


def Create_Folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory." + directory)


def Check_PM_idx_omega(omega, CHECK):
    P = np.searchsorted(omega, CHECK)
    M = np.searchsorted(omega, -CHECK)
    return (P, M)


webhook_url = (
    "https://hooks.slack.com/services/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"
)


@slack_sender(webhook_url=webhook_url, channel="#python-notification")
def Slack_Notification():
    pass


@desktop_sender(title="Knockknock Desktop Notifier")
def Desktop_Notification():
    pass


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("func:%r args:[%r, %r] took: %2.4f sec" % (f.__name__, args, kw, te - ts))
        return result

    return wrap
