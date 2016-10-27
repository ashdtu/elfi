import numpy as np


def MA2(n, N, t1, t2, prng=None, latents=None):
    if latents is None:
        if prng is None:
            prng = np.random.RandomState()
        latents = prng.randn(N,n+2) # i.i.d. sequence ~ N(0,1)
    u = np.atleast_2d(latents)
    y = u[:,2:] + t1 * u[:,1:-1] + t2 * u[:,:-2]
    return y


def autocov(lag, y):
    # weak stationary version
    y = (y - np.mean(y, axis=1, keepdims=True))
    tau = np.mean(y[:,lag:] * y[:,:-lag], axis=1, keepdims=True) / np.var(y, axis=1, keepdims=True)
    return tau

    # no stationary assumptions
    #X_t = y[:,:-lag]
    #X_s = y[:,lag:]
    #mu_t = np.mean(X_t, axis=1, keepdims=True)
    #mu_s = np.mean(X_s, axis=1, keepdims=True)
    #C = np.mean(X_t * X_s, axis=1, keepdims=True) - mu_t*mu_s
    # Autocorrelation
    #tau = C / np.var(y, axis=1, keepdims=True)
    #return tau

    # Original
    #y = (y - np.mean(y, axis=1, keepdims=True)) / np.var(y, axis=1, keepdims=True)
    #tau = np.sum(y[:,lag:] * y[:,:-lag], axis=1, keepdims=True)
    #return tau


def distance(x, y):
    d = np.linalg.norm( np.array(x) - np.array(y), ord=1, axis=0)
    return d

