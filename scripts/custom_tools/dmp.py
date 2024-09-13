#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


def get_model(t, y, n = 20, alphay = 4):
    dt = np.mean(np.diff(t))
    dy = np.diff(y)/dt
    dy = np.append(dy, dy[len(dy)-1])
    ddy = np.diff(dy)/dt
    ddy = np.append(ddy, ddy[len(ddy)-1])

    betay = alphay/4
    alphax = alphay/3 * t[len(t)-1]/5

    g = y[len(y)-1]
    fd = ddy - alphay*(betay*(g-y)-dy)

    x = np.exp(-alphax*t)
    fac = np.floor(len(t)/(n-1))
    vfac = np.arange(0,len(t),fac).astype(int)
    ci = np.take(x, vfac)
    hi = n/np.power(ci, 2)

    # tc = np.take(t, vfac)
    # plt.plot(t,x)
    # plt.plot(tc,ci,'ro')
    # plt.show()

    psii = np.empty([n,len(x)], dtype=float)
    for i in range(n):
        psii[i] = np.exp(-1 * np.inner(hi[i], np.power(x-ci[i], 2)))
        # plt.plot(t,psii[i])
    # plt.plot(tc, np.ones(tc.shape), 'ro')
    # plt.show()

    s = np.inner(x, g-y[0])
    wi = np.empty([n, 1])
    fn = np.empty(psii.shape)
    for i in range(n):
        psim = np.diag(psii[i])
        wi[i] = (np.dot(np.dot(np.transpose(s), psim), fd))/(np.dot(np.dot(np.transpose(s), psim), s))
        fn[i] = psii[i]*wi[i]
        # plt.plot(t, fn[i], ':')

    fn = np.sum(fn, axis=0)/np.sum(psii,axis=0)

    # plt.plot(t, fn)
    # plt.show()

    return fn, x, dt
