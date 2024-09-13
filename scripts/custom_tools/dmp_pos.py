#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as Rot


""" DMP : Dynamic Movement Primitives computing class """
class DMP:
    """
    DMP Constructor
    gets
    n: number of Gaussian kernels, int
    alphay = alpha_y coefficient for point attractor, float
    """
    def __init__(self, n=20, alphay=4):
        self.n = n
        self.alphay, self.betay, self.alphax = alphay, alphay / 4, alphay / 3
        self.ygd, self.y0d, self.qgd, self.q0d = np.empty(3), np.empty(3), np.empty(4), np.empty(4)
        self.sg_p, self.sg_q = 1, 1
        self.rotm_p, self.rotm_q = np.eye(3), np.eye(4)
        self.fn_learn_p = lambda ddy, dy, y : ddy - self.alphay * (self.betay * (self.ygd - y) - dy)
        self.fn_learn_q = lambda ddeq, deq, eq : ddeq + self.alphay * (self.betay * eq + deq)
        self.fit_ddyp = lambda dyp, yp, yg, fn : self.alphay * (self.betay * (yg - yp) - dyp) + fn
        self.fit_ddeq = lambda deq, eq, fn : (-1 * self.alphay * (self.betay * eq + deq)) + fn

    """
    __can_sys: Canonical system function
    gets
    t: time vector, NumPy float array (m, 1) (m = number of datapoints)
    tau: Tau time scaling factor, float
    returns
    x: canonical system, NumPy float array (m)
    psii: Gaussian kernels matrix, NumPy float array (n, m)
    """
    def __can_sys(self, t, tau):
        x = np.exp(-self.alphax * t / tau)
        if len(t) > 1:
            fac = np.floor(np.linspace(0, len(x) - 1, self.n)).astype(int)
            self.ci = np.take(x, fac)
            self.hi = self.n / np.power(self.ci, 2)
        psii = np.empty([self.n, len(x)], dtype=float)
        for i in range(self.n):
            psii[i] = np.exp(-1 * np.inner(self.hi[i], np.power(x - self.ci[i], 2)))
        return x, psii

    """
    __get_weights: Weight estimation from LWR function
    gets
    t: time vector, NumPy float array (m, 1)
    fd: forcing function, NumPy float array (m, dim)
    returns
    wi: weight matrix, NumPy float array (n, dim)
    x: canonical system, NumPy float array (m)
    """
    def __get_weights(self, t, fd):
        self.alphax = (self.alphay / 3) * (5 / t[-1])
        x, psii = self.__can_sys(t, 1)
        s = x
        wi = np.empty([self.n, fd.shape[1]])
        for i in range(self.n):
            psim = np.diag(psii[i])
            wi[i] = np.dot(np.dot(np.transpose(s), psim), fd) / np.dot(np.dot(np.transpose(s), psim), s)
        return wi, x

    """
    __get_scale_p: Cartesian scaling factor computing function 
    gets
    yg: Cartesian goal, NumPy float array (3)
    y0: Cartesian initial, NumPy float array (3)
    returns
    sg: Cartesian scaling factor, float
    """
    def __get_scale_p(self, yg, y0):
        sg = np.linalg.norm(yg - y0) / np.linalg.norm(self.ygd - self.y0d)
        return sg

    """
    __get_scale_q: Quaternion scaling factor computing function 
    gets
    qg: Cartesian goal position, NumPy float array (4)
    q0: Cartesian initial position, NumPy float array (1, 4)
    returns
    sg: Quaternion scaling factor, float
    """
    def __get_scale_q(self, qg, q0):
        sg = np.linalg.norm(qlog(qprod(qg, qconj(q0)))) / np.linalg.norm(qlog(qprod(self.qgd, qconj(self.q0d))))
        return sg

    """
    __get_rotm_p: Cartesian rotation matrix computing function 
    gets
    yg: Cartesian goal position, NumPy float array (3)
    y0: Cartesian initial position, NumPy float array (3)
    returns
    rotm: Cartesian rotation matrix, NumPy float array (3, 3)
    """
    def __get_rotm_p(self, yg, y0):
        nc = (yg - y0) / np.linalg.norm(yg - y0)
        nd = (self.ygd - self.y0d) / np.linalg.norm(self.ygd - self.y0d)
        # Free task (Koutras et al., 2020)
        k = np.cross(nd, nc)
        k = np.array([0, -k[2], k[1], k[2], 0, -k[0], -k[1], k[0], 0]).reshape((3, 3))
        val = np.clip(np.inner(np.transpose(nd), nc), -1, 1)
        theta = np.arccos(val)
        rotm = np.eye(3) + k * np.sin(theta) + k * k * (1 - np.cos(theta))
        # Classic scaling (Ijspeert et al., 2013)
        # rotm = np.diag(g - y0)
        # sg = 1
        return rotm

    """
    __get_rotm_q: Quaternion rotation matrix computing function 
    gets
    qg: Quaternion goal position, NumPy float array (4)
    q0: Quaternion initial position, NumPy float array (1, 4)
    returns
    rotm: Quaternion rotation matrix, NumPy float array (3, 3)
    """
    def __get_rotm_q(self, qg, q0):
        nc = 2 * qlog(qprod(qg, qconj(q0)))[0, 1:]
        nc = nc / np.linalg.norm(nc)
        nd = 2 * qlog(qprod(self.qgd, qconj(self.q0d)))[0, 1:]
        nd = nd / np.linalg.norm(nd)
        # Free task (Koutras et al., 2020)
        k = np.cross(nd, nc)
        k = np.array([0, -k[2], k[1], k[2], 0, -k[0], -k[1], k[0], 0]).reshape((3, 3))
        val = np.clip(np.inner(np.transpose(nd), nc), -1, 1)
        theta = np.arccos(val)
        rotm = np.eye(4)
        rotm[1:, 1:] = np.eye(3) + k * np.sin(theta) + k * k * (1 - np.cos(theta))
        # Classic scaling (Ijspeert et al., 2013)
        # sg = 1
        # rotm = np.diag(eq)
        return rotm

    """
    get_model_p: Cartesian DMP model computing function 
    gets
    t: time vector, NumPy float array (m, 1)
    y: Cartesian positions vector, NumPy float array (m, 3)
    returns
    wi: weight matrix, NumPy float array (n, 3)
    x: canonical system, NumPy float array (m)
    """
    def get_model_p(self, t, y):
        dy = np.divide(np.diff(y, axis=0), np.diff(t).reshape((-1, 1)))
        dy = np.append(dy, [dy[-1, :]], axis=0)
        ddy = np.divide(np.diff(dy, axis=0), np.diff(t).reshape((-1, 1)))
        ddy = np.append(ddy, [ddy[-1, :]], axis=0)

        self.t0 = t[0]
        self.ygd = y[-1, :]
        self.y0d = y[0, :]

        fd = self.fn_learn_p(ddy, dy, y)
        # fd = np.inner(fd, np.linalg.inv(np.diag(self.gd - self.y0d)))

        wi, x = self.__get_weights(t, fd)
        return wi, x

    """
    get_model_q: Quaternion DMP model computing function 
    gets
    t: time vector, NumPy float array (m, 1)
    q: Quaternion orientation vector, NumPy float array (m, 4)
    returns
    wi: weight matrix, NumPy float array (n, 4)
    x: canonical system, NumPy float array (m)
    """
    def get_model_q(self, t, q):
        self.qgd = q[-1, :]
        self.q0d = np.reshape(q[0, :], (1, 4))

        eq = 2 * qlog(qprod(self.qgd, qconj(q)))
        deq = np.divide(np.diff(eq, axis=0), np.diff(t).reshape((-1, 1)))
        deq = np.append(deq, [deq[-1, :]], axis=0)
        ddeq = np.divide(np.diff(deq, axis=0), np.diff(t).reshape((-1, 1)))
        ddeq = np.append(ddeq, [ddeq[-1, :]], axis=0)

        fd = self.fn_learn_q(ddeq, deq,eq)
        # fd = np.inner(fd, np.linalg.pinv(np.diag(eq[0, :])))

        wi, x = self.__get_weights(t, fd)
        return wi, x

    """
    fit_model_p: Cartesian DMP model batch fitting function 
    gets
    t: time vector, NumPy float array (m, 1)
    tau: Tau time scaling factor, float
    yg: Cartesian goal position, NumPy float array (3)
    y0: Cartesian initial position, NumPy float array (3)
    dy0: Cartesian initial velocity, NumPy float array (3)
    wi: weight matrix, NumPy float array (n, 3)
    returns
    t: time vector, NumPy float array (m * tau, 1)
    yf: Cartesian final [y, dy, ddy], NumPy float array (m * tau, 9)
    """
    def fit_model_p(self, t, tau, yg, y0, dy0, wi):
        lim = t[-1] * tau
        t = np.linspace(t[0], lim, 1 + len(t)*tau)
        dt = np.mean(np.diff(t))
        x, psii = self.__can_sys(t, tau)
        fn = np.multiply(x, np.inner(np.transpose(wi), np.transpose(psii))) / np.sum(psii, axis=0)
        fn = np.transpose(fn)

        sg = self.__get_scale_p(yg, y0)
        rotm = self.__get_rotm_p(yg, y0)

        yp = y0
        dyp = sg * np.inner(dy0, rotm)
        yf = np.empty([len(t), 9])
        for i in range(len(t)):
            ddyp = self.fit_ddyp(dyp, yp, yg, (sg * np.inner(fn[i, :], rotm)))
            yf[i, :] = np.array([yp.reshape(3), dyp.reshape(3), ddyp.reshape(3)]).reshape((1, -1))
            dyp = dyp + ddyp * dt / tau
            yp = yp + dyp * dt / tau
        return t, yf

    """
    fit_model_q: Quaternion DMP model batch fitting function 
    gets
    t: time vector, NumPy float array (m, 1)
    tau: Tau time scaling factor, float
    qg: Quaternion goal position, NumPy float array (1, 4)
    q0: Quaternion initial position, NumPy float array (4)
    deq0: Quaternion initial velocity error, NumPy float array (4)
    wi: weight matrix, NumPy float array (n, 4)
    returns
    t: time vector, NumPy float array (m * tau, 1)
    qf: Quaternion final [q, deq, ddeq], NumPy float array (m * tau, 12)
    """
    def fit_model_q(self, t, tau, qg, q0, deq0, wi):
        lim = t[-1] * tau
        t = np.linspace(t[0], lim, 1 + len(t) * tau)
        dt = np.mean(np.diff(t))
        x, psii = self.__can_sys(t, tau)
        fn = np.multiply(x, np.inner(np.transpose(wi), np.transpose(psii))) / np.sum(psii, axis=0)
        fn = np.transpose(fn)

        sg = self.__get_scale_q(qg, q0)
        rotm = self.__get_rotm_q(qg, q0)

        q = q0
        eq = 2 * qlog(qprod(qg, qconj(q0)))[0, :]
        deq = sg * np.inner(deq0, rotm)
        qf = np.empty([len(t), 12])
        for i in range(len(t)):
            ddeq = self.fit_ddeq(deq, eq, (sg * np.inner(fn[i, :], rotm)))
            qf[i, :] = np.array([q.reshape(4), deq.reshape(4), ddeq.reshape(4)]).reshape((1, -1))
            deq = deq + ddeq * dt / tau
            eq = eq + deq * dt / tau
            q = np.array(.5 * eq).reshape((1, 4))
            q = qprod(qconj(qexp(q)).reshape(4), qg.reshape((1, 4)))
        return t, qf

    """
    step_prefit_p: Cartesian DMP model step pre-fitting function 
    gets
    yg: Cartesian goal position, NumPy float array (3)
    y0: Cartesian initial position, NumPy float array (3)
    dy0: Cartesian initial velocity, NumPy float array (3)
    returns
    self.sg_p * np.inner(dy0, self.rotm_p): scaled and rotated dy0, NumPy float array (3)
    """
    def step_prefit_p(self, yg, y0, dy0):
        self.yg = yg
        self.sg_p = self.__get_scale_p(yg, y0)
        self.rotm_p = self.__get_rotm_p(yg, y0)
        return self.sg_p * np.inner(dy0, self.rotm_p)

    """
    step_prefit_q: Quaternion DMP model step pre-fitting function 
    gets
    qg: Quaternion goal position, NumPy float array (4)
    q0: Quaternion initial position, NumPy float array (1, 4)
    deq0: Quaternion initial velocity error, NumPy float array (4)
    returns
    self.sg_q * np.inner(deq0, self.rotm_q): scaled and rotated deq0, NumPy float array (4)
    """
    def step_prefit_q(self, qg, q0, deq0):
        self.qg = qg
        self.eq = 2 * qlog(qprod(qg, qconj(q0)))[0, :]
        self.sg_q = self.__get_scale_q(qg, q0)
        self.rotm_q = self.__get_rotm_q(qg, q0)
        return self.sg_q * np.inner(deq0, self.rotm_q)

    """
    step_fit_p: Cartesian DMP model step fitting function 
    gets
    t: time vector, NumPy float array (1, 1)
    dt: time step, float
    tau: Tau time scaling factor, float
    y_p: Cartesian previous data [y, dy, ddy], NumPy float array (1, 9)
    wi: weight matrix, NumPy float array (n, 3)
    returns
    x: canonical system, NumPy float array (1)
    psii: Gaussian kernels matrix, NumPy float array (n, 1)
    fn: Cartesian current forcing function values, NumPy float array (1, 3)
    yf: Cartesian current data [y, dy, ddy], NumPy float array (1, 9)
    """
    def step_fit_p(self, t, dt, tau, y_p, wi):
        x, psii = self.__can_sys(t, tau)
        fn = np.multiply(x, np.inner(np.transpose(wi), np.transpose(psii))) / np.sum(psii, axis=0)

        yp, dyp = y_p[:, 0:3], y_p[:, 3:6]
        ddyp = self.fit_ddyp(dyp, yp, self.yg, (self.sg_p * np.inner(fn.T, self.rotm_p)))
        dyp = dyp + ddyp * dt / tau
        yp = yp + dyp * dt / tau
        yf = np.array([yp.reshape(3), dyp.reshape(3), ddyp.reshape(3)]).reshape((1, -1))
        return x, psii, fn.T, yf

    """
    step_fit_q: Quaternion DMP model step fitting function 
    gets
    t: time vector, NumPy float array (1, 1)
    dt: time step, float
    tau: Tau time scaling factor, float
    q_p: Quaternion previous data [q, deq, ddeq], NumPy float array (1, 12)
    wi: weight matrix, NumPy float array (n, 4)
    returns
    x: canonical system, NumPy float array (1)
    psii: Gaussian kernels matrix, NumPy float array (n, 1)
    fn: Quaternion current forcing function values, NumPy float array (1, 4)
    qf: Quaternion current data [y, dy, ddy], NumPy float array (1, 6)
    """
    def step_fit_q(self, t, dt, tau, q_p, wi):
        x, psii = self.__can_sys(t, tau)
        fn = np.multiply(x, np.inner(np.transpose(wi), np.transpose(psii))) / np.sum(psii, axis=0)

        q, deq = q_p[:, 0:4], q_p[:, 4:8]
        ddeq = self.fit_ddeq(deq, self.eq, (self.sg_q * np.inner(fn.T, self.rotm_q)))
        deq = deq + ddeq * dt / tau
        self.eq = self.eq + deq * dt / tau
        q = np.array(.5 * self.eq).reshape((1, 4))
        q = qprod(qconj(qexp(q)).reshape(4), self.qg.reshape((1, 4)))
        qf = np.array([q.reshape(4), deq.reshape(4), ddeq.reshape(4)]).reshape((1, -1))
        return x, psii, fn.T, qf

    """
    step_prefit: Pose DMP model step pre-fitting function 
    gets
    yg: Cartesian goal position, NumPy float array (3)
    y0: Cartesian initial position, NumPy float array (3)
    dy0: Cartesian initial velocity, NumPy float array (3)
    qg: Quaternion goal position, NumPy float array (4)
    q0: Quaternion initial position, NumPy float array (1, 4)
    deq0: Quaternion initial velocity error, NumPy float array (4)
    returns
    dy0: scaled and rotated dy0, NumPy float array (4)
    deq0: scaled and rotated deq0, NumPy float array (4)
    """
    def step_prefit(self, yg, y0, dy0, qg, q0, deq0):
        dy0 = self.step_prefit_p(yg, y0, dy0)
        deq0 = self.step_prefit_q(qg, q0, deq0)
        return dy0, deq0

    """
    step_fit: Pose DMP model step fitting function 
    gets
    t: time vector, NumPy float array (1, 1)
    dt: time step, float
    tau: Tau time scaling factor, float
    y_p: Cartesian previous data [y, dy, ddy], NumPy float array (1, 9)
    wi_p: Cartesian weight matrix, NumPy float array (n, 3)
    q_p: Quaternion previous data [q, deq, ddeq], NumPy float array (1, 12)
    wi_q: Quaternion weight matrix, NumPy float array (n, 4)
    returns
    x: canonical system, NumPy float array (1)
    psii: Gaussian kernels matrix, NumPy float array (n, 1)
    fn_p: Cartesian current forcing function values, NumPy float array (1, 3)
    fn_q: Quaternion current forcing function values, NumPy float array (1, 4)
    yf: Cartesian current data [y, dy, ddy], NumPy float array (1, 9)
    qf: Quaternion current data [y, dy, ddy], NumPy float array (1, 6)
    """
    def step_fit(self, t, dt, tau, y_p, wi_p, q_p, wi_q):
        x, psii = self.__can_sys(t, tau)
        fn_p = np.multiply(x, np.inner(np.transpose(wi_p), np.transpose(psii))) / np.sum(psii, axis=0)
        fn_q = np.multiply(x, np.inner(np.transpose(wi_q), np.transpose(psii))) / np.sum(psii, axis=0)

        yp, dyp = y_p[:, 0:3], y_p[:, 3:6]
        ddyp = self.fit_ddyp(dyp, yp, self.yg, (self.sg_p * np.inner(fn_p.T, self.rotm_p)))
        dyp = dyp + ddyp * dt / tau
        yp = yp + dyp * dt / tau
        yf = np.array([yp.reshape(3), dyp.reshape(3), ddyp.reshape(3)]).reshape((1, -1))

        q, deq = q_p[:, 0:4], q_p[:, 4:8]
        ddeq = self.fit_ddeq(deq, self.eq, (self.sg_q * np.inner(fn_q.T, self.rotm_q)))
        deq = deq + ddeq * dt / tau
        self.eq = self.eq + deq * dt / tau
        q = np.array(.5 * self.eq).reshape((1, 4))
        q = qprod(qconj(qexp(q)).reshape(4), self.qg.reshape((1, 4)))
        qf = np.array([q.reshape(4), deq.reshape(4), ddeq.reshape(4)]).reshape((1, -1))

        return x, psii, fn_p.T, fn_q.T, yf, qf


"""
qlog: Quaternion logarithm function 
gets
qarr: Quaternion array, NumPy float array (m, 4) (m = size of Quaternions list)
returns
qres: Quaternion result array, NumPy float array (m, 4)
"""
def qlog(qarr):
    qres = np.empty(qarr.shape)
    for i in range(qres.shape[0]):
        u = qarr[i, 1:]
        v = qarr[i, 0]
        nu = np.linalg.norm(u)
        if nu == 0:
            qres[i, :] = np.array([0, 0, 0, 0])
        else:
            qres[i, :] = np.append([0], [np.arccos(v) * u / nu]).reshape((1, 4))
    return qres


"""
qexp: Quaternion exponential function 
gets
qarr: Quaternion array, NumPy float array (m, 4) (m = size of Quaternions list)
returns
qres: Quaternion result array, NumPy float array (m, 4)
"""
def qexp(qarr):
    qres = np.empty(qarr.shape)
    for i in range(qres.shape[0]):
        v = qarr[i, 1:]
        nv = np.linalg.norm(v)
        if nv == 0:
            qres[i, :] = np.array([1, 0, 0, 0]).reshape((1, 4))
        else:
            qres[i, :] = np.append([np.cos(nv)], [np.sin(nv) * v / nv]).reshape((1, 4))
    return qres


"""
qprod: Quaternion product function 
gets
q1: Quaternion common factor, NumPy float array (4)
qc: Quaternion list, NumPy float array (m, 4)
returns
qres: Quaternion result array, NumPy float array (m, 4)
"""
def qprod(q1, qc):
    qres = np.empty(qc.shape)
    for i in range(qres.shape[0]):
        q2 = qc[i, :]
        w = (q1[0] * q2[0]) - np.inner(q1[1:], np.transpose(q2[1:]))
        v = (q1[0] * q2[1:]) + (q2[0] * q1[1:]) + np.cross(q1[1:], q2[1:])
        qres[i, :] = np.append([w], [v]).reshape((1, 4))
    return qres


"""
qconj: Quaternion conjugate function 
gets
qarr: Quaternion array, NumPy float array (m, 4) (m = size of Quaternions list)
returns
qres: Quaternion result array, NumPy float array (m, 4)
"""
def qconj(qarr):
    qres = np.empty(qarr.shape)
    for i in range(qres.shape[0]):
        q = qarr[i, :]
        qres[i, :] = np.append([q[0]], [-1 * q[1:]]).reshape((1, 4))
    return qres


def main():
    print "This is a Cartesian DMP example"
    """ Cartesian trajectory data """
    dt = .01
    lim = 2.0
    t = np.linspace(0.0, lim, int(lim / dt)).reshape(-1)
    yin = np.empty((t.shape[0], 3))
    for i, it in enumerate(t):
        x = .10 + .50 * it**2
        y = .50 + .05 * it
        z = .50 + .50 * ((t[-1]-it) / t[-1])**2
        """
        x = .50 * it - .25 * np.sin(np.pi * (1.0 * it + .5))
        y = .50 * it - .25 * np.sin(np.pi * (2.0 * it + .5))
        z = .50 * it - .25 * np.sin(np.pi * (3.0 * it + .5))
        """
        yin[i, :] = np.array([x, y, z]).reshape(3)
    y0 = yin[0, :]
    dy0 = np.diff(yin[0:2, :], axis=0) / dt
    yg = np.multiply(yin[-1, :], [1.0, 1.0, 1.0])

    """ DMP object and parameters """
    n = 20
    alphay = 4
    dmp_obj = DMP(n, alphay)
    wi, x = dmp_obj.get_model_p(t, yin)

    """ Online approach example """
    _t = np.linspace(0.0, 4.0, int(4.0 / dt)).reshape(-1, 1) # t[0:np.floor(len(t) / 2).astype(int)] + .75
    _x = np.empty([len(_t)], dtype=float)
    fn = np.empty([len(_t), 3], dtype=float)
    psi = np.empty([len(_t), n], dtype=float)
    yf = np.empty([len(_t), 9], dtype=float)
    dy0 = dmp_obj.step_prefit_p(yg, y0, dy0)
    yd = np.array([y0.reshape(3), dy0.reshape(3), np.zeros(3)]).reshape((1, -1))
    for i, ti in enumerate(_t):
        _x[i], dpsii, fn[i, :], yd = dmp_obj.step_fit_p(ti, dt, 1, yd, wi)
        psi[i, :] = dpsii.reshape(-1)
        yf[i, :] = yd
    """ Online results plots """
    for i in range(wi.shape[1]):
        for j in range(n):
            plt.plot(_t, np.multiply(psi[:, j] * wi[j, i], _x), ':')
        plt.plot(_t, fn[:, i])
        plt.show()
    plt.plot(_t, yf[:, 0:3])
    plt.show()

    """ Offline approach example """
    wi, x = dmp_obj.get_model_p(t, yin)
    tn, yf = dmp_obj.fit_model_p(t, 1, yg, y0, dy0, wi)
    """ Offline results plots """
    plt.plot(tn, yf[:, 0:3])
    plt.plot(t, yin[:, 0:3], '--')
    plt.show()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(yf[:, 0], yf[:, 1], yf[:, 2])
    ax.plot(yin[:, 0], yin[:, 1], yin[:, 2], '--')
    plt.show()

    print "This is a Quaternion DMP example"
    """ Quaternion orientation data """
    ein = np.empty((t.shape[0], 3))
    for i, it in enumerate(t):
        ex = it * np.pi/t[-1] * .25
        ey = it * np.pi/t[-1] * .50
        ez = .00
        """
        ex = .30 * it - .25 * np.sin(np.pi * (1.5 * it + .5))
        ey = .50 * it - .25 * np.sin(np.pi * (2.0 * it + .5))
        ez = .70 * it - .25 * np.sin(np.pi * (2.5 * it + .5))
        """
        ein[i, :] = np.array([ex, ey, ez]).reshape(3)
    qin = Rot.from_euler('zxy', ein, degrees=True).as_quat()

    """ Offline approach example """
    wi, x = dmp_obj.get_model_q(t, qin)

    q0 = qin[0, :].reshape((1, 4))
    eq = 2 * qlog(qprod(qin[-1, :], qconj(qin)))
    deq0 = np.diff(eq[0:2, :], axis=0) / dt
    qg = qin[-1, :]  # Rot.from_euler('zyx', [1, 2, 3], degrees=True).as_quat()
    tn, qf = dmp_obj.fit_model_q(t, 1, qg, q0, deq0, wi)
    """ Offline results plots """
    fig, axs = plt.subplots(4)
    for i in range(4):
        axs[i].plot(tn, qf[:, i])
        axs[i].plot(t, qin[:, i], '--')
    plt.show()
    eout = Rot.from_quat(qf[:, 0:4]).as_euler('zxy', degrees=True)
    plt.plot(tn, eout)
    plt.plot(t, ein, '--')
    plt.show()


if __name__ == "__main__":
    main()
