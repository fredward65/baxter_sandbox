#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import quaternion as quat
from dual_quaternions import DualQuaternion


class DQDMP(object):
    """
    A class for computing Dynamic Movement Primitives (DMP)
    from Dual Quaternion pose data
    """

    def __init__(self, n=20, alpha_y=4):
        """
        DQ-DMP Object Constructor

        Parameters
        ----------
        n : int
            Number of Gaussian kernels in Psi
        alpha_y : float
            alpha_y coefficient
        """
        # Previous time to compute timestep
        self.prev_t = 0.0
        # Critically dampened point attractor system parameters
        self.alpha_y = alpha_y
        self.beta_y = self.alpha_y / 4
        self.alpha_x = self.alpha_y / 3
        # Number of Gaussian kernels
        self.n = n
        # Weights, centers and widths of Gaussian kernels
        self.w = np.empty((8, self.n))
        self.c_i = np.nan
        self.h_i = np.nan
        # Initial and goal dynamic conditions
        self.dq0d = np.nan
        self.dqgd = np.nan
        # Scaling factor parameters
        self.nv = np.nan
        self.ytd = np.nan
        # Optional normal plane vectors
        self.pg_p = np.nan
        # Second order systems
        self.fit_dtw = lambda tw, edq, fn: self.alpha_y * (np.multiply(self.beta_y, edq) - tw) + fn
        # Forcing function learning
        self.fn_learn = lambda dtw, tw, edq: dtw - self.alpha_y * (self.beta_y * edq - tw)
        # Forcing term reconstruction function
        self.fn_rct = lambda x, psi, w_i: (x * np.inner(psi, w_i)) / np.sum(psi, 1).reshape((-1, 1))

    @staticmethod
    def __log_dq(dq):
        """
        Dual Quaternion Logarithm

        Parameters
        ----------
        dq : DualQuaternion
            Dual Quaternion

        Returns
        -------
        log : numpy.ndarray
            Dual Quaternion Logarithm
        """
        dq = dq.dq_array()
        p = quat.from_float_array(dq[0:4])
        d = quat.from_float_array(dq[4:8])
        yp = np.log(p)
        yd = d * p.inverse()
        log = np.append(quat.as_float_array(yp), quat.as_float_array(yd))
        return log

    @staticmethod
    def __exp_dq(dq):
        """
        Dual Quaternion Exponential

        Parameters
        ----------
        dq : DualQuaternion
            Dual Quaternion

        Returns
        -------
        exp : numpy.ndarray
            Dual Quaternion Exponential
        """
        p = quat.from_float_array(dq[0:4])
        d = quat.from_float_array(dq[4:8])
        yp = np.exp(p)
        yd = d * yp
        exp = np.append(quat.as_float_array(yp), quat.as_float_array(yd))
        return exp

    @staticmethod
    def __dx_dt(t, x):
        """
        Differentiates x w.r.t. t

        Parameters
        ----------

        t : numpy.ndarray
            Time vector, (m)
        x : numpy.ndarray
            Data vector, (m, dim)

        Returns
        -------
        dx : numpy.ndarray
            Differentiated data vector, (m, dim)
        """
        # Timestep vector
        dt = np.diff(t, axis=0).reshape((-1, 1))
        # Differentiation
        dx = np.divide(np.diff(x, axis=0), dt)
        dx = np.append(dx, [dx[-1, :]], axis=0)
        return dx

    @staticmethod
    def __edq_from_pose(dq, dqg):
        """
        Dual Quaternion Error w.r.t Goal

        Parameters
        ----------
        dq : list
            DualQuaternion list

        Returns
        -------
        edq : list
            Dual Quaternion Error list
        """
        edq = []
        for dqi in dq:
            res = dqi.quaternion_conjugate() * dqg
            edq.append(DQDMP.__log_dq(res))
        edq = np.array(edq).reshape((-1, 8))
        return edq

    @staticmethod
    def __pose_from_edq(edq, dqg):
        """
        Dual Quaternion from Error w.r.t Goal

        Parameters
        ----------
        edq : list
            Dual Quaternion Error list

        Returns
        -------
        dq : list
            DualQuaternion list
        """
        dq = []
        for edqi in edq:
            res = edqi.quaternion_conjugate() * dqg
            dq.append(res)
        return dq

    @staticmethod
    def __twist_from_edq(t, edq):
        """
        Twist list from Dual Quaternion Error list

        Parameters
        ----------
        t : numpy.ndarray
            Time array
        edq : list
            Dual Quaternion Error list

        Returns
        -------
        tw : numpy.ndarray
            Twist list
        """
        r = []
        p = []
        for dqi in edq:
            dqd = DualQuaternion.from_dq_array(dqi).quat_pose_array()
            r.append(dqd[0:4])
            p.append(dqd[4:7])
        p = np.array(p).reshape((-1, 3))
        r = np.array(r).reshape((-1, 4))
        dr = DQDMP.__dx_dt(t, r)
        dp = DQDMP.__dx_dt(t, p)
        omega = []
        for (xr, xdr) in zip(r, dr):
            ri = quat.from_float_array(xr)
            dri = quat.from_float_array(xdr)
            tmp = 2 * dri * ri.conj()
            omega.append(quat.as_float_array(tmp))
        omega = np.array(omega).reshape((-1, 4))
        w_ = omega[:, 1:4]
        tw = []
        for (pi, vi, wi) in zip(p, dp, w_):
            d_ = np.cross(pi, wi) + vi
            dual = np.append(0., d_)
            prime = np.append(0., wi)
            tw.append(np.append(prime, dual))
        tw = np.array(tw).reshape((-1, 8))
        return tw

    @staticmethod
    def twist_from_dq(t, dq):
        """
        Twist list from Dual Quaternion list

        Parameters
        ----------
        t : numpy.ndarray
            Time array
        dq : list
            Dual Quaternion list

        Returns
        -------
        tw : numpy.ndarray
            Twist list
        """
        tw = []
        for dqi, pdq, ti, pti in zip(dq[1:], dq[0:-1], t[1:], t[0:-1]):
            dt = ti - pti
            tw_ = 2 * DQDMP.__log_dq(pdq.quaternion_conjugate() * dqi) / dt
            tw.append(tw_)
        tw.append(tw[-1])
        return np.array(tw).reshape((-1, 8))

    @staticmethod
    def pose_from_dq(dq):
        """
        Pose Data from Dual Quaternion list

        Parameters
        ----------
        dq : list
            Dual Quaternion list

        Returns
        -------
        r : numpy.ndarray
            Quaternion rotation data
        p : numpy.ndarray
            Cartesian position data
        """
        r = []
        p = []
        for dqi in dq:
            pt = dqi.quat_pose_array()
            r.append(pt[0:4])
            p.append(pt[4:7])
        r = np.array(r).reshape((-1, 4))
        p = np.array(p).reshape((-1, 3))
        return r, p

    def reset_t(self, t=0.0):
        """
        Reset Starting Time

        Parameters
        ----------
        t : float
            Starting time value (optional)
        """
        self.prev_t = t

    def __set_cfc(self, tg):
        """
        Computes coefficients for canonical system,
        centers and widths for Gaussian Kernels

        Parameters
        ----------
        tg : float
            Goal time
        """
        # Coefficient for canonical system adjusted to time period
        self.alpha_x = (self.alpha_y / 3) * (5 / tg)
        # Centers and Widths of Gaussian kernels
        self.c_i = np.exp(-self.alpha_x * ((np.linspace(1, self.n, self.n) - 1) / (self.n - 1)) * tg)
        self.h_i = self.n / np.power(self.c_i, 2)

    def __can_sys(self, t, tau=1):
        """
        Computes canonical system and Gaussian kernels

        Parameters
        ----------
        t : numpy.ndarray
            Time vector, (m)
        tau : float
            Time scaling variable Tau

        Returns
        -------
        x : numpy.ndarray
            Canonical system x, (m, 1)
        psi_i : numpy.ndarray
            Gaussian kernels Psi, (m, n)
        """
        # Canonical system
        x = np.exp(-self.alpha_x * t / tau).reshape((-1, 1))
        # Gaussian kernels
        psi_i = np.empty([len(x), self.n], dtype=float)
        for i in range(self.n):
            psi_i[:, i] = np.exp(-1 * np.inner(self.h_i[i], np.power(x - self.c_i[i], 2))).reshape(-1)
        return x, psi_i

    def __w_learn(self, x, psi_i, fd):
        """
        Gaussian kernel weights learning function

        Parameters
        ----------
        x : numpy.ndarray
            Canonical system data
        psi_i : numpy.ndarray
            Gaussian kernel data
        fd : numpy.ndarray
            Forcing term data

        Returns
        -------
        wi_i : numpy.ndarray
            Gaussian kernels weights w_i, (3, n)
        """
        # Compute weights
        w_i = np.empty([fd.shape[1], self.n])
        for i in range(self.n):
            psi_m = np.diag(psi_i[:, i])
            w_i[:, i] = np.dot(np.dot(x.T, psi_m), fd) / np.dot(np.dot(x.T, psi_m), x)
        return w_i

    def train_model(self, t, dq):
        """
        Get DMP Model from Dual Quaternion pose list

        Parameters
        ----------
        t : numpy.ndarray
            Time vector
        dq : list
            Dual Quaternion pose data, (m, DualQuaternion)

        Returns
        -------
        wi_i : numpy.ndarray
            Gaussian kernels weights w_i, (3, n)
        x : numpy.ndarray
            Canonical system x, (m, 1)
        """
        # Coefficient for canonical system, Centers and Widths of Gaussian kernels
        self.__set_cfc(t[-1]-t[0])
        # Compute Canonical system and Gaussian kernels
        x, psi_i = self.__can_sys(t)
        # Store training initial and goal conditions
        self.dq0d = dq[0]
        self.dqgd = dq[-1]
        # Time derivatives from q
        edq = self.__edq_from_pose(dq, self.dqgd)
        tw = self.twist_from_dq(t, dq)
        dtw = self.__dx_dt(t, tw)
        # dedq = self.__dx_dt(t, edq)
        # ddedq = self.__dx_dt(t, dedq)
        # Store demonstrated conditions for scaling
        # n_q = np.linalg.norm(quat.as_float_array(q), axis=1)
        # self.nv_q = np.sum((n_q - np.mean(n_q)) ** 2) / (len(n_q) - 1)
        # Forcing term from captured data
        fd = self.fn_learn(dtw, tw, edq)
        # Weight learning
        w_i = self.__w_learn(x, psi_i, fd)
        self.w = w_i
        return w_i, x

    def fit_model(self, t, dq0, tw0, dqg, tau=1):
        """
        Fit DMP Model to Dual Quaternion conditions

        Parameters
        ----------
        t : numpy.ndarray
            Time vector
        dq0 : DualQuaternion
            Initial Dual Quaternion pose
        tw0 : numpy.ndarray
            Initial Twist
        dqg : DualQuaternion
            Goal Dual Quaternion pose
        tau : float
            Time scaling variable Tau

        Returns
        -------
        dq_arr : numpy.ndarray
            Reconstructed Dual Quaternion pose
        """
        # Timestep vector
        dt = np.diff(t)
        dt = np.append(dt, dt[-1])
        # Recompute Canonical system
        x, psi_i = self.__can_sys(t, tau)
        # Reconstruct forcing term
        fn = self.fn_rct(x, psi_i, self.w)
        # Apply scaling factor
        # sg_q = self.__get_sf_q(q0, qg)
        # fn = np.dot(fn, sg_q)
        # Initial conditions
        dq = dq0
        tw = tw0
        # Reconstruct orientations
        dq_arr = []
        for (fni, dti) in zip(fn, dt):
            dq_arr.append(dq)
            edq = self.__log_dq(dq.quaternion_conjugate() * dqg)
            dtw = self.fit_dtw(tw, edq, fni)
            tw = tw + dtw * dti / tau
            dq = dq * DualQuaternion.from_dq_array(self.__exp_dq(tw * (dti/tau) / 2))
        return dq_arr

    def fit_step(self, t, dq, tw, dq0, dqg, tau=1):
        """
        Step-fit DMP Model to Dual Quaterion conditions

        Parameters
        ----------
        t : float
            Current time
        dq : DualQuaternion
            Current Dual Quaternion pose
        tw : numpy.ndarray
            Current Twist
        dq0 : DualQuaternion
            Initial Dual Quaternion pose
        dqg : DualQuaternion
            Goal Dual Quaternion pose
        tau : float
            Time scaling variable Tau

        Returns
        -------
        dq_n : DualQuaternion
            Next Dual Quaterion pose
        tw_n : numpy.ndarray
            Next Twist
        """
        # Timestep
        dt = t - self.prev_t
        # Recalculate Canonical system
        x, psi_i = self.__can_sys(np.array([t]), tau)
        # Reconstruct forcing term
        fn = self.fn_rct(x, psi_i, self.w)[0]
        # Apply scaling factor
        # sg_q = self.__get_sf_q(q0, qg)
        # fn = np.dot(fn, sg_q)
        # Reconstruct orientations
        edq = self.__log_dq(dq.quaternion_conjugate() * dqg)
        dtw = self.fit_dtw(tw, edq, fn)
        tw_n = tw + dtw * dt / tau
        dq_n = dq * DualQuaternion.from_dq_array(self.__exp_dq(tw * (dt/tau) / 2))
        # Store current time
        self.prev_t = t
        return dq_n, tw_n


def gen_data():
    # Generate training data
    t = np.linspace(0, 1, num=1000)
    x = np.cos(np.pi * (2 * t + 0)) + 1
    y = np.sin(np.pi * (2 * t + 1)) + 1  # + .005*t
    z = np.zeros(t.shape)
    tr_p = np.array([x, y, z]).T
    tr_r = quat.as_float_array(quat.from_euler_angles(tr_p))
    traj = np.c_[tr_r, tr_p]

    dqtr = []
    for tr in traj:
        dqtr.append(DualQuaternion.from_quat_pose_array(tr))

    return t, dqtr, tr_r, tr_p


def main():
    # Generate training data
    t, dqtr, tr_r, tr_p = gen_data()

    # DMP Training
    tau = 2
    n = 200
    alpha_y = 12
    dmp_obj = DQDMP(n, alpha_y)
    dmp_obj.train_model(t, dqtr)

    # DMP Reconstruction
    t2 = np.linspace(0, t[-1]*2, num=1000)
    dq0 = dqtr[0]
    dqg = dqtr[-1]
    tw0 = dmp_obj.twist_from_dq(t, dqtr)[0, :]
    # dqrs = dmp_obj.fit_model(t2, dq0, tw0, dqg, tau=tau)
    dq = dq0
    tw = tw0
    dqrs = []
    for ti in t2:
        dqrs.append(dq)
        dq, tw = dmp_obj.fit_step(ti, dq, tw, dq0, dqg, tau=tau)
    r, p = dmp_obj.pose_from_dq(dqrs)

    # Reconstruction vs Demonstration
    fig, axs = plt.subplots(2)
    axs[0].plot(t2, p, t, tr_p, '--')
    axs[1].plot(t2, r, t, tr_r, '--')
    plt.show()


if __name__ == "__main__":
    main()
