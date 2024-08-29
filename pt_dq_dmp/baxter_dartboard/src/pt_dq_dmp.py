#!/usr/bin/env python
import numpy as np
# import numpy as np
from dual_quaternions import DualQuaternion
from custom_tools.math_tools import *


class PTDQMP(object):
    """
    A class for computing Projectile Throwing Dynamic Movement Primitives (DMP)
    from full pose Dual Quaternion data
    """

    def __init__(self, n=20, alpha_y=4):
        """
        Projectile Throwing Dual Quaternion DMP Object

        :param int n: Number of Gaussian kernels in Psi
        :param float alpha_y: Dampening coefficient
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
        self.w_i = np.empty((8, self.n))
        self.c_i = np.nan
        self.h_i = np.nan
        # Initial and goal dynamic conditions
        self.dq0d = np.nan
        self.dqgd = np.nan
        self.twgd = np.nan
        # Scaling/Rotation dual quaternion
        self.dq_trf = None
        self.fn_scl = 1

    @staticmethod
    def __edq_from_dq_list(dq, dqg):
        """
        Dual Quaternion Error w.r.t. Goal from Dual Quaternion List

        :param numpy.ndarray dq: Dual Quaternion Pose List
        :param DualQuaternion dqg: Goal Pose
        :return: Dual Quaternion Pose Error List
        :rtype: numpy.ndarray
        """
        edq = np.empty(dq.shape[0], dtype=DualQuaternion)
        for i, dq_i in enumerate(dq):
            edq[i] = edq_from_dq(dq_i, dqg)
        return edq

    @staticmethod
    def __fn_rct(x, psi, w):
        """
        Forcing term reconstruction function

        :param numpy.ndarray x: Canonical System vector
        :param numpy.ndarray psi: Gaussian Kernel array
        :param numpy.ndarray w: Gausian Kernel weigths array
        :return: Forcing term
        :rtype: DualQuaternion
        """
        return DualQuaternion.from_dq_array(((x * np.inner(psi, w)) / np.sum(psi, 1)).ravel())

    @staticmethod
    def dq_from_pose(r, p):
        """
        Dual Quaternion list from Pose Data

        :param numpy.ndarray r: Quaternion parameters Orientation List (x, y, z, w)
        :param numpy.ndarray p: Cartesian Position List (x, y, z)
        :return: Dual Quaternion list
        :rtype: numpy.ndarray
        """
        dq =  np.empty(r.shape[0], dtype=DualQuaternion)
        for i, (ri, pi) in enumerate(zip(r, p)):
            dq[i] = DualQuaternion.from_quat_pose_array(np.append(ri, pi))
        return dq

    @staticmethod
    def pose_from_dq(dq):
        """
        Pose Data from Dual Quaternion list

        :param numpy.ndarray dq: Dual Quaternion List
        :return: Orientation and Position vector arrays
        :rtype: (numpy.ndarray, numpy.ndarray)
        """
        r, p = [], []
        for dqi in dq:
            pt = dqi.quat_pose_array()
            r.append(pt[0:4])
            p.append(pt[4:7])
        r = np.array(r).reshape((-1, 4))
        p = np.array(p).reshape((-1, 3))
        return r, p

    def __fit_dtw(self, tw, edq, fn):
        """
        Fit second order system

        :param DualQuaternion tw: Twist as Dual Quaternion
        :param DualQuaternion edq: Dual Quaternion Error w.r.t. Goal
        :param DualQuaternion fn: Forcing Term as Dual Quaternion
        :return: Twist Derivative w.r.t. time Dual Quaternion
        :rtype: DualQuaternion
        """
        dtw = fn + self.alpha_y * ((self.beta_y * edq) + (-1 * tw))
        return dtw

    def __fn_learn(self, dtw, tw, edq):
        """
        Forcing function learning

        :param numpy.ndarray dtw: Twist Derivative w.r.t. time DualQuaternion List
        :param numpy.ndarray tw: Twist DualQuaternion List
        :param numpy.ndarray edq: Dual Quaternion Error w.r.t. Goal DualQuaternion List
        :return: Forcing Term DualQuaternion List
        :rtype: numpy.ndarray
        """
        fn = dtw + (-1 * self.alpha_y * ((self.beta_y * edq) + (-1 * tw)))
        return fn

    def __set_cfc(self, tg):
        """
        Computes coefficients for canonical system,
        centers and widths for Gaussian Kernels.

        :param float tg: Time period to reach the goal
        """
        # Coefficient for canonical system adjusted to time period
        self.alpha_x = (self.alpha_y / 3) * (1 / tg)
        # Centers and Widths of Gaussian kernels
        self.c_i = np.exp(-self.alpha_x * ((np.linspace(1, self.n, self.n) - 1) / (self.n - 1)) * tg)
        self.h_i = self.n / np.power(self.c_i, 2)

    def __can_sys(self, t, tau=1):
        """
        Computes Canonical System and Gaussian Kernels

        :param numpy.ndarray t: Time vector, (m)
        :param float tau: Time scaling parameter
        :return: Canonical System vector and Gaussian Kernel array
        :rtype: (numpy.ndarray, numpy.ndarray)
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
        Gaussian Kernel weights learning function

        :param numpy.ndarray x: Canonical System vector
        :param numpy.ndarray psi_i: Gaussian Kernel array
        :param numpy.ndarray fd: Forcing Term array
        :return: Gaussian Kernel weights
        :rtype: numpy.ndarray
        """
        fd = dql_to_npa(fd)
        # Compute weights
        w_i = np.empty([fd.shape[1], self.n])
        for i in range(self.n):
            psi_m = np.diag(psi_i[:, i])
            w_i[:, i] = np.dot(np.dot(x.T, psi_m), fd) / np.dot(np.dot(x.T, psi_m), x)
        return w_i

    def reset_t(self, t=0.0):
        """
        Reset Starting Time

        :param float t: Starting time value (optional)
        """
        self.prev_t = t
        self.dq_trf = None

    def train_model(self, t, dq):
        """
        Get DMP Model from Dual Quaternion pose list

        :param numpy.ndarray t: Time vector
        :param numpy.ndarray dq: Dual Quaternion pose data
        :return: Gaussian Kernel weights array and Canonical System vector
        :rtype: (numpy.ndarray, numpy.ndarray)
        """
        # Coefficient for canonical system, Centers and Widths of Gaussian kernels
        self.__set_cfc(t[-1]-t[0])
        # Compute Canonical system and Gaussian kernels
        x, psi_i = self.__can_sys(t)
        # Time derivatives from q
        edq = self.__edq_from_dq_list(dq, dq[-1])
        tw = twist_from_dq_list(t, dq)
        dtw = dx_dt(t, tw)
        # Store training initial and goal conditions
        self.dq0d = dq[0]
        self.dqgd = dq[-1]
        self.twgd = tw[-1]
        # Forcing term from captured data
        fd = self.__fn_learn(dtw, tw, edq)
        # Weight learning
        w_i = self.__w_learn(x, psi_i, fd)
        self.w_i = w_i
        return w_i, x

    def fit_step(self, t, dq, tw, dqg, tau=1, dq0=None, twg=None):
        """
        Step-fit DMP Model to Dual Quaterion conditions

        :param float t: Current time
        :param DualQuaternion dq: Current Dual Quaternion pose
        :param DualQuaternion tw: Current Twist
        :param DualQuaternion dqg: Goal Dual Quaternion pose
        :param float tau: Time scaling parameter
        :param DualQuaternion dq0: Initial Dual Quaternion pose
        :param DualQuaternion or None twg: Goal Twist
        :return: Next Dual Quaternion pose and Next Twist
        :rtype: (DualQuaternion, DualQuaternion)
        """
        # Timestep
        dt = t - self.prev_t
        # Recalculate Canonical system
        x, psi_i = self.__can_sys(np.array([t]), tau)
        # Reconstruct forcing term
        fn = self.__fn_rct(x, psi_i, self.w_i)
        # if not twg is None:
        #     if not self.dq_trf:
        #         diff_d = dq_log(self.dq0d.inverse() * self.dqgd)
        #         diff_f = dq_log(dq0.inverse() * dqg)
        #         self.fn_scl = np.linalg.norm(diff_f.dq_array()) / np.linalg.norm(diff_d.dq_array())
        #         self.dq_trf = diff_f * diff_d.inverse()
        #         # print(self.fn_scl)
        #         # print(self.dq_trf)
        #     # fn = self.fn_scl * fn
        #     # fn = self.dq_trf * fn * self.dq_trf.inverse()
        #     # fn_ = dq_exp(fn)
        #     # fn_.q_r = self.dq_trf.q_r * dq_exp(fn).q_r
        #     # fn_.q_d = self.dq_trf.q_d * dq_exp(fn).q_d * dq_exp(fn).q_r.inverse * fn_.q_r
        #     # fn = dq_log(fn_)
        # Reconstruct pose
        edq = edq_from_dq(dq, dqg)
        dtw = self.__fit_dtw(tw, edq, fn)
        tw_n = tw + ((dt / tau) * dtw)
        dq_n = next_dq_from_twist(dt / tau, dq, tw)
        # Store current time
        self.prev_t = t
        return dq_n, tw_n

    def fit_model(self, t, dq0, tw0, dqg, tau=1, twg=None):
        """
        Fit DMP Model to Dual Quaternion conditions

        :param numpy.ndarray t: Time vector
        :param DualQuaternion dq0: Initial Dual Quaternion pose
        :param DualQuaternion tw0: Initial Twist
        :param DualQuaternion dqg: Goal Dual Quaternion pose
        :param float tau: Time scaling parameter
        :param DualQuaternion or None twg: Goal Twist
        :return: Reconstructed Dual Quaternion pose and Twist lists
        :rtype: (numpy.ndarray, numpy.ndarray)
        """
        # Initial conditions
        dq = dq0
        tw = tw0
    # Align Goal
        # dqg = dqg * align_dq_goal(dqg)
        # Reconstruct pose
        dq_arr = np.empty(t.shape[0], dtype=DualQuaternion)
        tw_arr = np.empty(t.shape[0], dtype=DualQuaternion)
        for i, ti in enumerate(t):
            dq_arr[i] = dq
            tw_arr[i] = (1 / tau) * tw
            dq, tw = self.fit_step(ti, dq, tw, dqg, tau=tau, dq0=dq0, twg=twg)
        return dq_arr, tw_arr


def main():
    pass


if __name__ == '__main__':
    main()
