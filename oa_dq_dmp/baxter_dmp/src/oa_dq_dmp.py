#!/usr/bin/env python

import numpy as np
import quaternion as quat
from dual_quaternions import DualQuaternion
from pyquaternion import Quaternion


class OADQMP(object):
    """
    A class for computing Obstacle Avoidance Dynamic Movement Primitives (DMP)
    from full pose Dual Quaternion data
    """

    def __init__(self, n=20, alpha_y=4, gamma_oa=16, beta_oa=32 / np.pi):
        """
        Obstacle Avoidance Dual Quaternion DMP Object

        :param int n: Number of Gaussian kernels in Psi
        :param float alpha_y: Dampening coefficient
        :param float gamma_oa: Steering modulation coefficient
        :param beta_oa: Obstacle influence coefficient
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
        # Obstacle Avoidance parameters
        self.gamma_oa = gamma_oa
        self.beta_oa = beta_oa
        # Second order systems
        self.fit_dtw = lambda tw, edq, fn: self.alpha_y * (np.multiply(self.beta_y, edq) - tw) + fn
        # Forcing function learning
        self.fn_learn = lambda dtw, tw, edq: dtw - self.alpha_y * (self.beta_y * edq - tw)
        # Forcing term reconstruction function
        self.fn_rct = lambda x, psi, w_i: (x * np.inner(psi, w_i)) / np.sum(psi, 1).reshape((-1, 1))

    @staticmethod
    def __log_dq(dq):
        """
        Dual Quaternion Logarithm (SE(3) -> se(3))

        :param DualQuaternion dq: Dual Quaternion
        :return: Dual Quaternion Logarithm
        :rtype: numpy.ndarray
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
        Dual Quaternion Exponential (se(3) -> SE(3))

        :param numpy.ndarray dq: Dual Quaternion
        :return: Dual Quaternion Exponential
        :rtype: DualQuaternion
        """
        p = quat.from_float_array(dq[0:4])
        d = quat.from_float_array(dq[4:8])
        yp = np.exp(p)
        yd = d * yp
        exp = DualQuaternion.from_dq_array(np.append(quat.as_float_array(yp), quat.as_float_array(yd)))
        return exp

    @staticmethod
    def __dx_dt(t, x):
        """
        Differentiate x w.r.t. t

        :param numpy.ndarray t: Time vector, (m)
        :param numpy.ndarray x: Data vector, (m, dim)
        :return: Differentiated data vector, (m, dim)
        :rtype: numpy.ndarray
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
        Dual Quaternion Error w.r.t. Goal

        :param List[DualQuaternion] dq: Dual Quaternion Pose list
        :param DualQuaternion dqg: Goal Pose
        :return edq: Dual Quaternion Pose Error List
        :rtype: numpy.ndarray
        """
        edq = []
        for dq_i in dq:
            edq.append(OADQMP.__log_dq(dq_i.quaternion_conjugate() * dqg))
        edq = np.array(edq).reshape((-1, 8))
        return edq

    @staticmethod
    def __pose_from_edq(edq, dqg):
        """
        Dual Quaternion Error w.r.t. Goal

        :param List[numpy.ndarray] edq: Dual Quaternion Pose Error list
        :param DualQuaternion dqg: Goal Pose
        :return: Dual Quaternion Pose List
        :rtype: List[DualQuaternion]
        """
        dq = []
        for edq_i in edq:
            dq.append(dqg * OADQMP.__exp_dq(edq_i).quaternion_conjugate())
        return dq

    @staticmethod
    def pose_from_dq(dq):
        """
        Pose Data from Dual Quaternion list

        :param List[DualQuaternion] dq: Dual Quaternion list
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

    @staticmethod
    def twist_from_dq(t, dq):
        """
        Twist list from Dual Quaternion list

        :param numpy.ndarray t: Time vector
        :param List[DualQuaternion] dq: Dual Quaternion list
        :return: Twist list
        :rtype: numpy.ndarray
        """
        tw = []
        for dqi, pdq, dti in zip(dq[1:], dq[0:-1], t[1:] - t[0:-1]):
            tw_ = (2 / dti) * OADQMP.__log_dq(pdq.quaternion_conjugate() * dqi)
            tw.append(tw_)
        tw.append(tw[-1])
        tw = np.array(tw).reshape((-1, 8))
        return tw

    @staticmethod
    def __compute_phi(vec, vel, gamma, beta):
        """
        Compute Phi Steering Factor from R3 vectors.

        :param numpy.ndarray vec: Position/Rotation Vector
        :param numpy.ndarray vel: Linear/Angular Velocity Vector
        :param float gamma: Gamma Modulation coefficient
        :param float beta: Beta Influence coefficient
        :return: Phi steering factor
        :rtype: numpy.ndarray
        """
        num = np.dot(vec, vel)
        den = np.linalg.norm(vec) * np.linalg.norm(vel)
        phi_ = np.arccos(num / den) if den != 0 else np.arccos(num)
        rax = np.cross(vec, vel)
        rax = rax if np.linalg.norm(rax) > 0 else np.array([0,0,1])
        # rm1 = np.eye(3) + rax
        # rm1 = np.eye(3) + np.sin(a) * rax + np.cos(a) * np.power(rax, 2)
        # phi = gamma * (np.matmul(rm1, vel)) * np.exp(-beta * phi_)
        qr = Quaternion(axis=rax, angle=phi_)
        phi = gamma * qr.rotate(vel) * np.exp(-beta * phi_)
        return phi

    def __set_cfc(self, tg):
        """
        Computes coefficients for canonical system,
        centers and widths for Gaussian Kernels.

        :param float tg: Time period to reach the goal
        """
        # Coefficient for canonical system adjusted to time period
        self.alpha_x = (self.alpha_y / 3) * (5 / tg)
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
        # Compute weights
        w_i = np.empty([fd.shape[1], self.n])
        for i in range(self.n):
            psi_m = np.diag(psi_i[:, i])
            w_i[:, i] = np.dot(np.dot(x.T, psi_m), fd) / np.dot(np.dot(x.T, psi_m), x)
        return w_i

    def avoid_obstacle(self, dq, dqo, tw):
        """
        Avoid Dual Quaternion Pose Obstacle.

        :param DualQuaternion dq: Current Pose
        :param DualQuaternion dqo: Obstacle Pose
        :param numpy.ndarray tw: Current Twist
        :return: Phi Steering Term
        :rtype: numpy.ndarray
        """
        # Setup Obstacle Avoidance Variables
        e_q_o = dq.quaternion_conjugate() * dqo
        d_pose = 2 * self.__log_dq(e_q_o)
        d_pos_ = np.array(d_pose[5:])
        d_rot_ = np.array(d_pose[1:4])
        vel_p = tw[5:]
        vel_r = tw[1:4]
        # Compute Position Part of Steering Term
        phi_p = self.__compute_phi(d_pos_, vel_p, self.gamma_oa, self.beta_oa)
        # Compute Rotation Part of Steering Term
        phi_r = self.__compute_phi(d_rot_, vel_r, self.gamma_oa, self.beta_oa)
        # Join Steering Term as se(3) vector
        phi = np.append(np.append(0, phi_r), np.append(0, phi_p))
        return phi

    def reset_t(self, t=0.0):
        """
        Reset Starting Time

        :param float t: Starting time value (optional)
        """
        self.prev_t = t

    def train_model(self, t, dq):
        """
        Get DMP Model from Dual Quaternion pose list

        :param numpy.ndarray t: Time vector
        :param List[DualQuaternion] dq: Dual Quaternion pose data
        :return: Gaussian Kernel weights array and Canonical System vector
        :rtype: (numpy.ndarray, numpy.ndarray)
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
        # Forcing term from captured data
        fd = self.fn_learn(dtw, tw, edq)
        # Weight learning
        w_i = self.__w_learn(x, psi_i, fd)
        self.w = w_i
        return w_i, x

    def fit_step(self, t, dq, tw, dq0, dqg, tau=1, dqo=None):
        """
        Step-fit DMP Model to Dual Quaterion conditions

        :param float t: Current time
        :param DualQuaternion dq: Current Dual Quaternion pose
        :param numpy.ndarray tw: Current Twist
        :param DualQuaternion dq0: Initial Dual Quaternion pose
        :param DualQuaternion dqg: Goal Dual Quaternion pose
        :param float tau: Time scaling parameter
        :param DualQuaternion dqo: Obstacle Dual Quaternion Pose
        :return: Next Dual Quaternion pose and Next Twist
        :rtype: (DualQuaternion, numpy.ndarray)
        """
        # Timestep
        dt = t - self.prev_t
        # Recalculate Canonical system
        x, psi_i = self.__can_sys(np.array([t]), tau)
        # Reconstruct forcing term
        fn = self.fn_rct(x, psi_i, self.w)[0]
        # Reconstruct pose
        edq = self.__log_dq(dq.quaternion_conjugate() * dqg)
        # Obstacle avoidance
        if dqo is not None:
            phi = self.avoid_obstacle(dq, dqo, tw)
            fn += phi
        dtw = self.fit_dtw(tw, edq, fn)
        tw_n = tw + (dt / tau) * dtw
        dq_n = dq * self.__exp_dq(0.5 * (dt / tau) * tw)
        # Store current time
        self.prev_t = t
        return dq_n, tw_n

    def fit_model(self, t, dq0, tw0, dqg, tau=1, dqo=None):
        """
        Fit DMP Model to Dual Quaternion conditions

        :param numpy.ndarray t: Time vector
        :param DualQuaternion dq0: Initial Dual Quaternion pose
        :param numpy.ndarray tw0: Initial Twist
        :param DualQuaternion dqg: Goal Dual Quaternion pose
        :param float tau: Time scaling parameter
        :param DualQuaternion dqo: Obstacle Dual Quaternion pose
        :return: Reconstructed Dual Quaternion pose and Twist lists
        :rtype: (List[DualQuaternion], List[numpy.ndarray])
        """
        # Initial conditions
        dq = dq0
        tw = tw0
        # Reconstruct pose
        dq_arr = []
        tw_arr = []
        for ti in t:
            dq_arr.append(dq)
            tw_arr.append(tw)
            dq, tw = self.fit_step(ti, dq, tw, dq0, dqg, tau=tau, dqo=dqo)
        return dq_arr, tw_arr


def main():
    pass


if __name__ == '__main__':
    main()
