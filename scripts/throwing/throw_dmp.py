#!/usr/bin/env python

import numpy as np
import quaternion as quat


class DMP(object):
    """
    A class for computing Dynamic Movement Primitives (DMP) from captured data.
    """

    def __init__(self, n=20, alpha_y=4):
        """
        DMP Object Constructor

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
        self.w_p = np.empty((3, self.n))
        self.c_i = np.empty(0)
        self.h_i = np.empty(0)
        # Initial and goal dynamic conditions
        self.y0d = np.empty(3)
        self.ygd = np.empty(3)
        self.dygd = np.empty(3)
        # Scaling and Rotating
        self.qrot = None
        self.sg = None
        # Second order systems
        self.fit_ddy = lambda dy, y, yg, fn: self.alpha_y * (self.beta_y * (yg - y) - dy) + fn
        # Forcing function learning
        self.fn_learn_p = lambda ddy, dy, y: ddy - self.alpha_y * (self.beta_y * (self.ygd - y) - dy)
        # Forcing term reconstruction function
        self.fn_rct = lambda x, psi, w_i: (x * np.inner(psi, w_i)) / np.sum(psi, 1).reshape((-1, 1))

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
    def __v_hat(v):
        """
        Unit vector

        Parameters
        ----------
        v : numpy.ndarray
            Vector

        Returns
        -------
        v_hat : numpy.ndarray
            Unit vector
        """
        v_hat = v / np.linalg.norm(v)
        return v_hat

    @staticmethod
    def __quat_from_vecs(a, b):
        """
        Compute Quaternion that rotates from vector a to vector b

        Parameters
        ----------
        a : numpy.ndarray
            First vector 3
        b : numpy.ndarray
            Second vector 3

        Returns
        -------
        q : quaternion.quaternion
            Quaternion rotates a to b
        """
        na_ = np.linalg.norm(a)
        nb_ = np.linalg.norm(b)
        if na_ != 0 and nb_ != 0:
            x = np.cross(a, b)
            w = np.dot(a, b) + np.sqrt(na_ ** 2 * nb_ ** 2)
            q = quat.from_float_array([w, x[0], x[1], x[2]]).normalized()
        else:
            q = quat.from_euler_angles(0, 0, 0)
        return q

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

    def reset_t(self, t=0.0):
        """
        Reset Starting Time

        Parameters
        ----------
        t : float
            Starting time value (optional)
        """
        self.prev_t = t

    def train_model(self, t, y):
        """
        Train DMP Model from captured data

        Parameters
        ----------
        t : numpy.ndarray
            Time vector
        y : numpy.ndarray
            Captured data, (m, n_)

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
        # Time derivatives from y
        dy = self.__dx_dt(t, y)
        ddy = self.__dx_dt(t, dy)
        # Store training initial and goal conditions
        self.y0d = y[0, :]
        self.ygd = y[-1, :]
        # Forcing term from captured data
        fd = self.fn_learn_p(ddy, dy, y)
        # Weight learning
        self.w_p = self.__w_learn(x, psi_i, fd)
        # Store demonstrated conditions for scaling
        y_, _, _ = self.fit_model(t, self.y0d, dy[-1, :], self.ygd)
        self.dygd = np.divide(np.diff(y_, axis=0), np.diff(t).reshape((-1, 1)))[-1, :]
        return self.w_p, x

    def fit_step(self, t, y, dy, yg, y0, tau=1, dyg=None):
        """
        Step-fit DMP Model to Cartesian conditions

        Parameters
        ----------
        t : float
            Current time value
        y : numpy.ndarray
            Current Cartesian position
        dy : numpy.ndarray
            Current Cartesian velocity
        yg : numpy.ndarray
            Goal Cartesian position
        y0 : numpy.ndarray
            Goal Cartesian position
        tau : float
            Time scaling variable Tau
        dyg : numpy.ndarray
            Goal velocity

        Returns
        -------
        y_n : numpy.ndarray
            Next Cartesian position
        dy_n : numpy.ndarray
            Next Cartesian velocity
        """
        # Timestep
        dt = t - self.prev_t
        # Recalculate Canonical system
        x, psi_i = self.__can_sys(np.array([t]), tau)
        # Reconstruct forcing term
        fn = self.fn_rct(x, psi_i, self.w_p)
        if self.qrot is None and dyg is not None:
            a_ = self.__v_hat(self.dygd)
            b_ = self.__v_hat(dyg)
            c_ = self.__v_hat(self.ygd - self.y0d)
            d_ = self.__v_hat(yg - y0)
            e_ = np.cross(c_, a_)
            f_ = np.cross(d_, b_)
            qrot = self.__quat_from_vecs(e_, f_)
            g_ = quat.rotate_vectors(qrot, a_)
            qres = self.__quat_from_vecs(g_, b_)
            self.qrot = qres * qrot
            # self.sg = np.linalg.norm(yg - y0) / np.linalg.norm(self.ygd - self.y0d)
        if self.qrot is not None:
            fn = quat.rotate_vectors(self.qrot, fn)
            # fn *= self.sg
        # Reconstruct trajectory
        ddy_n = self.fit_ddy(dy, y, yg, fn)
        dy_n = dy + ddy_n * dt / tau
        y_n = y + dy_n * dt / tau
        # Store current time
        self.prev_t = t
        return y_n, dy_n, ddy_n

    def fit_model(self, t, y0, dy0, yg, tau=1, dyg=None):
        """
        Fit DMP Model to a set of conditions

        Parameters
        ----------
        t : numpy.ndarray
            Time vector
        y0 : numpy.ndarray
            Initial vector
        dy0 : numpy.ndarray
            Initial velocity
        yg : numpy.ndarray
            Goal vector
        tau : float
            Time scaling variable Tau
        dyg : numpy.ndarray
            Goal velocity

        Returns
        -------
        y_arr : numpy.ndarray
            Reconstructed data
        dy_arr : numpy.ndarray
            Reconstructed velocity
        dy_arr : numpy.ndarray
            Reconstructed acceleration
        """
        # Reset local time variable
        self.reset_t(t[0])
        # Initial conditions
        y = y0
        dy = dy0
        # Reconstruct trajectory
        y_arr = np.empty([len(t), 3])
        dy_arr = np.empty([len(t), 3])
        ddy_arr = np.empty([len(t), 3])
        for i, ti in enumerate(t):
            y, dy, ddy = self.fit_step(ti, y, dy, yg, y0, tau=tau, dyg=dyg)
            y_arr[i, :] = np.array([y]).reshape((1, -1))
            dy_arr[i, :] = np.array([dy]).reshape((1, -1))
            ddy_arr[i, :] = np.array([ddy]).reshape((1, -1))
        return y_arr, dy_arr, ddy_arr


def main():
    pass


if __name__ == "__main__":
    main()
