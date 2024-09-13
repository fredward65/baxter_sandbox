#!/usr/bin/env python

import numpy as np
import quaternion as quat


class PoseDMP(object):
    """
    A class for computing Dynamic Movement Primitives (DMP)
    from pose data (Cartesian position and Quaternion orientation).
    """

    def __init__(self, n=20, alpha_y=4, mode='IJSPEERT_2002'):
        """
        Pose DMP Object Constructor

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
        self.w_q = np.empty((4, self.n))
        self.c_i = np.empty(0)
        self.h_i = np.empty(0)
        # Initial and goal dynamic conditions
        self.y0d = np.empty(3)
        self.ygd = np.empty(3)
        self.q0d = np.empty(0)
        self.qgd = np.empty(0)
        # Scaling factor parameters
        self.nv_p = 0
        self.nv_q = 0
        self.ytd = np.empty((2, 3))
        self.qtd = np.empty((2, 4))
        # Optional normal plane vectors
        self.pg_p = np.nan
        # Second order systems
        self.fit_ddy = lambda dy, y, yg, fn: self.alpha_y * (self.beta_y * (yg - y) - dy) + fn
        self.fit_ddeq = lambda deq, eq, fn: (-1 * self.alpha_y * (self.beta_y * eq + deq)) + fn
        # Forcing function learning
        self.fn_learn_p = lambda ddy, dy, y: ddy - self.alpha_y * (self.beta_y * (self.ygd - y) - dy)
        self.fn_learn_q = lambda ddeq, deq, eq: ddeq + self.alpha_y * (self.beta_y * eq + deq)
        # Forcing term reconstruction function
        self.fn_rct = lambda x, psi, w_i: (x * np.inner(psi, w_i)) / np.sum(psi, 1).reshape((-1, 1))
        # Scaling and Rotation mode
        self.mode = mode

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
        Unit Cartesian vector

        Parameters
        ----------
        v : numpy.ndarray
            Cartesian vector

        Returns
        -------
        v_hat : numpy.ndarray
            Unit Cartesian vector
        """
        v_hat = v / np.linalg.norm(v)
        return v_hat

    @staticmethod
    def __tan_vec(x):
        """
        Get non-zero tangent vector from list

        Parameters
        ----------
        x : numpy.ndarray
            Vector list, (m, dim)

        Returns
        -------
        xt : numpy.ndarray
            Vector tangent to the first displacement, (dim)
        """
        xt = PoseDMP.__v_hat(x[0, :])
        for x_c in x:
            xt = x_c - x[0, :]
            if np.linalg.norm(xt) > 0:
                xt = PoseDMP.__v_hat(xt)
                break
        return xt

    @staticmethod
    def __rod_rm(a, b, fac=1.0):
        """
        Compute rotation matrix from Rodrigues' formula

        Parameters
        ----------
        a : numpy.ndarray
            First Cartesian vector
        b : numpy.ndarray
            Second Cartesian vector
        fac : float
            Angle factor

        Returns
        -------
        rot : numpy.ndarray
            Rotation matrix (3, 3)
        """
        a = PoseDMP.__v_hat(a)
        b = PoseDMP.__v_hat(b)
        k = np.cross(a, b)
        th = np.arccos(np.dot(a, b)) * fac
        cth = np.cos(th)
        sth = np.sin(th)
        rot = np.array([cth + (k[0]**2)*(1-cth),      k[0]*k[1]*(1-cth) - k[2]*sth, k[1]*sth + k[0]*k[2]*(1-cth),
                        k[2]*sth + k[0]*k[1]*(1-cth), cth + (k[1]**2)*(1-cth),     -k[0]*sth + k[1]*k[2]*(1-cth),
                       -k[1]*sth + k[0]*k[2]*(1-cth), k[0]*sth + k[1]*k[2]*(1-cth), cth + (k[2]**2)*(1-cth)]
                       ).reshape((3, 3))
        # k = np.array([0, -k[2], k[1], k[2], 0, -k[0], -k[1], k[0], 0]).reshape((3, 3))
        # rot = np.eye(3) + k * np.sin(th) + k * k * (1 - np.cos(th))
        return rot

    @staticmethod
    def __plane_rm(nd, ng, pd, pg):
        """
        Compute planar rotation matrix

        Parameters
        ----------
        nd : numpy.ndarray
            Vector from difference in demonstrated Initial and Goal positions
        ng : numpy.ndarray
            Vector from difference in desired Initial and Goal positions
        pd : numpy.ndarray
            Vector normal to the demonstrated task plane
        pg : numpy.ndarray
            Vector normal to the desired task plane

        Returns
        -------
        rm : numpy.ndarray
            Planar rotation matrix (3, 3)
        """
        nd = PoseDMP.__v_hat(nd)
        ng = PoseDMP.__v_hat(ng)
        rm1 = np.array([nd, pd, np.cross(nd, pd)]).reshape((3, 3)).T
        rm2 = np.array([ng, pg, np.cross(ng, pg)]).reshape((3, 3)).T
        rm = np.matmul(rm1, rm2.T)
        return rm

    @staticmethod
    def __get_sg(nd, ng, nv=0.0, mode=''):
        """
        Computes scaling factor

        Parameters
        ----------
        nd : numpy.ndarray
            Demonstrated difference vector
        ng : numpy.ndarray
            Desired difference vector
        nv : float
            Unbiased variance of the demonstrated data
        mode : str
            Mode selector
            IJSPEERT_2002 = Ijspeert et al., 2002
            IJSPEERT_2013 = Ijspeert et al., 2013
            KOUTRAS_* = Koutras et al., 2020
            LIENDO_* = Liendo et al., 2022

        Returns
        -------
        sg : numpy.ndarray
            Scaling factor
        """
        sg = 1
        if 'IJSPEERT' in mode:
            if '2002' in mode:
                """Vanilla DMP (Ijspeert et al., 2002)"""
                sg = sg
            elif '2013' in mode:
                """Classic DMP (Ijspeert et al., 2013)"""
                sg = np.diag(ng / nd)
        elif 'KOUTRAS' in mode:
            """Global scaling (Koutras et al., 2020)"""
            sg = np.linalg.norm(ng) / np.linalg.norm(nd)
        elif 'LIENDO' in mode:
            """Our proposition (Liendo et al., 2022)"""
            sg = np.abs((nv + np.linalg.norm(ng)) / (nv + np.linalg.norm(nd)))
        return sg

    @staticmethod
    def __get_rm(nd, ng, pd=np.array([0, 0, 1]), pg=np.array([0, 0, 1]), vec=np.empty(0), mode=''):
        """
        Computes rotation matrix

        Parameters
        ----------
        nd : numpy.ndarray
            Demonstrated difference vector
        ng : numpy.ndarray
            Desired difference vector
        pd : numpy.ndarray
            Vector normal to the demonstrated task plane
        pg : numpy.ndarray
            Vector normal to the desired task plane
        vec : numpy.ndarray
            Initial and final demonstrated tangent vectors
        mode : str
            Mode selector
            IJSPEERT_* = Ijspeert et al., 2002, 2013
            KOUTRAS_FREE = Free task, Koutras et al., 2020
            KOUTRAS_PLANE = Planar task, Koutras et al., 2020
            LIENDO_* = Planar task, Liendo et al., 2022

        Returns
        -------
        rm : numpy.ndarray
            Rotation matrix
        """
        rm = np.eye(3)
        if 'IJSPEERT' in mode:
            """Classic DMP (Ijspeert et al., 2013)"""
            rm = rm
        elif 'KOUTRAS' in mode:
            if 'FREE' in mode:
                """Free task (Koutras et al., 2020)"""
                rm = PoseDMP.__rod_rm(nd, ng)
            elif 'PLANE' in mode:
                """Planar task (Koutras et al., 2020)"""
                rm = PoseDMP.__plane_rm(nd, ng, pd, pg)
        elif 'LIENDO' in mode:
            """Planar Tasks - Our proposition"""
            k = 5000
            # v_max = PoseDMP.__v_hat(np.sum(vec, axis=0))
            v_max = np.dot(vec[0, :], PoseDMP.__rod_rm(vec[1, :], vec[0, :], .5))
            nd = nd + (np.exp(-k * np.linalg.norm(nd)) * np.cross(v_max, pd))
            ng = ng + (np.exp(-k * np.linalg.norm(ng)) * np.cross(v_max, pg))
            rm = PoseDMP.__plane_rm(nd, ng, pd, pg)
        return rm

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

    def __get_sf_p(self, y0, yg):
        """
        Computes position scaling-rotation matrix

        Parameters
        ----------
        y0 : numpy.ndarray
            Initial Cartesian position
        yg : numpy.ndarray
            Goal Cartesian position

        Returns
        -------
        sg : numpy.ndarray
            Scaling-Rotation matrix, (3, 3)
        """
        # Vectors normal to task planes
        pd_p = np.abs(np.cross(self.y0d, self.ygd))
        pd_p = self.__v_hat(pd_p) if np.linalg.norm(pd_p) > 0 else self.__v_hat(np.cross(self.ytd[0], self.ytd[1]))
        pg_p = self.pg_p if hasattr(self.pg_p, "shape") else np.abs(np.cross(y0, yg))
        pg_p = self.__v_hat(pg_p) if np.linalg.norm(pg_p) > 0 else np.dot(pd_p, self.__rod_rm(self.y0d, y0))
        # Difference vectors
        nd = (self.ygd - self.y0d)
        ng = (yg - y0)
        # Scaling factor
        sg = self.__get_sg(nd, ng, self.nv_p, mode=self.mode)
        # Rotation matrix
        rm = self.__get_rm(nd, ng, pd=pd_p, pg=pg_p, vec=self.ytd, mode=self.mode)
        # Resulting scaling matrix
        sg = rm * sg
        return sg

    def __get_sf_q(self, q0, qg):
        """
        Computes orientation scaling-rotation matrix

        Parameters
        ----------
        q0 : quaternion.quaternion
            Initial Cartesian position
        qg : quaternion.quaternion
            Goal Cartesian position

        Returns
        -------
        sg : numpy.ndarray
            Scaling-Rotation matrix, (4, 4)
        """
        # Difference vectors
        nd = quat.as_float_array(2 * np.log(self.qgd * self.q0d.conj()))[1:]
        ng = quat.as_float_array(2 * np.log(qg * q0.conj()))[1:]
        # Scaling factor
        sg = self.__get_sg(nd, ng, self.nv_q, mode=self.mode)
        # Rotation matrix
        rm = self.__get_rm(nd, ng, vec=self.qtd, mode=self.mode)
        # Resulting scaling matrix
        sg = rm * sg
        sg = np.c_[[0, 0, 0], sg]
        sg = np.r_[np.array([1, 0, 0, 0]).reshape((1, -1)), sg]
        return sg

    def reset_t(self, t=0.0):
        """
        Reset current time variable

        Parameters
        ----------
        t : float
            Current time, default to 0.0
        """
        self.prev_t = t

    def set_mode(self, mode):
        """
        Set DMP mode

        Parameters
        ----------
        mode : str
            Mode selector
            IJSPEERT_2002 = Ijspeert et al., 2002
            IJSPEERT_2013 = Ijspeert et al., 2013
            KOUTRAS_FREE = Free task, Koutras et al., 2020
            KOUTRAS_PLANE = Planar task, Koutras et al., 2020
            LIENDO_* = Planar task, Liendo et al., 2022
        """
        self.mode = mode

    def set_pg_p(self, pg_p):
        """
        Set normal vector for desired task plane

        Parameters
        ----------
        pg_p : numpy.nparray
            Normal vector to desired task plane
        """
        self.pg_p = pg_p

    def train_model_p(self, t, y):
        """
        Get DMP Model from Cartesian trajectory

        Parameters
        ----------
        t : numpy.ndarray
            Time vector
        y : numpy.ndarray
            Cartesian trajectory data, (m, 3)

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
        # Store demonstrated conditions for scaling
        n_p = np.linalg.norm(y - self.y0d, axis=1)
        self.nv_p = np.sum((n_p - np.mean(n_p)) ** 2) / (len(n_p) - 1)
        self.ytd[0, :] = self.__tan_vec(y)
        self.ytd[1, :] = self.__tan_vec(np.flip(y, axis=0))
        # Forcing term from captured data
        fd = self.fn_learn_p(ddy, dy, y)
        # Weight learning
        self.w_p = self.__w_learn(x, psi_i, fd)
        return self.w_p, x

    def fit_model_p(self, t, y0, dy0, yg, tau=1):
        """
        Fit DMP Model to Cartesian conditions

        Parameters
        ----------
        t : numpy.ndarray
            Time vector
        y0 : numpy.ndarray
            Initial Cartesian position
        dy0 : numpy.ndarray
            Initial Cartesian velocity
        yg : numpy.ndarray
            Goal Cartesian position
        tau : float
            Time scaling variable Tau

        Returns
        -------
        y_arr : numpy.ndarray
            Reconstructed trajectory
        dy_arr : numpy.ndarray
            Reconstructed velocity
        dy_arr : numpy.ndarray
            Reconstructed acceleration
        """
        # Timestep vector
        dt = np.diff(t)
        dt = np.append(dt, dt[-1])
        # Recompute Canonical system
        x, psi_i = self.__can_sys(t, tau)
        # Reconstruct forcing term
        fn = self.fn_rct(x, psi_i, self.w_p)
        # Apply scaling factor
        sg = self.__get_sf_p(y0, yg)
        fn = np.dot(fn, sg)
        # Initial conditions
        y = y0
        dy = dy0
        # Reconstruct trajectory
        y_arr = np.empty([len(t), 3])
        dy_arr = np.empty([len(t), 3])
        ddy_arr = np.empty([len(t), 3])
        for i, dt_c in enumerate(dt):
            ddy = self.fit_ddy(dy, y, yg, fn[i, :])
            y_arr[i, :] = np.array([y]).reshape((1, -1))
            dy_arr[i, :] = np.array([dy]).reshape((1, -1))
            ddy_arr[i, :] = np.array([ddy]).reshape((1, -1))
            dy = dy + ddy * dt_c / tau
            y = y + dy * dt_c / tau
        return y_arr, dy_arr, ddy_arr

    def fit_step_p(self, t, y, dy, y0, yg, tau=1):
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
        y0 : numpy.ndarray
            Initial Cartesian position
        yg : numpy.ndarray
            Goal Cartesian position
        tau : float
            Time scaling variable Tau

        Returns
        -------
        y_n : numpy.ndarray
            Next Cartesian position
        dy_n : numpy.ndarray
            Next Cartesian velocity
        ddy_n : numpy.ndarray
            Next Cartesian acceleration
        """
        # Timestep
        dt = t - self.prev_t
        # Recalculate Canonical system
        x, psi_i = self.__can_sys(np.array([t]), tau)
        # Reconstruct forcing term
        fn = self.fn_rct(x, psi_i, self.w_p)
        # Apply scaling factor
        sg = self.__get_sf_p(y0, yg)
        fn = np.dot(fn, sg)
        # Reconstruct trajectory
        ddy_n = self.fit_ddy(dy, y, yg, fn)
        dy_n = dy + ddy_n * dt / tau
        y_n = y + dy_n * dt / tau
        # Store current time
        self.prev_t = t
        return y_n, dy_n, ddy_n

    def train_model_q(self, t, q):
        """
        Get DMP Model from Quaternion orientation list

        Parameters
        ----------
        t : numpy.ndarray
            Time vector
        q : numpy.ndarray
            Quaternion orientation data, (m, quaternion)

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
        # Time derivatives from q
        eq = quat.as_float_array(2 * np.log(np.multiply(q[-1], q.conj())))
        deq = self.__dx_dt(t, eq)
        ddeq = self.__dx_dt(t, deq)
        # Store training initial and goal conditions
        self.q0d = q[0]
        self.qgd = q[-1]
        # Store demonstrated conditions for scaling
        n_q = np.linalg.norm(quat.as_float_array(q), axis=1)
        self.nv_q = np.sum((n_q - np.mean(n_q)) ** 2) / (len(n_q) - 1)
        # Forcing term from captured data
        fd = self.fn_learn_q(ddeq, deq, eq)
        # Weight learning
        w_i = self.__w_learn(x, psi_i, fd)
        self.w_q = w_i
        return w_i, x

    def fit_model_q(self, t, q0, deq0, qg, tau=1):
        """
        Fit DMP Model to Quaternion conditions

        Parameters
        ----------
        t : numpy.ndarray
            Time vector
        q0 : quaternion.quaternion
            Initial Quaternion orientation
        deq0 : quaternion.quaternion
            Initial Quaternion error derivative
        qg : quaternion.quaternion
            Goal Quaternion orientation
        tau : float
            Time scaling variable Tau

        Returns
        -------
        q_arr : numpy.ndarray
            Reconstructed Quaternion orientation
        eq_arr : numpy.ndarray
            Reconstructed Quaternion orientation error
        deq_arr : numpy.ndarray
            Reconstructed Quaternion orientation error velocity
        ddeq_arr : numpy.ndarray
            Reconstructed Quaternion orientation error acceleration
        """
        # Timestep vector
        dt = np.diff(t)
        dt = np.append(dt, dt[-1])
        # Recompute Canonical system
        x, psi_i = self.__can_sys(t, tau)
        # Reconstruct forcing term
        fn = quat.as_quat_array(self.fn_rct(x, psi_i, self.w_q))
        # Apply scaling factor
        sg_q = self.__get_sf_q(q0, qg)
        fn = np.dot(fn, sg_q)
        # Initial conditions
        q = q0
        eq = 2 * np.log(np.multiply(qg, q.conj()))
        deq = deq0
        # Reconstruct orientations
        q_arr = np.empty(len(t), dtype="quaternion")
        eq_arr = np.empty(len(t), dtype="quaternion")
        deq_arr = np.empty(len(t), dtype="quaternion")
        ddeq_arr = np.empty(len(t), dtype="quaternion")
        for i, dt_c in enumerate(dt):
            ddeq = self.fit_ddeq(deq, eq, fn[i])
            q_arr[i] = q
            eq_arr[i] = eq
            deq_arr[i] = deq
            ddeq_arr[i] = ddeq
            deq = deq + ddeq * dt_c / tau
            eq = eq + deq * dt_c / tau
            q = np.multiply(np.exp(.5 * eq).conj(), qg)
        return q_arr, eq_arr, deq_arr, ddeq_arr

    def fit_step_q(self, t, q, deq, q0, qg, tau=1):
        """
        Step-fit DMP Model to Cartesian conditions

        Parameters
        ----------
        t : float
            Current time value
        q : quaternion.quaternion
            Current Quaternion orientation
        deq : quaternion.quaternion
            Current Quaternion error velocity
        q0 : quaternion.quaternion
            Initial Quaternion orientation
        qg : quaternion.quaternion
            Goal Quaternion orientation
        tau : float
            Time scaling variable Tau

        Returns
        -------
        q_n : quaternion.quaternion
            Next Quaterion orientation
        deq_n : quaternion.quaternion
            Next Quaternion error velocity
        """
        # Timestep
        dt = t - self.prev_t
        # Recalculate Canonical system
        x, psi_i = self.__can_sys(np.array([t]), tau)
        # Reconstruct forcing term
        fn = quat.as_quat_array(self.fn_rct(x, psi_i, self.w_q))[0]
        # Apply scaling factor
        sg_q = self.__get_sf_q(q0, qg)
        fn = np.dot(fn, sg_q)
        # Reconstruct orientations
        eq = 2 * np.log(np.multiply(qg, q.conj()))
        ddeq = self.fit_ddeq(deq, eq, fn)
        deq_n = deq + ddeq * dt / tau
        eq = eq + deq * dt / tau
        q_n = np.multiply(np.exp(.5 * eq).conj(), qg)
        # Store current time
        self.prev_t = t
        return q_n, deq_n

    def train_model(self, t, y, q):
        """
        Get DMP Model from Cartesian trajectory

        Parameters
        ----------
        t : numpy.ndarray
            Time vector
        y : numpy.ndarray
            Cartesian trajectory data, (m, 3)
        q : numpy.ndarray
            Quaternion orientation data, (m, quaternion.quaternion)

        Returns
        -------
        wi_p : numpy.ndarray
            Position Gaussian kernels weights w_p, (3, n)
        wi_q : numpy.ndarray
            Orientation Gaussian kernels weights w_p, (4, n)
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
        # Store demonstrated conditions for scaling
        n_p = np.linalg.norm(y - self.y0d, axis=1)
        self.nv_p = np.sum((n_p - np.mean(n_p)) ** 2) / (len(n_p) - 1)
        self.ytd[0, :] = self.__tan_vec(y)
        self.ytd[1, :] = self.__tan_vec(np.flip(y, axis=0))
        # Forcing term from captured data
        fd_p = self.fn_learn_p(ddy, dy, y)

        # Time derivatives from q
        eq = quat.as_float_array(2 * np.log(np.multiply(q[-1], q.conj())))
        deq = self.__dx_dt(t, eq)
        ddeq = self.__dx_dt(t, deq)
        # Store training initial and goal conditions
        self.q0d = q[0]
        self.qgd = q[-1]
        # Store demonstrated conditions for scaling
        n_q = np.linalg.norm(quat.as_float_array(q), axis=1)
        self.nv_q = np.sum((n_q - np.mean(n_q)) ** 2) / (len(n_q) - 1)
        self.qtd[0, :] = quat.as_float_array(np.log(q[1] * q[0].conj()))
        self.qtd[1, :] = quat.as_float_array(np.log(q[-2] * q[-1].conj()))
        # Forcing term from captured data
        fd_q = self.fn_learn_q(ddeq, deq, eq)

        # Weight learning for position model
        self.w_p = self.__w_learn(x, psi_i, fd_p)
        # Weight learning for orientation model
        self.w_q = self.__w_learn(x, psi_i, fd_q)
        return self.w_p, self.w_q, x

    def fit_model(self, t, y0, dy0, yg, q0, deq0, qg, tau=1):
        """
        Fit DMP Model to pose Cartesian and Quaternion conditions

        Parameters
        ----------
        t : numpy.ndarray
            Time vector, (m)
        y0 : numpy.ndarray
            Initial Cartesian position
        dy0 : numpy.ndarray
            Initial Cartesian velocity
        yg : numpy.ndarray
            Goal Cartesian position
        q0 : quaternion.quaternion
            Initial Quaternion orientation
        deq0 : quaternion.quaternion
            Initial Quaternion error derivative
        qg : quaternion.quaternion
            Goal Quaternion orientation
        tau : float
            Time scaling variable Tau

        Returns
        -------
        y_arr : numpy.ndarray
            Reconstructed Cartesian trajectory, (m, 3)
        q_arr : numpy.ndarray
            Reconstructed Quaternion orientation, (m, quaternion.quaternion)
        """
        # Timestep vector
        dt = np.diff(t)
        dt = np.append(dt, dt[-1])
        # Recompute Canonical system
        x, psi_i = self.__can_sys(t, tau)
        # Reconstruct forcing term
        fn_p = self.fn_rct(x, psi_i, self.w_p)
        fn_q = self.fn_rct(x, psi_i, self.w_q)
        # Apply scaling factor
        sg_p = self.__get_sf_p(y0, yg)
        fn_p = np.dot(fn_p, sg_p)
        sg_q = self.__get_sf_q(q0, qg)
        fn_q = quat.as_quat_array(np.dot(fn_q, sg_q))
        # Initial conditions
        y = y0
        dy = dy0
        q = q0
        eq = 2 * np.log(np.multiply(qg, q.conj()))
        deq = deq0
        # Reconstruct pose
        y_arr = np.empty([len(t), 3])
        q_arr = np.empty(len(t), dtype="quaternion")
        for i, dt_c in enumerate(dt):
            # Compute current accelerations
            ddy = self.fit_ddy(dy, y, yg, fn_p[i, :])
            ddeq = self.fit_ddeq(deq, eq, fn_q[i])
            # Store current pose data
            y_arr[i, :] = np.array([y]).reshape((1, -1))
            q_arr[i] = q
            # Compute next pose data
            dy = dy + ddy * dt_c / tau
            y = y + dy * dt_c / tau
            deq = deq + ddeq * dt_c / tau
            eq = eq + deq * dt_c / tau
            q = np.multiply(np.exp(.5 * eq).conj(), qg)
        return y_arr, q_arr

    def fit_step(self, t, y, dy, y0, yg, q, deq, q0, qg, tau=1):
        """
        Step-fit DMP Model to pose conditions

        Parameters
        ----------
        t : float
            Current time value
        y : numpy.ndarray
            Current Cartesian position
        dy : numpy.ndarray
            Current Cartesian velocity
        y0 : numpy.ndarray
            Initial Cartesian position
        yg : numpy.ndarray
            Goal Cartesian position
        q : quaternion.quaternion
            Current Quaternion orientation
        deq : quaternion.quaternion
            Current Quaternion error velocity
        q0 : quaternion.quaternion
            Initial Quaternion orientation
        qg : quaternion.quaternion
            Goal Quaternion orientation
        tau : float
            Time scaling variable Tau

        Returns
        -------
        y_n : numpy.ndarray
            Next Cartesian position
        dy_n : numpy.ndarray
            Next Cartesian velocity
        q_n : quaternion.quaternion
            Next Quaterion orientation
        deq_n : quaternion.quaternion
            Next Quaternion error velocity
        """
        # Timestep
        dt = t - self.prev_t
        # Recalculate Canonical system
        x, psi_i = self.__can_sys(np.array([t]), tau)
        # Reconstruct forcing terms
        fn_p = self.fn_rct(x, psi_i, self.w_p)
        fn_q = self.fn_rct(x, psi_i, self.w_q)
        # Apply scaling factor
        sg_p = self.__get_sf_p(y0, yg)
        fn_p = np.dot(fn_p, sg_p)
        sg_q = self.__get_sf_q(q0, qg)
        fn_q = quat.as_quat_array(np.dot(fn_q, sg_q))[0]
        # Reconstruct trajectory
        ddy_n = self.fit_ddy(dy, y, yg, fn_p)
        dy_n = dy + ddy_n * dt / tau
        y_n = y + dy_n * dt / tau
        # Reconstruct orientations
        eq = 2 * np.log(np.multiply(qg, q.conj()))
        ddeq = self.fit_ddeq(deq, eq, fn_q)
        deq_n = deq + ddeq * dt / tau
        eq = eq + deq * dt / tau
        q_n = np.multiply(np.exp(.5 * eq).conj(), qg)
        # Store current time
        self.prev_t = t
        return y_n, dy_n, q_n, deq_n


def main():
    pass


if __name__ == "__main__":
    main()
