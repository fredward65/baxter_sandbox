#!/usr/bin/env python

import numpy as np
from pyquaternion import Quaternion
from dual_quaternions import DualQuaternion
from math_tools import dx_dt, q_rot_from_vec, quat_rot


class ProjectileModel(object):
    def __init__(self):
        self.g = -9.80665
        self.tf = 0
        self.dy0 = np.zeros(3)
        self.k = np.zeros(3)
        # Condition Estimator Model
        self.dz0_eq = lambda tf, z0, z: -(((self.g * tf**2) / 2) - z + z0)/tf
        self.dx0_eq = lambda tf, dz0, ik: (self.k[ik] / self.k[2]) * (dz0 + self.g * tf)
        # Trivial Kinetic Model
        self.x_eq = lambda t, x0, dx0: dx0 * t + x0
        self.z_eq = lambda t, z0, dz0: .5 * self.g * t**2 + dz0 * t + z0

    def tf_eq(self, delta_y):
        kz = self.k[2]
        ak = (kz**2 - 1)
        num = - 2 * (ak * delta_y[2] + kz * np.sqrt(-1 *  ak * (delta_y[0]**2 + delta_y[1]**2)))
        den = self.g * ak
        np.seterr(invalid='raise')
        try:
            res = np.sqrt(num / den)
        except FloatingPointError:
            res = np.nan
            print('Not possible')
        return res

    @staticmethod
    def align_k(qk, dy):
        vec_k = np.multiply(qk.rotate([0, 0, -1]), np.array([1, 1, 0]))
        vec_dy = np.multiply(dy, np.array([1, 1, 0]))
        q_rot = quat_rot(Quaternion(vector=vec_k), Quaternion(vector=vec_dy)) * qk
        k = q_rot.rotate([0, 0, -1])
        return k


    def solve(self, y0, yf, qk):
        # self.k = qk.rotate([0, 0, -1])
        self.k = self.align_k(qk, yf - y0)
        self.tf = self.tf_eq(yf - y0)
        self.dy0[2] = self.dz0_eq(self.tf, y0[2], yf[2])
        self.dy0[0] = self.dx0_eq(self.tf, self.dy0[2], 0)
        self.dy0[1] = self.dx0_eq(self.tf, self.dy0[2], 1)
        # Dual Quaternion and Twist Dual Quaternion Computation
        q_0, tw_0 = self.compute_dq(y0)
        # print(['{:0.2f}'.format(q_i) for q_i in q_0.dq_array()])
        # print(['{:0.2f}'.format(dq_i) for dq_i in tw_0.dq_array()])
        return self.dy0, self.tf

    @staticmethod
    def dq_from_vecs(dy0, dy1, y0, dt):
        q0 = q_rot_from_vec(dy0)
        q1 = q_rot_from_vec(dy1)
        wi = 2 * (q1 - q0) * (1 / dt) * q0.conjugate
        vi = (Quaternion(vector=y0) * wi)
        vi = (vi - vi.w) + Quaternion(vector=dy0)
        q_0 = DualQuaternion.from_quat_pose_array(np.append(q0.elements, y0))
        tw_0 = DualQuaternion.from_dq_array(np.append(wi.elements, vi.elements))
        return q_0, tw_0

    def compute_dq(self, y0):
        # q0 = self.rotation_from_vec(self.dy0)
        r, t = self.evaluate(y0, n=3, tf=self.tf/1000)
        dr = dx_dt(t, r)
        q_0, tw_0 = self.dq_from_vecs(dr[0, :], dr[1, :], r[0, :], np.diff(t)[0])
        return q_0, tw_0

    def evaluate(self, y0, n=100, tf=None):
        tf_ = tf if not tf is None else self.tf
        t = np.linspace(0, tf_, num=n)
        x = self.x_eq(t, y0[0], self.dy0[0])
        y = self.x_eq(t, y0[1], self.dy0[1])
        z = self.z_eq(t, y0[2], self.dy0[2])
        r = np.c_[x, y, z]
        return r, t


def main():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    fig, ax_ = plt.subplots(4, 3)
    [[axis.set_ylim(-15, 15) for axis in axes] for axes in ax_]

    ax.set_xlim3d(-2, 2)
    ax.set_ylim3d(-2, 2)
    ax.set_zlim3d(-2, 2)

    r = 1
    n1 = 2
    nl = 10
    n2 = 20

    pm = ProjectileModel()
    for kz, kl in zip(np.linspace(-1, 1, num=n1), range(n1)):
        y0 = np.array([0, 0, 0])
        for ki in np.linspace(-nl, nl, num=n2):
            alpha = np.pi * (0 + 2 * kl / n1)
            yf = np.array([r * np.sin(alpha), r * np.cos(alpha), 2*kz])
            k = np.array([yf[0]-y0[0], yf[1]-y0[1], ki])
            qk = q_rot_from_vec(-k)
            dy0, tf = pm.solve(y0, yf, qk)
            y, t = pm.evaluate(y0, n=200)

            def draw_rot(vec, v0, l=.25, a=1.):
                q_r = q_rot_from_vec(vec)
                w = q_r.rotate([0, 0, 1])
                ax.quiver(v0[0], v0[1], v0[2], w[0], w[1], w[2], length=l, pivot='tail', color=col, alpha=a)

            k_ = qk.rotate([0, 0, 1])
            print("kz: %5.3f tf:%5.3f dz0:%5.3f err:%5.3f" % (k_[2], tf, dy0[2], np.linalg.norm(y[-1, :] - yf)))
            if not np.isnan(tf):
                col = (nl - ki) / (nl * 2)
                col = (col, 1 - col, .5 * (1 + kz))
                ax.plot([yf[0]], [yf[1]], [yf[2]], 'xk')
                ax.plot(y[:, 0], y[:, 1], y[:, 2], color=col)
                idx = int(len(y) / 2)
                lbl = "tf:%5.3f, dz0:%5.3f" % (tf, dy0[2])
                ax.text(y[idx, 0], y[idx, 1], y[idx, 2], lbl, color=col, horizontalalignment='center')
                dyf = np.diff(y[-2:, :], axis=0)[0, :] / np.diff(t[-2:])

                # Twist Dual Quaternion Rendering
                dy = dx_dt(t, y)
                dt = np.diff(t)
                tw_ = np.empty((t.shape[0] - 2), dtype=DualQuaternion)
                for i, (yi, pdyi, dyi, dti) in enumerate(zip(y[0:-2], dy[0:-2], dy[1:], dt)):
                    q_i, tw_i = pm.dq_from_vecs(pdyi, dyi, yi, dti)
                    tw_[i] = tw_i
                for i, axes in enumerate(ax_):
                    axes[0].plot(t[:-2], [q_i.q_r.elements[i] for q_i in tw_], color=col)
                    axes[2].plot(t[:-2], [q_i.q_d.elements[i] for q_i in tw_], color=col)
                    axes[1].plot(t if i > 0 else [0], dy[:, i-1] if i > 0 else [0], color=col)

                draw_rot(dy0, y0)
                draw_rot(dyf, y[-1, :])
            draw_rot(k_, yf, l=.5, a=.5)
    plt.show()


if __name__ == '__main__':
    main()
