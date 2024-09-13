#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import quaternion as quat
from dual_quaternions import DualQuaternion


class ProjectileModel(object):
    def __init__(self):
        self.g = -9.80665
        self.dxy0 = 0
        self.dz0 = 0
        self.tf = 0
        self.ang = 0
        # Condition Estimator Model
        self.tf_eq = lambda x0, x, z0, z, th: -(2**(1/2)*(-self.g*(z-z0 + np.tan(th)*(x-x0)))**(1/2)) / self.g
        self.dz0_eq = lambda tf, z0, z: -((self.g*tf**2)/2 - z + z0)/tf
        self.dxy0_eq = lambda tf, dz0, th: (self.g*tf + dz0)*np.tan(.5*np.pi+th)
        # Trivial Kinetic Model
        self.xy_eq = lambda t, xy0, dxy0: dxy0*t + xy0
        self.z_eq = lambda t, z0, dz0: .5*self.g*t**2 + dz0*t + z0

    def solve(self, x0, x, z0, z, theta):
        self.tf = self.tf_eq(x0, x, z0, z, theta)
        self.dz0 = self.dz0_eq(self.tf, z0, z)
        self.dxy0 = self.dxy0_eq(self.tf, self.dz0, theta)
        return self.dxy0, self.dz0, self.tf

    def evaluate(self, x0, z0, n=100):
        t = np.linspace(0, self.tf, num=n)
        x = self.xy_eq(t, x0, self.dxy0)
        z = self.z_eq(t, z0, self.dz0)
        return x, z, t

    def solve3d(self, xyz0, xyz, theta):
        xy_ = xyz[0:2] - xyz0[0:2]
        self.ang = np.arctan2(xy_[1], xy_[0])
        hyp = np.linalg.norm(xy_)
        self.solve(0, hyp, xyz0[-1], xyz[-1], theta)
        dx = self.dxy0 * np.cos(self.ang)
        dy = self.dxy0 * np.sin(self.ang)
        return [dx, dy, self.dz0], self.tf

    def evaluate3d(self, xyz0, n=100):
        xy, z, t = self.evaluate(0, xyz0[-1], n=n)
        x = xy * np.cos(self.ang) + xyz0[0]
        y = xy * np.sin(self.ang) + xyz0[1]
        xyz = np.c_[x, y, z]
        return xyz, t


def main():
    x0, x = .0, 2.
    z0, z = .5, 0.
    theta = np.deg2rad(60)

    pm = ProjectileModel()
    dx0, dz0, tf = pm.solve(x0, x, z0, z, theta)
    print(tf, dx0, dz0)

    x_, z_, t_ = pm.evaluate(x0, z0)
    print(np.rad2deg(np.arctan2(x_[-1]-x_[-2], z_[-1]-z_[-2]) - .5*np.pi))

    fig, axs = plt.subplots(2)
    axs[0].plot(t_, x_, t_, z_)
    axs[1].plot(x_, z_)
    # axs[1].quiver(x0, z0, dx0, dz0)
    # axs[1].quiver(x_[-1], z_[-1], x_[-2]-x_[-1], z_[-2]-z_[-1])
    plt.show()


def main2():
    xyz0 = np.array([0., 0., .5])
    xyz_ = np.array([2., 1., .0])
    theta = np.deg2rad(60)

    pm = ProjectileModel()
    dxyz0, tf = pm.solve3d(xyz0, xyz_, theta)
    print(tf, dxyz0)

    xyz, t = pm.evaluate3d(xyz0)
    plt.figure()
    plt.plot(t, xyz)
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    plt.show()


if __name__ == '__main__':
    main2()
