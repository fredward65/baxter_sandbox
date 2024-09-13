#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import quaternion as quat
from mpl_toolkits import mplot3d
from pose_dmp import PoseDMP


def main():
    # Generate training data
    t = np.linspace(0, 1, num=1000)
    x = np.cos(np.pi*(2*t + 0)) + 1
    y = np.sin(np.pi*(2*t + 1)) + 1  # + .005*t
    z = np.zeros(t.shape)
    traj = np.array([x, y, z]).T
    traj_q = quat.from_euler_angles(traj)

    dt = np.diff(t, axis=0).reshape((-1, 1))

    # Matrix rotations for fiddling
    # th = np.pi*.25
    # rot = np.array([1, 0, 0, 0, np.cos(th), -np.sin(th), 0, np.sin(th), np.cos(th)]).reshape((3, 3))
    # rot = np.array([np.cos(th), 0, -np.sin(th), 0, 1, 0, np.sin(th), 0, np.cos(th)]).reshape((3, 3))
    # rot = np.array([np.cos(th), -np.sin(th), 0, np.sin(th), np.cos(th), 0, 0, 0, 1]).reshape((3, 3))

    th = -np.pi * .5
    rot1 = np.array([np.cos(th), -np.sin(th), 0, np.sin(th), np.cos(th), 0, 0, 0, 1]).reshape((3, 3))
    th = -np.pi * .25
    rot2 = np.array([np.cos(th), 0, -np.sin(th), 0, 1, 0, np.sin(th), 0, np.cos(th)]).reshape((3, 3))
    rot = np.dot(rot2, rot1)
    # rot = np.eye(3)

    pg_p = np.dot(rot, np.array([0, 0, 1]))

    offset = np.array([0, 0, 1])
    dy0 = np.dot(rot, np.divide(np.diff(traj, axis=0), dt)[0, :]) * 0
    y0 = traj[0, :] + offset
    yg = traj[0, :] + np.dot(rot, traj[-1, :]-traj[0, :]) + offset

    q0 = traj_q[0]
    qg = traj_q[-1]
    eq = 2 * np.log(np.multiply(qg, traj_q.conj()))
    deq0 = np.divide(np.diff(eq, axis=0), dt)[0, 0]

    # Pose DMP Model object
    dmp_obj = PoseDMP(n=100, alpha_y=24)
    # Model training
    dmp_obj.train_model(t, traj, traj_q)
    # New time vector
    t_new = np.linspace(0, 1, num=200)
    # Set desired task plane vector
    dmp_obj.set_pg_p(pg_p)
    # Model batch reconstruction
    # traj_rec, traj_q_rec = dmp_obj.fit_model(t_new, y0, dy0, yg, q0, deq0, qg)
    # Model step reconstruction
    y = y0
    dy = dy0
    q = q0
    deq = deq0
    traj_rec = np.empty([len(t_new), traj.shape[1]])
    traj_q_rec = np.empty(len(t_new), dtype="quaternion")
    for i, t_step in enumerate(t_new):
        traj_rec[i, :] = y.reshape(-1, traj.shape[1])
        traj_q_rec[i] = q
        y, dy, q, deq = dmp_obj.fit_step(t_step, y, dy, y0, yg, q, deq, q0, qg)

    """ Plots """
    # Position
    plt.plot(t, traj, '--k')
    plt.plot(t_new, traj_rec)
    plt.show()
    ax = plt.axes(projection='3d')
    ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2], '--k')
    ax.plot3D(traj_rec[:, 0], traj_rec[:, 1], traj_rec[:, 2])
    ax.plot3D([-0], [-0], [-0])
    ax.plot3D([3], [3], [3])
    plt.show()
    # Orientation
    plt.plot(t, quat.as_float_array(traj_q), '--k')
    plt.plot(t_new, quat.as_float_array(traj_q_rec))
    plt.show()
    plt.plot(t, quat.as_euler_angles(traj_q), '--k')
    plt.plot(t_new, quat.as_euler_angles(traj_q_rec))
    plt.show()


if __name__ == "__main__":
    main()
