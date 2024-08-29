#!/usr/bin/env python

import numpy as np
from copy import deepcopy
from custom_tools.math_tools import dq_log, dql_to_npa, dx_dt, quat_rot, twist_from_dq_list, vel_from_twist
from custom_tools.projectile_model import ProjectileModel
from dual_quaternions import DualQuaternion
from pt_dq_dmp import PTDQMP
from pyquaternion import Quaternion


def main():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Get data from file
    data_path = './src' + '/pt_dq_dmp' + '/baxter_dartboard' + '/resources' + '/demo_throw_left_5.csv'
    data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
    _, d_idx, counts = np.unique(data[:, 1:], axis=0, return_index=True, return_counts=True)
    data = data[np.sort(d_idx), :]

    # Parse data to time, rotation and translation
    t = np.linspace(0, data[-1, 0] - data[0, 0], num=data.shape[0])
    r = data[:, 4:]
    p = data[:, 1:4]

    dmp_obj = PTDQMP(n=100, alpha_y=20)
    dq_list = dmp_obj.dq_from_pose(r, p)
    tw_list = twist_from_dq_list(t, dq_list)

    # Train DMP Model
    dmp_obj.train_model(t, dq_list)

    tau = .25

    # dqg_rot = DualQuaternion.from_dq_array(np.append(dq_list[-1].q_r.elements, [0, 0, 0, 0]))
    rot = Quaternion._from_axis_angle(np.array([0, 0, 1]), .25 * np.pi)
    dq_off = DualQuaternion.from_quat_pose_array(np.append(rot.elements, [.0, .0, .0]))
    # dq_off = dqg_rot.inverse() * dq_off * dqg_rot

    # dq0 = dq_list[0]
    # dqg = dq_list[-1] * dq_off
    dq0 = dq_list[0] * dq_off
    dqg = dq0 * dq_list[0].inverse() * dq_list[-1]
    tw0 = DualQuaternion.from_dq_array(np.zeros(8))
    t_ = np.linspace(0, 2, num=600)
    dq_l, tw_l = dmp_obj.fit_model(t_, dq0, tw0, dqg, tau=tau, twg=None)
    r_, p_ = dmp_obj.pose_from_dq(dq_l)

    # Goal Index
    t_idx = np.where(t_ <= tau * t[-1])[0][-1]
    # Goal Error (Demonstrated vs Reconstructed)
    dq_er = dq_log(dqg.inverse() * dq_l[t_idx])
    d_err = np.linalg.norm(dq_er.q_d.elements)
    q_err = np.linalg.norm(dq_er.q_r.elements)
    print("GOAL POSITION ERROR : %5.5f" % d_err)
    print("GOAL ROTATION ERROR : %5.5f" % q_err)
    # Linear Velocity Error from Twist
    vel_d = Quaternion(vector=dx_dt(t, p)[-1])  # vel_from_twist(dq_list[-1], tw_list[-1])
    vel_f = Quaternion(vector=dx_dt(t_, p_)[t_idx])  # vel_from_twist(dq_l[t_idx], tw_l[t_idx])
    ratio = vel_d.norm / vel_f.norm
    rot_v = quat_rot(vel_d.normalised, vel_f.normalised)
    print(" DEMONSTRATED VEL : " + np.array2string(vel_d.elements[1:], precision=3))
    print("RECONSTRUCTED VEL : " + np.array2string(vel_f.elements[1:], precision=3))
    print("VEL ROT : %s, %5.3f " % (np.array2string(rot_v.axis, precision=3), rot_v.angle))
    print("TAU VS VEL RATIO : " + np.array2string(np.array([tau, ratio]), precision=3))

    """ TRAJECTORY VS TIME """
    plt.figure()
    plt.subplot(221)
    plt.plot(t, r, '--', t_[:t_idx+1], r_[:t_idx+1], t_[t_idx:], r_[t_idx:])
    plt.subplot(223)
    plt.plot(t, p, '--', t_[:t_idx+1], p_[:t_idx+1], t_[t_idx:], p_[t_idx:])
    plt.subplot(222)

    """ TWIST VS TIME """
    tw_list_ = dql_to_npa(tw_list)
    tw_l_ = dql_to_npa(tw_l)
    plt.plot(t, tw_list_[:, 0:5], '--', t_, tw_l_[:, 0:5])
    plt.subplot(224)
    plt.plot(t, tw_list_[:, 5:], '--', t_, tw_l_[:, 5:])

    """ 3D TRAJECTORY """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim3d(0, 1)
    ax.set_ylim3d(0, 1)
    ax.set_zlim3d(0, 1)
    ax.plot(p[:, 0], p[:, 1], p[:, 2], 'k')
    ax.plot(p_[:t_idx+1, 0], p_[:t_idx+1, 1], p_[:t_idx+1, 2], 'r')
    ax.plot(p_[t_idx:, 0], p_[t_idx:, 1], p_[t_idx:, 2], 'c')
    plt.show()


if __name__ == '__main__':
    main()
