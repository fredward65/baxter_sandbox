#!/usr/bin/env python

import baxter_interface
import numpy as np
import quaternion
import rospy
from throwing.throw_dmp import DMP
from matplotlib import pyplot as plt
from throwing.projectile_model import ProjectileModel
from mpl_toolkits import mplot3d
from baxter_core_msgs.msg import AssemblyState, DigitalIOState
from baxter_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
from baxter_interface import CHECK_VERSION, Limb, Gripper
from custom_tools.ik_tools import IK_Limb, map_file


def gen_data(n):
    t = np.linspace(0, 1, num=n)
    x0, xf = [0, 1]
    x = x0 + (xf - x0) * (6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3)
    y = 0 * t
    z = .25 * xf - (x - .5 * xf) ** 2
    tr_p = np.c_[x, y, z]
    return t, tr_p


def get_data(fname):
    """ Read data from CSV file """
    traj = np.recfromcsv(fname)
    t = np.array([])
    tr_p = np.array([])
    for tr in traj:
        tr_ = tr.tolist()
        t = np.append(t, tr_[0])
        p = np.array(tr_[1:4])
        tr_p = np.append(tr_p, p).reshape((-1, 3))
    # tr_p[:, 0] *= -1
    # tr_p[:, 1] *= -1
    # tr_p[:, 2] /= 5
    t = np.linspace(t[0], t[-1], t.shape[0])
    return t, tr_p


def state(s_flag):
    while rospy.wait_for_message('/robot/state', AssemblyState).enabled is not s_flag:
        rs = baxter_interface.RobotEnable(CHECK_VERSION)
        if s_flag:
            rs.enable()
        else:
            rs.disable()


def main():
    """ Get/Generate data """
    # Get data from CSV
    t, tr_p = get_data('./trj_files/demo_throw.csv')
    # Generate MJT data
    # t, dqtr, tr_p, tr_r = gen_data(250)

    """ DMP Training """
    # DMP parameters
    n = 50
    alpha_y = 8
    dmp_obj = DMP(n, alpha_y)
    # Train model
    dmp_obj.train_model(t, tr_p)

    """ DMP Reconstruction """
    # Initial values
    off = np.array([.8, .8, .0])
    y0 = tr_p[0, :] + off
    yg = tr_p[-1, :] + off
    dy0 = np.diff(tr_p[0:2, :], axis=0) / np.diff(t[0:2])

    qd = quaternion.as_float_array(quaternion.from_euler_angles(0, .4*np.pi, 0))

    # Projectile Model
    pm = ProjectileModel()
    ph = yg
    pg = np.array([1., 0., .5])
    th = np.deg2rad(80)
    dyg, tf = pm.solve3d(ph, pg, th)
    tr_l, t_l = pm.evaluate3d(ph)
    # Time scaling from velocities
    f1 = np.linalg.norm(dmp_obj.dygd) / np.linalg.norm(dyg)
    # f2 = np.linalg.norm(dmp_obj.ygd - dmp_obj.y0d) / np.linalg.norm(yg - y0)
    tau = 1 * f1  # / f2
    print(tau, dyg, dmp_obj.dygd)
    # New time vector
    t_ = np.linspace(0, tau * t[-1], num=np.floor(t[-1]*200).astype('int'))
    # Reconstruct model
    tr_f, _, _ = dmp_obj.fit_model(t_, y0, dy0, yg, tau=tau, dyg=dyg)
    # Reconstruct Projectile Launch
    pf = tr_f[-1, :]
    dyf = np.divide(np.diff(tr_f, axis=0), np.diff(t_).reshape((-1, 1)))[-1, :]
    print(tf, dyg, dyf)
    x_ = pm.xy_eq(t_l, pf[-3], dyf[-3])
    y_ = pm.xy_eq(t_l, pf[-2], dyf[-2])
    z_ = pm.z_eq(t_l, pf[-1], dyf[-1])
    tr_r = np.c_[x_, y_, z_]
    # Dart aim error
    err = np.linalg.norm(tr_r[-1, :] - pg)

    print("Initializing node... ")
    rospy.init_node("dmp_pose_modeling")
    print("Getting robot state... ")
    rs = baxter_interface.RobotEnable(CHECK_VERSION)
    init_state = rs.state().enabled
    txt = "ENABLED" if init_state else "DISABLED"
    print(txt)

    def clean_shutdown():
        print("Exiting example...")
        if not init_state:
            print("Disabling robot...")
            rs.disable()

    rospy.on_shutdown(clean_shutdown)

    """ Parameters for IK Service """
    limb_ = 'left'
    ik_limb = IK_Limb(limb_)

    grip = baxter_interface.Gripper(limb_, CHECK_VERSION)
    if grip.error():
        grip.reset()
    if (not grip.calibrated() and grip.type() != 'custom'):
        grip.calibrate()
    grip.command_position(75, block=True)

    plt.plot(t, tr_p, '--k', t_, tr_f)
    plt.show()

    """ Initialize movement """
    limb = baxter_interface.Limb(limb_)
    pose_0 = ik_limb.ik_solve(y0, qd)
    pose_g = ik_limb.ik_solve(yg, qd)
    if pose_0 and pose_g:
        state(True)
        print("Moving to neutral position...")
        limb.move_to_neutral()
        print("Moving to start position...")
        init_joint = ik_limb.ik_solve(y0, qd)
        limb.move_to_joint_positions(init_joint)
        """ DMP Demonstration """
        tfl = []
        yfl = []
        y = y0
        dy = dy0
        start_time = rospy.get_time()  # t2[0]
        current_time = start_time
        prev_time = current_time
        ti = current_time - start_time
        tf = rospy.Duration.from_sec(t_[-1] + 2)
        tl = rospy.Duration.from_sec(t_[-1])
        print("Moving...")
        dmp_obj.reset_t()
        while rospy.Duration.from_sec(ti) <= tf:
            current_time = rospy.get_time()
            ti = current_time - start_time
            y, dy, _ = dmp_obj.fit_step(ti, y, dy, yg, y0, tau=tau, dyg=dyg)
            tfl.append(ti)
            yfl.append(y)
            init_joint = ik_limb.ik_solve(y[0], qd)
            if init_joint:
                limb.set_joint_positions(init_joint, raw=True)
            print(ti)
            if rospy.Duration.from_sec(ti) > tl:
                print("THROW!")
                grip.open(block=False)
            prev_time = current_time
        print(1 / np.mean(np.diff(tfl)))
        yfl = np.array(yfl).reshape((-1, 3))
        limb.move_to_neutral()
        state(False)

        """ Plot results """
        # # 2D Plot
        # plt.plot(t, tr_p, '--', t_, tr_f)
        # plt.plot(t_[-1] + t_l, tr_l, ':b')
        # plt.plot(t_[-1].reshape((-1, 1)), tr_l[0, :].reshape((-1, 3)), 'xb')
        # plt.plot((t_[-1]+t_l[-1]).reshape((-1, 1)), tr_l[-1, :].reshape((-1, 3)), 'xb')
        # plt.plot(t_[-1] + t_l, tr_r, '-.r')
        # plt.plot(t_[-1].reshape((-1, 1)), tr_r[0, :].reshape((-1, 3)), 'xr')
        # plt.plot((t_[-1]+t_l[-1]).reshape((-1, 1)), tr_r[-1, :].reshape((-1, 3)), 'xr')
        # 3D Plot
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(tr_f[:, 0], tr_f[:, 1], tr_f[:, 2], '--k')
        ax.plot3D(yfl[:, 0], yfl[:, 1], yfl[:, 2])
        ax.plot3D(tr_p[:, 0], tr_p[:, 1], tr_p[:, 2], '--k')
        ax.plot3D([y0[0]], [y0[1]], [y0[2]], '*b')
        ax.plot3D([yg[0]], [yg[1]], [yg[2]], '*b')
        ax.plot3D(tr_f[:, 0], tr_f[:, 1], tr_f[:, 2])
        ax.plot3D([tr_f[-1, 0]], [tr_f[-1, 1]], [tr_f[-1, 2]], '*r')
        ax.plot3D(tr_l[:, 0], tr_l[:, 1], tr_l[:, 2], ':g')
        ax.plot3D([tr_l[-1, 0]], [tr_l[-1, 1]], [tr_l[-1, 2]], 'xb')
        ax.plot3D(tr_r[:, 0], tr_r[:, 1], tr_r[:, 2], '-.m')
        ax.plot3D([tr_r[-1, 0]], [tr_r[-1, 1]], [tr_r[-1, 2]], 'xr')
        lims = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).T
        true_lims = [min(lims[0]), max(lims[1])]
        ax.set_xlim3d(true_lims)
        ax.set_ylim3d(true_lims)
        ax.set_zlim3d(true_lims)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        plt.title('Aim error: %.3f m' % err)
        plt.show()


if __name__ == "__main__":
    main()
