#!/usr/bin/env python

import argparse
import baxter_interface
import cv_bridge
import matplotlib.pyplot as plt
import numpy as np
import rospy
from baxter_core_msgs.msg import AssemblyState, DigitalIOState
from baxter_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
from baxter_interface import CHECK_VERSION, Limb, Gripper
from custom_tools.ik_tools import IK_Limb, map_file
from custom_tools.dq_dmp import DQDMP
from custom_tools.pose_from_img import DetectBall, open_camera, publish_marker, reset_cameras
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker
from dual_quaternions import DualQuaternion


def state(s_flag):
    while rospy.wait_for_message('/robot/state', AssemblyState).enabled is not s_flag:
        rs = baxter_interface.RobotEnable(CHECK_VERSION)
        if s_flag:
            rs.enable()
        else:
            rs.disable()


def main():
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt)
    required = parser.add_argument_group('required arguments', description=main.__doc__)
    required.add_argument('-f', '--file', metavar='PATH', required=True, help='path to input file')
    required.add_argument('-l', '--limb', dest='limb', required=True, help='the limb to use, either left or right')
    parser.add_argument('-r', '--raw', default=False, dest='raw', help='raw joint values')
    parser.add_argument('-m', '--mirror', const=True, dest='mirror', action='append_const', help='mirror cartesian values')
    parser.add_argument('-n', '--ngaus', dest='n', help='number of gaussian kernels, default n = 50')
    parser.add_argument('-t', '--tau', dest='tau', help='time scale factor, default tau = 1')
    args = parser.parse_args(rospy.myargv()[1:])
    if args.limb is (not "left" and not "right") or None:
        parser.print_usage()
        parser.exit(0, "No arm selected.\n")
    mirror = -1 if args.mirror else 1

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

    global flag

    rospy.on_shutdown(clean_shutdown)

    """ Parameters for IK Service """
    ik_limb = IK_Limb(args.limb)

    names, bib = map_file(args.file)
    t = bib[:, 0]
    t = np.linspace(0, t[-1], num=len(t))
    dt = np.mean(np.diff(t))
    print("File mapped")

    grip = baxter_interface.Gripper(args.limb, CHECK_VERSION)
    if grip.error():
        grip.reset()
    if (not grip.calibrated() and grip.type() != 'custom'):
        grip.calibrate()
    grip.command_position(75, block=True)

    n = int(args.n) if args.n else 50
    tau = float(args.tau) if args.tau else 1
    alphay = 8
    dmp_obj = DQDMP(n, alphay)

    """ Dual Quaternion DMP """
    print("Modeling %s..." % names[1:])
    dqtr = []
    for bibi in bib:
        dqdata = np.append(bibi[4:8], bibi[1:4])
        dqtr.append(DualQuaternion.from_quat_pose_array(dqdata))
    dq0 = dqtr[0]
    dqg = dqtr[-1]
    tw0 = dmp_obj.twist_from_dq(t, dqtr)[0, :]
    # dedq0 = dmp_obj.dedq_from_dq(t, dqtr, dqg)[0, :]
    qd, yd = dmp_obj.pose_from_dq(dqtr)

    wi, x = dmp_obj.train_model(t, dqtr)
    print(wi.shape)

    t2 = np.linspace(0, t[-1]*tau*3, num=np.floor(len(t)*tau))
    # dqrs = dmp_obj.fit_model(t2, dq0, dedq0, dqg, tau=tau)
    dqrs = dmp_obj.fit_model(t2, dq0, tw0, dqg, tau=tau)
    r, p = dmp_obj.pose_from_dq(dqrs)

    fig, axs = plt.subplots(2)
    axs[0].plot(t, yd, '--', t2, p, ':')
    axs[1].plot(t, qd, '--', t2, r, ':')
    plt.show()

    """ Moving to initial pose """
    state(True)
    limb = baxter_interface.Limb(args.limb)
    print("Moving to neutral position...")
    limb.move_to_neutral()
    print("Moving to start position...")
    init_joint = ik_limb.ik_solve(yd[0], qd[0])
    limb.move_to_joint_positions(init_joint)

    # """ Reproduce demonstration data """
    # start_time = t[0]
    # prev_time = start_time
    # print("Moving...")
    # for (ti, qi, yi) in zip(t, qd, yd):
    #     current_time = ti
    #     dt = current_time - prev_time
    #     init_joint = ik_limb.ik_solve(yi, qi)
    #     if init_joint:
    #         limb.set_joint_positions(init_joint, raw=args.raw)
    #     prev_time = current_time
    #     rospy.sleep(dt)
    # rospy.sleep(1)
    # limb.move_to_neutral()
    # rospy.sleep(1)
    #
    # print("Moving to start position...")
    # init_joint = ik_limb.ik_solve(yd[0], qd[0])
    # limb.move_to_joint_positions(init_joint)

    """ DMP Demonstration """
    tfl = []
    yfl = []
    qfl = []
    dqg = dqg * DualQuaternion.from_quat_pose_array([1, 0, 0, 0, .5, .0, .1])
    dq = dq0
    tw = tw0
    # dedq = dedq0
    start_time = rospy.get_time() # t2[0]
    current_time = start_time
    prev_time = current_time
    ti = current_time - start_time
    tf = rospy.Duration.from_sec(t2[-1])
    print("Moving...")
    while rospy.Duration.from_sec(ti) <= tf:
        current_time = rospy.get_time()
        ti = current_time - start_time
        # dq, dedq = dmp_obj.fit_step(ti, dq, dedq, dq0, dqg, tau=tau)
        dq, tw = dmp_obj.fit_step(ti, dq, tw, dq0, dqg, tau=tau)
        qf, yf = dmp_obj.pose_from_dq([dq])
        tfl.append(ti)
        yfl.append(yf[0])
        qfl.append(qf[0])
        init_joint = ik_limb.ik_solve(yf[0], qf[0])
        if init_joint:
            limb.set_joint_positions(init_joint, raw=args.raw)

        if rospy.Duration.from_sec(ti) >= rospy.Duration.from_sec(t[-1]*tau):
            grip.open(block=False)
        # dt = current_time - prev_time
        # rospy.sleep(dt)
        prev_time = current_time
    limb.move_to_neutral()
    state(False)

    yfl = np.array(yfl).reshape((-1, 3))
    qfl = np.array(qfl).reshape((-1, 4))
    fig, axs = plt.subplots(2)
    axs[0].plot(t, yd, '--', t2, p, ':', tfl, yfl)
    axs[1].plot(t, qd, '--', t2, r, ':', tfl, qfl)
    plt.show()


def moon():
    """ Moving to initial pose """
    print("Moving to neutral position...")
    limb.move_to_neutral()
    print("Moving to start position...")
    init_joint = ik_limb.ik_solve(y0, q0[0])
    limb.move_to_joint_positions(init_joint)

    """ Reading ball position """
    sbutton = '/robot/digital_io/torso_%s_button_ok/state' % args.limb
    flag = False
    print("Press button to start movement")
    while not flag:
        flag = rospy.wait_for_message(sbutton, DigitalIOState).state
    rospy.sleep(2)
    yg[0] += 1*radius
    # yg[1] -= 0*radius * np.sign(yg[1]) * mirror
    dy0, deq0 = dmp_obj.step_prefit(yg, y0, dy0, qg, q0, deq0)
    yd = np.array([y0.reshape(3), 0 * dy0.reshape(3), np.zeros(3)]).reshape((1, -1))
    qd = np.array([q0[0].reshape(4), 0 * deq0[0].reshape(4), np.zeros(4)]).reshape((1, -1))

    flag = False

    def button_callback(msg):
        global flag
        flag = msg.state

    rospy.Subscriber(sbutton, DigitalIOState, button_callback)

    print("Starting DMP movement...")
    goal_joint = ik_limb.ik_solve(yg, qg)
    start_time = rospy.get_time()
    prev_time = start_time
    while not flag and goal_joint:
        current_time = rospy.get_time()
        dt = current_time - prev_time
        ti = np.array(current_time - start_time).reshape(-1, 1)
        _x, _dpsi, _fn_p, _fn_q, yd, qd = dmp_obj.step_fit(ti, dt, tau, yd, wi_y, qd, wi_q)
        pos, rot = yd.ravel(), qd.ravel()
        init_joint = ik_limb.ik_solve(pos[0:3], rot[0:4])
        if init_joint:
            limb.set_joint_positions(init_joint, raw=args.raw)
        prev_time = current_time
    print("DMP movement done" if goal_joint else "DMP not possible")

    print("Moving to neutral position...")
    limb.move_to_neutral()

    state(False)


if __name__ == '__main__':
    main()
