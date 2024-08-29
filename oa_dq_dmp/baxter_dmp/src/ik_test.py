#!/usr/bin/env python

import baxter_interface
import matplotlib.pyplot as plt
import numpy as np
import rospy
from baxter_core_msgs.msg import AssemblyState, DigitalIOState
from baxter_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
from baxter_interface import CHECK_VERSION, Limb, Gripper
from dual_quaternions import DualQuaternion
from mpl_toolkits import mplot3d
from oa_dq_dmp import OADQMP
from pyquaternion import Quaternion
from custom_tools.ik_tools import IKLimb


def state(s_flag):
    while rospy.wait_for_message('/robot/state', AssemblyState).enabled is not s_flag:
        rs = baxter_interface.RobotEnable(CHECK_VERSION)
        if s_flag:
            rs.enable()
        else:
            rs.disable()


def main():
    """ Test Baxter movement """
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
    ik_limb = IKLimb(limb_)

    """ Initialize movement """
    limb = baxter_interface.Limb(limb_)

    y = np.linspace(0, 1, num=10)
    for yi in y:
        p0 = np.array([.75, yi, .1])
        q0 = np.array([-.707, 0, 0, .707])
        pose = ik_limb.ik_solve(p0, q0)
        if pose:
            print(p0, q0)
            state(True)
            print("Moving to neutral position...")
            limb.move_to_neutral()
            limb.move_to_joint_positions(pose)
            limb.move_to_neutral()
            state(False)


if __name__ == '__main__':
    main()
