#!/usr/bin/env python

import baxter_interface
import numpy as np
import rospy
from baxter_core_msgs.msg import AssemblyState
from baxter_interface import CHECK_VERSION
from custom_tools.ik_tools import IKLimb
from dual_quaternions import DualQuaternion
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from oa_dq_dmp import OADQMP


def state(s_flag):
    while rospy.wait_for_message('/robot/state', AssemblyState).enabled is not s_flag:
        rs = baxter_interface.RobotEnable(CHECK_VERSION)
        if s_flag:
            rs.enable()
        else:
            rs.disable()


def create_mrk(id_, type_, scale_, color_):
    mrk = Marker()

    mrk.header.frame_id = "base"

    mrk.id = id_
    mrk.type = type_

    mrk.color.r = color_[0]
    mrk.color.g = color_[1]
    mrk.color.b = color_[2]
    mrk.color.a = 1

    mrk.pose.position.x = 0
    mrk.pose.position.y = 0
    mrk.pose.position.z = 0
    mrk.pose.orientation.x = 0.0
    mrk.pose.orientation.y = 0.0
    mrk.pose.orientation.z = 0.0
    mrk.pose.orientation.w = 1.0

    mrk.scale.x = scale_[0]
    mrk.scale.y = scale_[1]
    mrk.scale.z = scale_[2]

    return mrk


def create_point(pi):
    pi_ = Point()
    pi_.x = pi[0]
    pi_.y = pi[1]
    pi_.z = pi[2]
    return pi_


def gen_data(n, off=np.array([.10, .0, -.10])):
    import quaternion as quat

    t = np.linspace(0, 1, num=n)
    y0, yf = [0, 1]
    x = 0 * t
    y = 1 - (y0 + (yf - y0) * (6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3))
    z = .25 * yf - (y - .5 * yf) ** 2
    tr_p = np.multiply([1, .45, .3], np.c_[x, y, z]) + off
    tr_r = quat.as_float_array(quat.from_euler_angles(x - .5 * np.pi  + ((4 * z) / np.pi), x, x))

    dqtr = []
    for p, r in zip(tr_p, tr_r):
        dqtr.append(DualQuaternion.from_quat_pose_array(np.append(r, p)))

    t *= 3

    return t, dqtr, tr_p, tr_r


def main():
    """ Main function """
    """ DMP-Related Variables """
    # Generate MJT data
    t, dqtr, p, r = gen_data(300, off=np.array([0.50, 0.10, 0.10]))

    # DMP Training
    n = 100
    alpha_y = 12
    dmp_obj = OADQMP(n, alpha_y, gamma_oa=20, beta_oa=12/np.pi)
    dmp_obj.train_model(t, dqtr)

    # Initial pose values
    dq0 = dqtr[0]
    dqg = dqtr[-1]
    tw0 = np.zeros((1, 8))[0]
    idx = int(.5 * len(t)) - 1
    dqo = dqtr[idx]

    # Time-related values
    tau = 1
    fac = 1.5
    t_ = np.linspace(0, tau * fac * t[-1], num=np.floor(.5 * fac * t.shape[0]).astype('int'))
    dmp_obj.reset_t(t_[0])

    """ ROS Node """
    rospy.init_node("dmp_pose_modeling")
    rospy.loginfo("Initializing node... ")
    rospy.loginfo("Getting robot state... ")
    rs = baxter_interface.RobotEnable(CHECK_VERSION)
    init_state = rs.state().enabled
    rospy.loginfo("Currently" + "ENABLED" if init_state else "DISABLED")

    def clean_shutdown():
        rospy.loginfo("Exiting example...")
        if not init_state:
            rospy.loginfo("Disabling robot...")
            rs.disable()

    rospy.on_shutdown(clean_shutdown)

    mrk_pub = rospy.Publisher("/markers", Marker, queue_size=1)

    path = create_mrk(0, 4, (.005, 0, 0), (0, 0, 1))

    """ Parameters for IK Service """
    limb_ = 'left'
    ik_limb = IKLimb(limb_)

    """ Initialize movement """
    limb = baxter_interface.Limb(limb_)
    pose_0 = None
    pose_g = None

    while not pose_0 or not pose_g:
        pose_0 = ik_limb.ik_solve(dq0.translation(), dq0.q_r.elements)
        pose_g = ik_limb.ik_solve(dqg.translation(), dqg.q_r.elements)

    state(True)
    rospy.loginfo("Moving to neutral pose...")
    limb.move_to_neutral()
    rospy.loginfo("Moving to initial pose...")
    limb.move_to_joint_positions(pose_0)

    rate = 1 / np.mean(np.diff(t))
    rate_ = rospy.Rate(rate)

    pose = None

    """ Demonstrated path """
    for ti, pi, qi in zip(t, p, r):
        pose_ = ik_limb.ik_solve(pi, qi)
        if pose_:
            pose = pose_
            limb.set_joint_positions(pose, raw=True)
        path.header.stamp = rospy.Time.now()
        path.points.append(create_point(pi))
        mrk_pub.publish(path)
        rate_.sleep()
    limb.move_to_joint_positions(pose)
    rospy.sleep(2)

    """ DMP Reconstructed path """
    obstacle = create_mrk(1, 2, (.025, .025, .025), (1, 0, 1))
    obstacle.pose.position.x = p[idx, 0]
    obstacle.pose.position.y = p[idx, 1]
    obstacle.pose.position.z = p[idx, 2]

    path = create_mrk(2, 4, (.005, 0, 0), (0, 1, 0))

    dq = dq0
    tw = tw0

    rospy.loginfo("Moving to initial pose...")
    limb.move_to_joint_positions(pose_0)
    mrk_pub.publish(obstacle)

    ti = 0.0
    tf = t_[-1] + 1
    dmp_obj.reset_t(ti)

    rospy.loginfo("Moving...")
    while ti <= tf:
        pi = dq.translation()
        pose = ik_limb.ik_solve(pi, dq.q_r.elements)
        path.points.append(create_point(pi))
        mrk_pub.publish(path)
        if pose:
            limb.set_joint_positions(pose, raw=True)
        dq, tw = dmp_obj.fit_step(ti, dq, tw, dq0, dqg, tau=tau, dqo=dqo)
        ti = ti + (1.0/rate)
        rate_.sleep()
    limb.move_to_joint_positions(pose)
    rospy.sleep(2)

    rospy.loginfo("Moving to neutral pose...")
    limb.move_to_neutral()
    state(False)


if __name__ == '__main__':
    main()
