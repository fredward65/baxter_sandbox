#!/usr/bin/env python

import argparse
import baxter_interface
import numpy as np
import rospy
from custom_tools import ring_buffer
from baxter_core_msgs.msg import AssemblyState
from baxter_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
from baxter_interface import CHECK_VERSION
from baxter_interface.digital_io import DigitalIO
from baxter_interface.limb import Limb
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from std_msgs.msg import Header
from tf.transformations import quaternion_multiply, euler_from_quaternion as q_to_eu, quaternion_from_euler as eu_to_q
from visualization_msgs.msg import Marker

global flag_move, init_pose, data_queue, mrk_type, target_id
flag_move = False
init_pose = {'right_s0': 0.876670020277,
             'right_s1': -0.0421844716668,
             'right_e0': 0.00191747598486,
             'right_e1': 1.53321379749,
             'right_w0': -0.363936941926,
             'right_w1': 0.106611664758,
             'right_w2': -0.0360485485153}


def state(flag):
    rs = baxter_interface.RobotEnable(CHECK_VERSION)
    while rospy.wait_for_message('/robot/state', AssemblyState).enabled != flag:
        if flag:
            rs.enable()
        else:
            rs.disable()


def build_pose(p, o):
    pose_stamped = PoseStamped()
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    pose = Pose(position=Point(x=p.x, y=p.y, z=p.z), orientation=o)
    pose_stamped.header = hdr
    pose_stamped.pose = pose
    return pose_stamped


def marker_callback(vmrk):
    global mrk_type, target_id
    if flag_move and vmrk.id == target_id:
        mrk_type = vmrk.type
        data_queue.enqueue(vmrk)


def update_state(*args):
    if args[0]:
        print("Changing state...")
        global l, flag_move
        flag_move = not flag_move
        if not flag_move:
            l.move_to_joint_positions(init_pose)


def quat_from_vecs(v1, v2):
    v1 = np.array([v1.x, 0, v1.z])
    v2 = np.array([v2.x, 0, v2.z])
    n = np.cross(v1, v2)
    n = n / np.linalg.norm(n)
    th = np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))
    n = np.multiply(n, np.sin(th / 2))
    q = Quaternion(np.cos(th / 2), n[0], n[1], n[2])
    return q


def rotate_v_q(v, q):
    v = quaternion_multiply([q[0], q[1], q[2], q[3]], [v[0], v[1], v[2], 0])
    v = quaternion_multiply(v, [-q[0], -q[1], -q[2], q[3]])
    return v[0:3]


def follow_marker():
    global side, init_pose, flag_move, data_queue, mrk_type, l
    # mrk = Limb(side).endpoint_pose()
    # pos, rot = mrk["position"], mrk["orientation"]
    if flag_move:
        v_mrk = data_queue.dequeue()
        if v_mrk and not rospy.is_shutdown():
            mrk = v_mrk.pose.position
            q_ = v_mrk.pose.orientation
            q_ = q_to_eu([q_.x, q_.y, q_.z, q_.w], axes='rxyz')
            print(rotate_v_q([0, 0, 1], eu_to_q(q_[0], q_[1], q_[2], axes='rxyz')))
            q_off = eu_to_q(q_[0], q_[1], 0, axes='rxyz')
            ns = "/ExternalTools/%s/PositionKinematicsNode/IKService" % side
            rospy.wait_for_service(ns)
            iksvc = rospy.ServiceProxy(ns, SolvePositionIK)

            mrk_off = [.00, .00 if mrk_type == 1 else .00, .00 if mrk_type == 2 else .12]
            mrk_off = rotate_v_q(mrk_off, q_off)
            [mrk.x, mrk.y, mrk.z] = [mrk.x + mrk_off[0], mrk.y + mrk_off[1], mrk.z + mrk_off[2]]
            q = eu_to_q(0, np.pi / 2, 0, axes='rxyz')
            q = quaternion_multiply(q, q_off)
            o = Quaternion(q[0], q[1], q[2], q[3])
            goal = build_pose(mrk, o)

            ikreq = SolvePositionIKRequest()
            ikreq.pose_stamp.append(goal)
            try:
                resp = iksvc(ikreq)
                if resp.isValid[0]:
                    rospy.loginfo("Success! Valid Joint Solution")
                    joint_angles = dict(zip(resp.joints[0].name, resp.joints[0].position))
                    l.set_joint_positions(joint_angles, raw=False)
                else:
                    rospy.loginfo("Error: No Valid Joint Solution")
            except rospy.ServiceException as e:
                rospy.loginfo("Service call failed: %s" % (e,))
    else:
        pass


def main():
    parser = argparse.ArgumentParser(description="Arm Selector")
    parser.add_argument('-l', '--left', const='left', dest='actions', action='store_const', help='Use left arm')
    parser.add_argument('-r', '--right', const='right', dest='actions', action='store_const', help='Use right arm')
    parser.add_argument('-t', '--target', dest='t_id', help='Assign marker target ID, 0 by default')
    arg = parser.parse_args(rospy.myargv()[1:])
    if arg.actions is None:
        parser.print_usage()
        parser.exit(0, "No arm selected.\n")
    global side, target_id
    side = arg.actions if arg.actions == "right" or arg.actions == "left" else "left"
    m = -1 if side == "right" else 1
    target_id = int(arg.t_id) if arg.t_id else 0

    print("Starting node...")
    rospy.init_node("reach_%s_arm" % side, anonymous=True)
    init_flag = rospy.wait_for_message('/robot/state', AssemblyState).enabled

    global l, init_pose
    l = Limb(side)
    state(True)
    init_pose = l.joint_angles()
    l.move_to_joint_positions(init_pose)

    global data_queue
    data_queue = ring_buffer.RingBuffer(50)

    rospy.Subscriber("/detected_marker", Marker, marker_callback, queue_size=1)

    button = DigitalIO("torso_%s_button_ok" % side)
    button.state_changed.connect(update_state)

    def shutdown_hook():
        print("Killing process")
        l.move_to_joint_positions(init_pose)
        state(init_flag)

    rospy.on_shutdown(shutdown_hook)
    print("Node started")
    print("Press %s torso button to start" % side)

    while not rospy.is_shutdown():
        follow_marker()


if __name__ == "__main__":
    main()
