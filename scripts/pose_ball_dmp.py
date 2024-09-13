#!/usr/bin/env python

import argparse
import baxter_interface
import cv_bridge
import numpy as np
import rospy
from baxter_core_msgs.msg import AssemblyState, DigitalIOState
from baxter_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
from baxter_interface import CHECK_VERSION, Limb
from custom_tools.ik_tools import IK_Limb, map_file
from custom_tools.dmp_pos import DMP, qconj, qlog, qprod
from custom_tools.pose_from_img import DetectBall, open_camera, publish_marker, reset_cameras
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker


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
    print("ENABLED") if init_state else print("DISABLED")

    def clean_shutdown():
        print("Exiting example...")
        if not init_state:
            print("Disabling robot...")
            rs.disable()

    global flag

    rospy.on_shutdown(clean_shutdown)

    """ Camera parameters """
    dist = np.float32([[0.0, 0.0, 0.0, 0.0, 0.0]]).reshape(-1, 1)
    mtx = np.float32([[410.0, 0.0, 640.0], [0.0, 410.0, 400.0], [0.0, 0.0, 1.0]]).reshape(-1, 3)
    bridge = cv_bridge.CvBridge()
    """ ball_detector parameters """
    radius, hsv_low, hsv_high = .10, [40, 40, 10], [65, 255, 255]  # .11, [4, 80, 20], [15, 255, 255]
    ball_detector = DetectBall(hsv_low, hsv_high, radius, dist, mtx)
    v_ball = Marker()
    v_ball.type = 2
    v_ball.header.frame_id = "/base"
    v_ball.color.r, v_ball.color.g, v_ball.color.b, v_ball.color.blank = [.0, 1., .0, 1.]
    v_ball.scale.x, v_ball.scale.y, v_ball.scale.z = np.multiply(ball_detector.obj_r * 2, np.array([1, 1, 1]))
    """ Subscriber for head and head_camera tf """
    rospy.Subscriber("/tf", TFMessage, ball_detector.tf_callback)
    """ Publisher for Marker msg (RViz) """
    pub = rospy.Publisher("/detected_marker", Marker, queue_size=1)

    def image_callback(ros_img):
        cv_raw = bridge.imgmsg_to_cv2(ros_img, desired_encoding="passthrough")
        """ Pose from ball_detector """
        t_list_b, r_list_b = ball_detector.pose_from_img(cv_raw)
        publish_marker(pub, t_list_b, r_list_b, v_ball)

    """ ROS Image subscriber """
    cam_name = "head_camera"
    print("Starting ball detection...")
    rospy.Subscriber("/cameras/%s/image" % cam_name, Image, image_callback)
    reset_cameras()
    open_camera(cam_name)

    """ Parameters for IK Service """
    ik_limb = IK_Limb(args.limb)

    names, bib = map_file(args.file)
    t = bib[:, 0]
    dt = np.mean(np.diff(t))
    print("File mapped")

    n = int(args.n) if args.n else 50
    tau = float(args.tau) if args.tau else 1
    alphay = 4
    dmp_obj = DMP(n, alphay)

    """ Position DMP """
    print("Modeling %s..." % names[1:4])
    y = bib[:, 1:4]
    y[:, 1] *= mirror
    y0 = y[0, :]
    dy0 = np.diff(y[0:2, :], axis=0)
    yg = y[-1, :]
    wi_y, x = dmp_obj.get_model_p(t, y)

    """ Orientation DMP """
    print("Modeling %s..." % names[4:])
    q = bib[:, 4:]
    q[:, 0] *= mirror
    q[:, 2] *= mirror
    q0 = q[0, :].reshape((1, 4))
    eq = 2 * qlog(qprod(q[-1, :], qconj(q)))
    deq0 = np.diff(eq[0:2, :], axis=0) / dt
    qg = q[-1, :]
    wi_q, x = dmp_obj.get_model_q(t, q)

    """ Moving to initial pose """
    state(True)
    limb = baxter_interface.Limb(args.limb)
    print("Moving to neutral position...")
    limb.move_to_neutral()
    print("Moving to start position...")
    init_joint = ik_limb.ik_solve(y0, q0[0])
    limb.move_to_joint_positions(init_joint)

    """ DMP Demonstration """
    dmp_obj.step_prefit(yg, y0, dy0, qg, q0, deq0)
    yd = np.array([y0.reshape(3), 0 * dy0.reshape(3), np.zeros(3)]).reshape((1, -1))
    qd = np.array([q0[0].reshape(4), 0 * deq0[0].reshape(4), np.zeros(4)]).reshape((1, -1))
    start_time = t[0]
    prev_time = start_time
    for ti in t:
        current_time = ti
        dt = current_time - prev_time
        ti = np.array(current_time - start_time).reshape(-1, 1)
        _x, _dpsi, _fn_p, _fn_q, yd, qd = dmp_obj.step_fit(ti, dt, .5, yd, wi_y, qd, wi_q)
        pos, rot = yd.ravel(), qd.ravel()
        init_joint = ik_limb.ik_solve(pos[0:3], rot[0:4])
        if init_joint:
            limb.set_joint_positions(init_joint, raw=args.raw)
        prev_time = current_time
        rospy.sleep(dt)
    rospy.sleep(1)
    limb.move_to_neutral()

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
        v_mrk_p = v_ball.pose.position
        yg = [v_mrk_p.x, v_mrk_p.y, v_mrk_p.z]
        flag = rospy.wait_for_message(sbutton, DigitalIOState).state
    rospy.sleep(2)
    yg[0] += 1*radius
    # yg[1] -= 0*radius * np.sign(yg[1]) * mirror
    print("Goal position: {0}, {1}, {2}".format(*yg))
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
