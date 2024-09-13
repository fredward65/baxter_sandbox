#!/usr/bin/env python

import argparse, baxter_interface, numpy as np, rospy, sys
from baxter_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
from baxter_interface import CHECK_VERSION
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from std_msgs.msg import Header
from tf.transformations import quaternion_from_euler as eu_to_q


def try_float(x):
    try:
        return float(x)
    except ValueError:
        return None


def clean_line(line, names):
    line = [try_float(x) for x in line.rstrip().split(',')]
    combined = zip(names[1:], line[1:])
    cleaned = [x for x in combined if x[1] is not None]
    command = dict(cleaned)
    return command, line


def build_pose(pose):
    p = Point(x=pose['posx'], y=pose['posy'], z=pose['posz'])
    # q = eu_to_q(pose['rota'], pose['rotb'], pose['rotc'])
    o = Quaternion(pose['rotx'], pose['roty'], pose['rotz'], pose['rotw'])
    goal = PoseStamped()
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    pose = Pose(position=p, orientation=o)
    goal.header = hdr
    goal.pose = pose
    return goal


def map_file(filename, side):
    # grip = baxter_interface.Gripper(side, CHECK_VERSION)
    """
    if grip.error():
        grip.reset()
    if (not grip.calibrated() and grip.type() != 'custom'):
        grip.calibrate()
    """
    print("Solving poses from: %s" % (filename,))

    ns = "/ExternalTools/%s/PositionKinematicsNode/IKService" % side
    rospy.wait_for_service(ns)
    iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
    ikreq = SolvePositionIKRequest()

    with open(filename, 'r') as f:
        lines = f.readlines()
    keys = lines[0].rstrip().split(',')

    bib = []
    vtime = []
    for values in lines[1:]:
        cmd, values = clean_line(values, keys)
        vtime.append(values[0])
        goal = build_pose(cmd)
        ikreq.pose_stamp.append(goal)

    try:
        resp = iksvc(ikreq)
        for i, sresp in enumerate(resp.isValid):
            sresult = "success"
            if resp.isValid[i]:
                # rospy.loginfo("Success! Valid Joint Solution")
                joint_angles = dict(zip(resp.joints[i].name, resp.joints[i].position))
                bib.append(joint_angles)
            else:
                # rospy.loginfo("Error: No Valid Joint Solution")
                sresult = "error  \n\r"
                bib.append(None)
            sys.stdout.write("\r Record %d of %d: %s" % (i, len(resp.isValid) - 1, sresult))
            sys.stdout.flush()
    except rospy.ServiceException as e:
        rospy.loginfo("Service call failed: %s" % (e,))

    return bib, vtime


def main():
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt)
    parser.add_argument('-f', '--file', metavar='PATH', required=True, help='path to input file')
    parser.add_argument('-l', '--limb', dest='side', required=True, help='the limb to record, either left or right')
    parser.add_argument('-r', '--raw', const=False, dest='raw', action='append_const', help='raw joint angles')
    args = parser.parse_args(rospy.myargv()[1:])
    if args.side is (not "left" and not "right") or None:
        parser.print_usage()
        parser.exit(0, "No arm selected.\n")

    print("Initializing node... ")
    rospy.init_node("joint_position_file_playback")
    print("Getting robot state... ")
    rs = baxter_interface.RobotEnable(CHECK_VERSION)
    init_state = rs.state().enabled

    def clean_shutdown():
        print("\nExiting example...")
        if not init_state:
            print("Disabling robot...")
            rs.disable()
    rospy.on_shutdown(clean_shutdown)

    bib, vtime = map_file(args.file, args.side)
    dt = np.mean(np.diff(vtime))
    rate = rospy.Rate(1 / dt)

    i = 0
    while not bib[i] and i < (len(bib) - 1):
        i += 1

    print("Enabling robot... ")
    rs.enable()
    l = baxter_interface.Limb(args.side)
    print("Moving to neutral position...")
    l.move_to_neutral()

    if i < (len(bib)-1):        
        print("\nMoving to start position...")
        l.move_to_joint_positions(bib[i])
        print("Playing back...")
        start_time = rospy.get_time()
        for i, ctime in enumerate(vtime):
            sys.stdout.write("\r Record %d of %d" % (i, len(vtime) - 1))
            sys.stdout.flush()
            while (rospy.get_time() - start_time) < ctime:
                if rospy.is_shutdown():
                    print("\n Aborting - ROS shutdown")
                    return False
                if bib[i]:
                    l.set_joint_positions(bib[i], raw=True if args.raw else False)
                rate.sleep()
    else:
        print("Bad luck, mate")
    rospy.sleep(2)

    print("\nMoving to neutral position...")
    l.move_to_neutral()


if __name__ == '__main__':
    main()
