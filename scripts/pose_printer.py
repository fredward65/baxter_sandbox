#!/usr/bin/env python

import argparse
import rospy
import baxter_interface
from baxter_interface import CHECK_VERSION, Limb
from baxter_core_msgs.msg import DigitalIOState
from tf.transformations import euler_from_quaternion as q_to_eu

def main():
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt, description=main.__doc__)
    required = parser.add_argument_group('required arguments')
    required.add_argument('-l', '--limb', dest='limb', required=True,
                          help='the limb to record, either left or right')
    parser.add_argument('-r', '--record-rate', type=int, default=100, dest='rate',
                        help='rate at which to record (default: 100)')
    args = parser.parse_args(rospy.myargv()[1:])
    if args.limb is (not "left" and not "right") or None:
        parser.print_usage()
        parser.exit(0, "No arm selected.\n")

    print("Initializing node... ")
    rospy.init_node("end_effector_pose_printer")

    _limb = Limb(args.limb)
    _rate = rospy.Rate(args.rate)

    _start_time = rospy.get_time()
    p_time = 0
    dt = 0
    while not rospy.is_shutdown():
        mrk = _limb.endpoint_pose()
        c_time = rospy.get_time() - _start_time
        dt = abs(c_time - p_time)
        print(dt)
        p_time = c_time
        _rate.sleep()

    print("\nDone.")


if __name__ == '__main__':
    main()
