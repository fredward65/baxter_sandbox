#!/usr/bin/env python

import argparse
import baxter_interface
import rospy
from baxter_interface import CHECK_VERSION
from custom_tools import ring_buffer
from visualization_msgs.msg import Marker
from tf.transformations import euler_from_quaternion as q_to_eu

global rs, data_queue


def stop_callback():
    global rs
    rs.disable()


def pose_callback(ref_msg):
    global data_queue
    data_queue.enqueue(ref_msg.pose)


def main():
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt, description=main.__doc__)
    required = parser.add_argument_group('required arguments')
    required.add_argument('-f', '--file', dest='filename', required=True,
                          help='the file name to record to')
    parser.add_argument('-r', '--record-rate', type=int, default=100, dest='rate',
                        help='rate at which to record (default: 100)')
    args = parser.parse_args(rospy.myargv()[1:])

    print("Initializing node... ")
    rospy.init_node("end_effector_pose_recorder")
    print("Getting robot state... ")
    global rs
    rs = baxter_interface.RobotEnable(CHECK_VERSION)
    print("Enabling robot... ")
    rs.enable()

    global data_queue
    data_queue = ring_buffer.RingBuffer(50)
    _rate = rospy.Rate(args.rate)

    # rospy.Subscriber("/end_effector_pose", Marker, pose_callback)

    rospy.on_shutdown(stop_callback)

    print("Recording. Press Ctrl-C to stop.")

    with open(args.filename, 'w') as f:
        f.write('time,posx,posy,posz,rota,rotb,rotc\n')
        _start_time = rospy.get_time()
        while not rospy.is_shutdown():
            mrk = baxter_interface.Limb("left").endpoint_pose()  # data_queue.dequeue()
            if mrk:
                try:
                    pos = mrk["position"]
                    rot = mrk["orientation"]
                    f.write("%f," % (rospy.get_time() - _start_time,))
                    f.write(str(pos.x) + ',' + str(pos.y) + ',' + str(pos.z) + ',')
                    eu = q_to_eu([rot.x, rot.y, rot.z, rot.w])
                    f.write(str(eu[0]) + ',' + str(eu[1]) + ',' + str(eu[2]) + '\n')
                except ValueError:
                    print("Error receiving pose data")
                _rate.sleep()

    print("\nDone.")


if __name__ == '__main__':
    main()
