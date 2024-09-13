#!/usr/bin/env python

import argparse
import baxter_interface
import numpy as np
import rospy
from baxter_interface import CHECK_VERSION
from custom_tools import ring_buffer
from visualization_msgs.msg import Marker

global data_queue, target_id


class HeadHandler(object):
    def __init__(self):
        self._head = baxter_interface.Head()
        # verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        print("Running. Ctrl-c to quit")

    def clean_shutdown(self):
        print("\nExiting example...")
        self.set_neutral()
        if not self._init_state and self._rs.state().enabled:
            print("Disabling robot...")
            self._rs.disable()

    def set_neutral(self):
        self._head.set_pan(0.0, timeout=5)

    def move(self):
        global data_queue
        self.set_neutral()
        # self._head.command_nod()
        command_rate = rospy.Rate(100)
        control_rate = rospy.Rate(1000)
        dt = .2
        angle = 0.0
        vel = 0.0
        while not rospy.is_shutdown():
            v_mrk = data_queue.dequeue()
            if v_mrk:
                v1 = np.array([v_mrk.x, v_mrk.y, 0])
                v2 = np.array([1, 0, 0])
                th = np.sign(v_mrk.y) * np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))
                th = np.clip(th, -.9 * np.pi/2, .9 * np.pi/2)
                acc = 4 * (1 * (th - angle) - vel)
                vel = vel + acc * dt
                angle = angle + vel * dt
                while (not rospy.is_shutdown() and
                       not (abs(self._head.pan() - angle) <= baxter_interface.HEAD_PAN_ANGLE_TOLERANCE)):
                    self._head.set_pan(angle, speed=.5, timeout=0)
                    control_rate.sleep()
            command_rate.sleep()


def marker_callback(v_mrk):
    global target_id
    if v_mrk.id == target_id:
        data_queue.enqueue(v_mrk.pose.position)


def main():
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt, description=main.__doc__)
    parser.add_argument('-t', '--target', default=0, dest='t_id', help='Assign marker target ID, 0 by default')
    arg = parser.parse_args(rospy.myargv()[1:])

    global data_queue, target_id
    data_queue = ring_buffer.RingBuffer(50)
    target_id = int(arg.t_id)

    print("Initializing node... ")
    rospy.init_node("head_aim_node")
    rospy.Subscriber("/detected_marker", Marker, marker_callback, queue_size=1)
    head = HeadHandler()
    rospy.on_shutdown(head.clean_shutdown)
    print("Panning head... ")
    head.move()
    print("Done.")


if __name__ == '__main__':
    main()
