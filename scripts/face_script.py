#!/usr/bin/env python

import argparse
import cv2
import cv_bridge
import numpy as np
import rospy

from custom_tools import ring_buffer
from custom_tools.draw_on_screen import draw_eyes
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker

DIR = './src/baxter_sandbox/scripts/aruco_tools/'
bridge = cv_bridge.CvBridge()

global data_queue, pub, target_id, c_mrk, s_mrk


def marker_callback(v_mrk):
    global target_id
    if v_mrk.id == target_id:
        data_queue.enqueue(v_mrk.color)


def face_publish(dt):
    global data_queue, pub, target_id, c_mrk, s_mrk
    v_mrk = data_queue.dequeue()
    t_mrk = np.array([v_mrk.r, v_mrk.g, v_mrk.b]).reshape(3, 1) if v_mrk else np.array([0, 0, 1]).reshape(3, 1)
    acc = 4 * (1 * (t_mrk - c_mrk) - s_mrk)
    s_mrk = s_mrk + acc * dt
    c_mrk = c_mrk + s_mrk * dt
    cv_face = draw_eyes(255 * np.ones((800, 1280, 3), np.uint8), c_mrk)
    msg = cv_bridge.CvBridge().cv2_to_imgmsg(cv2.resize(cv_face, (1024, 600)), encoding="passthrough")
    pub.publish(msg)
    cv2.waitKey(1)


def main():
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt, description=main.__doc__)
    parser.add_argument('-t', '--target', default=0, dest='t_id', help='Assign marker target ID, 0 by default')
    arg = parser.parse_args(rospy.myargv()[1:])

    global data_queue, target_id, pub, c_mrk, s_mrk
    data_queue = ring_buffer.RingBuffer(50)
    target_id = int(arg.t_id)

    rospy.init_node("face_eyes_node", anonymous=True)
    pub = rospy.Publisher("/robot/xdisplay", Image, queue_size=1)
    rospy.Subscriber("/detected_marker", Marker, marker_callback, queue_size=1)
    c_mrk = np.array([0, 0, 1]).reshape(3, 1)
    s_mrk = np.array([0, 0, 0]).reshape(3, 1)

    def shutdown_hook():
        cv_face = np.zeros((800, 1280, 3), np.uint8)
        msg = cv_bridge.CvBridge().cv2_to_imgmsg(cv2.resize(cv_face, (1024, 600)), encoding="passthrough")
        pub.publish(msg)
        cv2.destroyAllWindows()
    rospy.on_shutdown(shutdown_hook)

    dt = .05
    rate = rospy.Rate(1/dt)
    while not rospy.is_shutdown():
        face_publish(dt)
        rate.sleep()


if __name__ == "__main__":
    main()
