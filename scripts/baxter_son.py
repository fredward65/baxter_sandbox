#!/usr/bin/env python

import cv2
import cv_bridge
import numpy as np
import rospy
import time

from baxter_interface.digital_io import DigitalIO
from custom_tools.draw_on_screen import draw_eyes
from sensor_msgs.msg import Image


class DrawFace(object):
    def __init__(self, scr_size=(800, 1280, 3)):
        rospy.init_node("draw_face_node", anonymous=True)
        self.pub = rospy.Publisher("/robot/xdisplay", Image, queue_size=1)
        self.button = DigitalIO("torso_left_button_back")
        self.button.state_changed.connect(self.update_state)
        self.scr_size = scr_size
        self.scr_blank = 255 * np.ones(self.scr_size, np.uint8)
        self.dt = .05
        self.c_mrk = np.array([0, 0, 1]).reshape(3, 1)
        self.s_mrk = np.array([0, 0, 0]).reshape(3, 1)
        self.flag = False
        self.x = 0
        self.y = 0

    @staticmethod
    def img2msg(img):
        return cv_bridge.CvBridge().cv2_to_imgmsg(cv2.resize(img, (1024, 600)), encoding="passthrough")

    def update_state(self, *args):
        if args[0]:
            self.flag = not self.flag

    def shutdown_hook(self):
        cv_face = np.zeros(self.scr_size, np.uint8)
        msg = self.img2msg(cv_face)
        self.pub.publish(msg)
        cv2.destroyAllWindows()

    def face_publish(self):
        v_mrk = None
        t_mrk = np.array([self.x, self.y, 1]).reshape(3, 1)
        acc = 4 * (1 * (t_mrk - self.c_mrk) - self.s_mrk)
        self.s_mrk = self.s_mrk + acc * self.dt
        self.c_mrk = self.c_mrk + self.s_mrk * self.dt
        cv_face = draw_eyes(self.scr_blank.copy(), self.c_mrk)
        msg = self.img2msg(cv_face)
        self.pub.publish(msg)
        if not self.flag:
            self.x = np.clip(np.random.randn(), -.75, .75) if np.floor(time.time()) % 5 == 0 else self.x
            self.y = np.clip(np.random.randn(), -.75, .75) if np.floor(time.time()) % 3 == 0 else self.y
        else:
            self.x = -.75
            self.y = 0
        cv2.waitKey(1)

    def loop(self):
        rate = rospy.Rate(1 / self.dt)
        while not rospy.is_shutdown():
            self.face_publish()
            rate.sleep()


def main():
    draw_face = DrawFace()
    rospy.on_shutdown(draw_face.shutdown_hook)
    draw_face.loop()


if __name__ == "__main__":
    main()
