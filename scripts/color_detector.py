#!/usr/bin/env python

import cv2
import cv_bridge
import numpy as np
import rospy
from custom_tools import ring_buffer
from sensor_msgs.msg import Image

bridge = cv_bridge.CvBridge()

global data_queue

# Camera distortion parameters from Baxter's head_camera
dist = np.float32([[0.0, 0.0, 0.0, 0.0, 0.0]]).reshape(-1, 1)
mtx = np.float32([[410.0, 0.0, 640.0], [0.0, 410.0, 400.0], [0.0, 0.0, 1.0]]).reshape(-1, 3)
coef = np.float32([[-410.0, -0.0, 639.0, 0.0, 0.0, -410.0, 399.0, 0.0, 0.0, 0.0, 1.0, 0.0]]).reshape(-1, 3)


def tb_callback(val):
    pass


def get_contours(cv_img, img):
    im, cont, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in cont:
        area = cv2.contourArea(cnt)
        if 500 < area < 50000:
            cv2.drawContours(cv_img, cnt, -1, (0, 255, 0), 1)
            per = cv2.arcLength(cnt, True)
            apr = cv2.approxPolyDP(cnt, 0.02*per, True)
            x, y, w, h = cv2.boundingRect(apr)
            sides = len(apr)
            if sides < 5:
                cv2.rectangle(cv_img, (x, y), (x+w, y+h), (0, 255, 255), 1)
                color = (0, 255, 255)
            else:
                cv2.circle(cv_img, (x + w/2, y + h/2), w/2 if w > h else h/2, (255, 255, 0), 2)
                color = (255, 255, 0)
            cv2.putText(cv_img, "Vrtx: %i" % sides, (x, y + h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(cv_img, "Area: %i" % area, (x, y + h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return cv_img


def img_processor():
    global data_queue
    ros_img = data_queue.dequeue()
    if ros_img:
        cv_raw = bridge.imgmsg_to_cv2(ros_img, desired_encoding="passthrough")
        cv_img = cv2.undistort(cv_raw, mtx, dist, coef, mtx)
        cv_img = cv2.blur(cv_img, (4, 4))

        cv_img_hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)  # hue, sat, val = cv2.split(cv_img_hsv)

        # Sliders to set two values (v_min and v_max); Hue [0, 180], Sat [0, 255], Val [0, 255]
        cv2.namedWindow('Color Mask')
        cv2.createTrackbar('Val Min', 'Color Mask', 0, 255, tb_callback)
        cv2.createTrackbar('Val Max', 'Color Mask', 0, 255, tb_callback)
        v_min = cv2.getTrackbarPos('Val Min', 'Color Mask')
        v_max = cv2.getTrackbarPos('Val Max', 'Color Mask')

        # 'Orange' objects mask based on HSV values
        m_low_1 = np.array([5, 120, 50]).astype(np.uint8)
        m_high_1 = np.array([11, 255, 250]).astype(np.uint8)
        cv_mask_1 = cv2.inRange(cv_img_hsv, m_low_1, m_high_1)

        # 'Blue' objects mask based on HSV values [H, S, V]
        m_low_2 = np.array([80, 110, 50]).astype(np.uint8)  # [80, 100, 0]
        m_high_2 = np.array([130, 225, 100]).astype(np.uint8)  # [130, 179, 100]
        cv_mask_2 = cv2.inRange(cv_img_hsv, m_low_2, m_high_2)

        # Join color masks
        v_mask_1 = cv2.inRange(cv_img_hsv, m_low_1, m_high_1)
        v_mask_2 = cv2.inRange(cv_img_hsv, m_low_2, m_high_2)
        v_mask = cv2.bitwise_or(v_mask_1, v_mask_2)
        cv2.imshow('Color Mask', cv2.resize(v_mask, (int(v_mask.shape[1] * .25), int(v_mask.shape[0] * .25))))

        # Get contours from masks
        cv_img_mask = cv2.bitwise_or(cv_mask_1, cv_mask_2)
        cv_img_mskd = cv2.bitwise_and(cv_img, cv_img, mask=cv_img_mask)
        cv_img_gray = cv2.cvtColor(cv_img_mskd, cv2.COLOR_BGR2GRAY)
        cv_img_blur = cv2.blur(cv_img_gray, (2, 2))
        cv_img_edge = cv2.Canny(cv_img_blur, 40, 60)
        krn = np.ones((2, 2), np.uint8)
        cv_img_dlt = cv2.dilate(cv_img_edge, krn, iterations=1)
        cv_img = get_contours(cv_img, cv_img_dlt)

        cv_img = cv2.resize(cv_img, (2 * cv_img.shape[1] / 3, 2 * cv_img.shape[0] / 3))
        cv2.imshow('Image', cv_img)

        cv2.waitKey(1)


def image_callback(ros_img):
    global data_queue
    data_queue.enqueue(ros_img)


def main():
    global data_queue
    data_queue = ring_buffer.RingBuffer(50)

    rospy.init_node("camera_subscriber", anonymous=True)
    cam_name = "head_camera"
    rospy.Subscriber("/cameras/%s/image" % cam_name, Image, image_callback)

    while not rospy.is_shutdown():
        img_processor()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
