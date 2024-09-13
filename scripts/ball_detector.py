#!/usr/bin/env python

import cv2
import cv_bridge
import numpy as np
import rospy
from custom_tools import ring_buffer
from scipy.spatial.transform import Rotation as Rot
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker

bridge = cv_bridge.CvBridge()

global data_queue, pub, mtx_head, mtx_head_b
ref_l = None

# Camera distortion parameters
dist = np.float32([[0.0, 0.0, 0.0, 0.0, 0.0]]).reshape(-1, 1)
mtx = np.float32([[410.0, 0.0, 640.0], [0.0, 410.0, 400.0], [0.0, 0.0, 1.0]]).reshape(-1, 3)
coef = np.float32([[-410.0, -0.0, 639.0, 0.0, 0.0, -410.0, 399.0, 0.0, 0.0, 0.0, 1.0, 0.0]]).reshape(-1, 4)
coef = np.float32([[-410.0, -0.0, 639.0, 0.0, -410.0, 399.0, 0.0, 0.0, 1.0]]).reshape(-1, 3)

[obj_w, obj_h, obj_r] = [.210, .297, .11]
objp_1 = np.array([0, 0, 0, 0, obj_w, 0, obj_h, obj_w, 0, obj_h, 0, 0], np.float32).reshape(-1, 3)
objp_2 = np.array([0, 0, 0, 0, obj_h, 0, obj_w, obj_h, 0, obj_w, 0, 0], np.float32).reshape(-1, 3)
objs = np.array([0, 0, 0, obj_r, 0, 0, 0, obj_r, 0, obj_r, obj_r, 0], np.float32).reshape(-1, 3)
axis = np.float32([[.1, 0, 0], [0, .1, 0], [0, 0, -.1]]).reshape(-1, 3)


def tb_callback(val):
    pass


def draw(cv_img, corners, img_pts):
    corner = tuple(corners[0].ravel())
    cv_img = cv2.line(cv_img, corner, tuple(img_pts[0].ravel()), (0, 0, 255), 5)
    cv_img = cv2.line(cv_img, corner, tuple(img_pts[1].ravel()), (0, 255, 0), 5)
    cv_img = cv2.line(cv_img, corner, tuple(img_pts[2].ravel()), (255, 0, 0), 5)
    return cv_img


def head_callback(ref_msg):
    global mtx_head, mtx_head_b
    for tf in ref_msg.transforms:
        if tf.child_frame_id == "head":
            head_tf = tf.transform
            t_head = np.float32([head_tf.translation.x,
                                 head_tf.translation.y,
                                 head_tf.translation.z]).ravel().reshape(3, 1)
            r_head = Rot.from_quat([head_tf.rotation.x,
                                    head_tf.rotation.y,
                                    head_tf.rotation.z,
                                    head_tf.rotation.w]).as_dcm()
            mtx_head_b = np.concatenate((np.c_[r_head.T, -t_head], [[0, 0, 0, 1]]), axis=0)
        if tf.child_frame_id == "head_camera":
            head_camera_tf = tf.transform
            t_head = np.float32([head_camera_tf.translation.x,
                                 head_camera_tf.translation.y,
                                 head_camera_tf.translation.z * -1]).ravel().reshape(3, 1)
            r_head = Rot.from_quat([head_camera_tf.rotation.x,
                                    head_camera_tf.rotation.y,
                                    head_camera_tf.rotation.z,
                                    head_camera_tf.rotation.w]).as_dcm()
            r_head = np.dot(Rot.from_rotvec(np.deg2rad(-38) * np.array([0, 1, 0])).as_dcm(), r_head)
            mtx_head = np.concatenate((np.c_[r_head, t_head], [[0, 0, 0, 1]]), axis=0)


def get_transform(r_vec, t_vec):
    mtx_rot = np.float32([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).reshape(4, 4)
    dst, j = cv2.Rodrigues(r_vec)
    r_mrk = dst.reshape(3, 3)
    t_mrk = t_vec.ravel().reshape(3, 1)
    mtx_mrk = np.concatenate((np.c_[r_mrk, t_mrk], [[0, 0, 0, 1]]), axis=0)
    mtx_true = np.dot(mtx_head_b, mtx_head)
    mtx_true = np.dot(mtx_true, mtx_mrk)
    mtx_true = np.dot(mtx_rot, mtx_true)
    t_true = mtx_true[0:3, 3]
    r_true = mtx_true[0:3, 0:3]
    r_true = Rot.from_dcm(r_true).as_quat()
    return t_true, r_true


def get_bounding_box(x, y, w, h):
    return np.array([x + w / 2, y + h / 2,
                     x + w,     y + h / 2,
                     x + w / 2, y + h,
                     x + w    , y + h],
                    np.float32).reshape(4, 1, -1)


def get_contours(cv_img, img):
    global pub
    im, cont, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    i = 0
    for cnt in cont:
        r_flag = False
        area = cv2.contourArea(cnt)
        if 500 < area < 50000:
            cv2.drawContours(cv_img, cnt, -1, (0, 255, 0), 1)
            cv_img_ct = np.zeros(img.shape, np.uint8)
            cv_img_ct = cv2.blur(cv2.drawContours(cv_img_ct, [cnt], -1, 255, -1), (2, 2))
            circles = cv2.HoughCircles(cv_img_ct, cv2.HOUGH_GRADIENT, 1.4, 100, param1=50, param2=30)
            if circles is not None:
                r_flag = True
            per = cv2.arcLength(cnt, True)
            apr = cv2.approxPolyDP(cnt, 0.05 * per, True)
            x, y, w, h = cv2.boundingRect(apr)
            sides = len(apr)
            v_mrk = Marker()
            v_mrk.type = 2
            v_mrk.id = i
            v_mrk.header.frame_id = "/base"
            [v_mrk.scale.x, v_mrk.scale.y, v_mrk.scale.z] = [0.22, 0.22, 0.22]
            # Filter polygons
            if r_flag:
                color = (0, 127, 255)
                cv2.circle(cv_img, (x + w / 2, y + h / 2), w / 2 if w > h else h / 2, color, 2)
                pos = get_bounding_box(x, y, w, h)
                cv2.rectangle(cv_img, (x + w / 2, y + h / 2), (x + w, y + h), (0, 255, 255), 1)
                ret, r_vec, t_vec = cv2.solvePnP(objs, pos, mtx, dist)
                img_pts, jac = cv2.projectPoints(axis, r_vec, t_vec, mtx, dist)
                cv_img = draw(cv_img, pos, img_pts)
                [t_res, r_res] = get_transform(r_vec, t_vec)
                t_res += [obj_r, obj_r, .00]
                cv2.putText(cv_img, "pos: %5.2f, %5.2f, %5.2f" % (t_res[0], t_res[1], t_res[2]), (x, y + h + 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                [v_mrk.color.r, v_mrk.color.g, v_mrk.color.b, v_mrk.color.a] = [t_vec[0], t_vec[1], t_vec[2], 1.0]
                [v_mrk_p, v_mrk_o] = [v_mrk.pose.position, v_mrk.pose.orientation]
                [v_mrk_p.x, v_mrk_p.y, v_mrk_p.z] = t_res
                [v_mrk_o.x, v_mrk_o.y, v_mrk_o.z, v_mrk_o.w] = r_res
                [v_mrk.pose.position, v_mrk.pose.orientation] = [v_mrk_p, v_mrk_o]
                v_mrk.header.stamp = rospy.Time.now()
                pub.publish(v_mrk)
            else:
                color = (0, 255, 255)
                cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 255, 255), 1)
            cv2.putText(cv_img, "ID: %i" % i, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(cv_img, "Vrtx: %i" % sides, (x, y + h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(cv_img, "Area: %i" % area, (x, y + h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return cv_img


def image_callback(ros_img):
    global data_queue
    data_queue.enqueue(ros_img)


def img_processor(m_low=[5, 80, 20], m_high=[15, 255, 255]):
    global data_queue
    ros_img = data_queue.dequeue()
    if ros_img:
        cv_raw = bridge.imgmsg_to_cv2(ros_img, desired_encoding="passthrough")
        cv_img = cv2.undistort(cv_raw, mtx, dist, None, mtx)
        cv2.imshow('Image Undist', cv_img)
        cv_img = cv2.blur(cv_img, (2, 2))
        cv_img_hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)  # hue, sat, val = cv2.split(cv_img_hsv)

        # Coloured objects mask based on HSV values
        m_low_1 = np.array(m_low).astype(np.uint8)
        m_high_1 = np.array(m_high).astype(np.uint8)
        cv_img_mask_1 = cv2.inRange(cv_img_hsv, m_low_1, m_high_1)

        # Get contours from masks
        cv_img_mskd = cv2.bitwise_and(cv_img, cv_img, mask=cv_img_mask_1)
        cv_img_gray = cv2.cvtColor(cv_img_mskd, cv2.COLOR_BGR2GRAY)
        cv_img_blur = cv2.blur(cv_img_gray, (2, 2))
        cv_img_edge = cv2.Canny(cv_img_blur, 40, 60)
        krn = np.ones((2, 2), np.uint8)
        cv_img_dlt = cv2.dilate(cv_img_edge, krn, iterations=1)
        cv_img = get_contours(cv_raw, cv_img_dlt)

        cv_img = cv2.resize(cv_img, (2 * cv_img.shape[1] / 3, 2 * cv_img.shape[0] / 3))
        cv2.imshow('Image', cv_img)

        cv2.waitKey(1)


def main():
    global data_queue, pub
    data_queue = ring_buffer.RingBuffer(50)

    rospy.init_node("ball_camera_subscriber", anonymous=True)
    rospy.Subscriber("/tf", TFMessage, head_callback)
    rospy.sleep(1)

    cam_name = "head_camera"
    rospy.Subscriber("/cameras/%s/image" % cam_name, Image, image_callback)
    pub = rospy.Publisher("/detected_marker", Marker, queue_size=1)

    while not rospy.is_shutdown():
        obj_r = .10
        img_processor([50, 40, 10], [65, 255, 255])

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
