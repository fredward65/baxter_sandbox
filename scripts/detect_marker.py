#!/usr/bin/env python

import PIL
import cv2
import cv_bridge
import numpy as np
import rospy
import std_srvs.srv
# from aruco_tools.utils import load_coefficients
from baxter_interface.camera import CameraController
# from baxter_interface.digital_io import DigitalIO
from custom_tools import ring_buffer
from cv2 import aruco
from scipy.spatial.transform import Rotation as Rot
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker

bridge = cv_bridge.CvBridge()

global v_mrk, pub, data_queue, mtx_head, mtx_head_b

dist = np.float32([[0.0, 0.0, 0.0, 0.0, 0.0]]).reshape(-1, 1)
mtx = np.float32([[410.0, 0.0, 640.0], [0.0, 410.0, 400.0], [0.0, 0.0, 1.0]]).reshape(-1, 3)
coef = np.float32([[-410.0, -0.0, 639.0, 0.0, 0.0, -410.0, 399.0, 0.0, 0.0, 0.0, 1.0, 0.0]]).reshape(-1, 4)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)
params = aruco.DetectorParameters_create()
[size_of_marker, length] = [0.125, 0.05]
target_id = 0


"""
def update_state(*args):
    DIR = './src/baxter_sandbox/scripts/aruco_tools/'
    if args[0]:
        print "Button pressed. Taking picture..."
        cv2.imwrite(DIR+'img'+str(rospy.get_time())+'.jpg', cv_save)
        print "Picture taken! "+str(rospy.get_time())
        msg = cv_bridge.CvBridge().cv2_to_imgmsg(cv2.resize(cv_save, (1024, 600)), encoding="passthrough")
        pub.publish(msg)
"""


def reset_cameras():
    print('Resetting cameras...')
    reset_srv = rospy.ServiceProxy('cameras/reset', std_srvs.srv.Empty)
    rospy.wait_for_service('cameras/reset', timeout=10)
    reset_srv()
    print('Done resetting')


def open_camera(camera='head_camera', res=(1280, 800)):
    print('Starting %s' % camera)
    cam = CameraController(camera)
    cam.resolution = res
    cam.open()
    print('%s started' % camera)


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


def image_callback(ros_img):
    global data_queue
    data_queue.enqueue(ros_img)


def marker_publish():
    global v_mrk, pub, data_queue, mtx_head, mtx_head_b
    # mtx, dist = load_coefficients(DIR+'calibration_charuco.yml')
    ros_img = data_queue.dequeue()
    if ros_img:
        cv_raw = bridge.imgmsg_to_cv2(ros_img, desired_encoding="passthrough")
        cv_img = cv2.undistort(cv_raw, mtx, dist, coef, mtx)
        cv_blr = cv2.blur(cv_img, (2, 2))
        cv_gray = cv2.cvtColor(cv_blr, cv2.COLOR_BGR2GRAY)
        cv_mrk = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2BGR)
        corners, ids, rip = aruco.detectMarkers(cv_gray, aruco_dict, parameters=params)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
        for i, corner in enumerate(corners):
            cr2 = cv2.cornerSubPix(cv_gray, corner, winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria)
            [r_vec, t_vec, _] = aruco.estimatePoseSingleMarkers(cr2, size_of_marker, mtx, dist)
            if type(t_vec) != 'NoneType':
                cv_mrk = aruco.drawAxis(cv_mrk, mtx, dist, r_vec[0], t_vec[0], length)
                cv_mrk = aruco.drawDetectedMarkers(cv_mrk, [corners[i]], ids[i])
                dst, jac = cv2.Rodrigues(r_vec[0])
                mtx_rot = np.float32([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).reshape(4, 4)
                t_mrk = t_vec[0].ravel().reshape(3, 1)
                [v_mrk.color.r, v_mrk.color.g, v_mrk.color.b, v_mrk.color.a] = [t_mrk[0], t_mrk[1], t_mrk[2], 1.0]
                r_mrk = dst.reshape(3, 3)
                mtx_mrk = np.concatenate((np.c_[r_mrk, t_mrk], [[0, 0, 0, 1]]), axis=0)
                mtx_true = np.dot(mtx_head_b, mtx_head)
                mtx_true = np.dot(mtx_true, mtx_mrk)
                mtx_true = np.dot(mtx_rot, mtx_true)
                t_true = mtx_true[0:3, 3]
                r_true = mtx_true[0:3, 0:3]
                r_true = Rot.from_dcm(r_true).as_quat()
                cv2.putText(cv_mrk, "%5.2f, %5.2f, %5.2f" % (t_true[0], t_true[1], t_true[2] + .82),
                            (corner[-1,-1,0], corner[-1,-1,1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 64), 1)
                v_mrk.id = int(ids[i])  # target_id
                v_mrk.pose.position.x = t_true[0]
                v_mrk.pose.position.y = t_true[1]
                v_mrk.pose.position.z = t_true[2]
                v_r = v_mrk.pose.orientation
                [v_r.x, v_r.y, v_r.z, v_r.w] = r_true
                v_mrk.pose.orientation = v_r
                v_mrk.header.stamp = rospy.Time.now()
                pub.publish(v_mrk)
        cv2.imshow('Marker', cv2.resize(cv_mrk,(1024,600)))
        cv2.waitKey(1)


def main():
    global v_mrk, pub, data_queue
    data_queue = ring_buffer.RingBuffer(50)
    v_mrk = Marker()
    v_mrk.type = 1
    v_mrk.header.frame_id = "/base"
    [v_mrk.scale.x, v_mrk.scale.y, v_mrk.scale.z] = [0.15, 0.15, 0.01]
    [v_mrk.color.r, v_mrk.color.g, v_mrk.color.b, v_mrk.color.a] = [0.0, 0.0, 1.0, 1.0]

    rospy.init_node("aruco_camera_node", anonymous=True)
    cam_name = "head_camera"
    reset_cameras()
    open_camera(cam_name)  # , res=(960, 600)
    rospy.Subscriber("/cameras/%s/image" % cam_name, Image, image_callback)
    rospy.Subscriber("/tf", TFMessage, head_callback)
    pub = rospy.Publisher("/detected_marker", Marker, queue_size=1)
    # button = DigitalIO("torso_left_button_ok")
    # button.state_changed.connect(update_state)
    while not rospy.is_shutdown():
        marker_publish()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
