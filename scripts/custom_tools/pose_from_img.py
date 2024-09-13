#!/usr/bin/env python

import cv2
import cv_bridge
import numpy as np
import rospy
import std_srvs.srv
from baxter_interface.camera import CameraController
from cv2 import aruco
from scipy.spatial.transform import Rotation as Rot
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker


""" Object detection parent class """
class DetectObject(object):
    """
    Constructor
    gets
    dist: distortion coefficients vector (4, 5, 8, 12 or 14)
    mtx: input camera matrix (3x3)
    """
    def __init__(self, dist, mtx):
        # Camera distortion parameters
        self.dist, self.mtx = dist, mtx
        self.offset = [.00, .00, .00] # Pedestal z offset
        self.mtx_head, self.mtx_head_b = np.eye(4), np.eye(4)
        self.hsv_low, self.hsv_high = [0, 0, 0], [179, 255, 255]

    """
    Transformation Reference Frame callback to subscribe to "/tf"
    gets
    ref_msg: TFMessage
    """
    def tf_callback(self, ref_msg):
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
                self.mtx_head_b = np.concatenate((np.c_[r_head.T, -t_head], [[0, 0, 0, 1]]), axis=0)
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
                self.mtx_head = np.concatenate((np.c_[r_head, t_head], [[0, 0, 0, 1]]), axis=0)

    """
    Reference Frame transform from /head_camera to /base
    gets
    t_vec: Cartesian translation vector (x, y, z)
    r_vec: Rotation vector, Euler angles
    returns
    t_dict: Cartesian translation dict {'x', 'y', 'z'[, 'id']}
    r_dict: Quaternion rotation dict {'x', 'y', 'z', 'w', [, 'id']}
    """
    def get_transform(self, t_vec, r_vec):
        mtx_rot = np.float32([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).reshape(4, 4)
        dst, _ = cv2.Rodrigues(r_vec)
        r_mrk = dst.reshape(3, 3)
        t_mrk = t_vec.ravel().reshape(3, 1)
        mtx_mrk = np.concatenate((np.c_[r_mrk, t_mrk], [[0, 0, 0, 1]]), axis=0)
        mtx_true = np.dot(self.mtx_head_b, self.mtx_head)
        mtx_true = np.dot(mtx_true, mtx_mrk)
        mtx_true = np.dot(mtx_rot, mtx_true)
        t_true = mtx_true[0:3, 3]
        t_true += self.offset
        r_true = mtx_true[0:3, 0:3]
        r_true = Rot.from_dcm(r_true).as_quat()
        t_dict = {'x': t_true[0], 'y': t_true[1], 'z': t_true[2]}
        r_dict = {'x': r_true[0], 'y': r_true[1], 'z': r_true[2], 'w': r_true[3]}
        return t_dict, r_dict

    """
    Pose From Image, image processing function
    gets
    cv_raw: OpenCV raw image (BGR or BGRA)
    returns
    t_list: Quaternion rotation dict list {'x', 'y', 'z', 'w', [, 'id']}
    t_list: Quaternion rotation dict list {'x', 'y', 'z', 'w', [, 'id']}
    """
    def pose_from_img(self, cv_raw):
        # Image pre-processing
        cv_img = cv2.undistort(cv_raw, self.mtx, self.dist, None, self.mtx)
        cv_img = cv2.blur(cv_img, (4, 4))
        cv_img_hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

        # Object masking based on HSV values
        cv_img_mask = cv2.inRange(cv_img_hsv, self.hsv_low, self.hsv_high)

        # Generate object mask
        cv_img_mskd = cv2.bitwise_and(cv_img, cv_img, mask=cv_img_mask)
        cv_img_gray = cv2.cvtColor(cv_img_mskd, cv2.COLOR_BGR2GRAY)
        cv_img_blur = cv2.blur(cv_img_gray, (2, 2))
        cv_img_edge = cv2.Canny(cv_img_blur, 40, 60)
        krn = np.ones((2, 2), np.uint8)
        cv_img_dlt = cv2.dilate(cv_img_edge, krn, iterations=1)

        # Get contours from mask
        t_list, r_list = self.pose_from_contours(cv_img_dlt)
        return t_list, r_list

    """
    Pose From Contours, contours processing function
    gets 
    cv_img_dlt: OpenCV dilated image from pose_from_img
    returns
    [], []
    """
    def pose_from_contours(self, cv_img_dlt):
        return [], []


""" Ball detector class, inherits DetectObject """
class DetectBall(DetectObject):
    """
    Constructor
    gets
    hsv_low: lower HSV threshold to mask colours [0-180, 0-255, 0-255]
    hsv_high: upper HSV threshold to mask colours [0-180, 0-255, 0-255]
    radius: ball to detect radius, in meters
    dist: distortion coefficients vector (4, 5, 8, 12 or 14)
    mtx: input camera matrix (3x3)
    """
    def __init__(self, hsv_low, hsv_high, radius, dist, mtx):
        super(DetectBall, self).__init__(dist, mtx)
        self.hsv_low = np.array(hsv_low).astype(np.uint8)
        self.hsv_high = np.array(hsv_high).astype(np.uint8)
        self.obj_r = radius
        self.obj_s = np.array([0, 0, 0, self.obj_r, 0, 0,
                               0, self.obj_r, 0, self.obj_r, self.obj_r, 0], np.float32).reshape(-1, 3)
        self.offset = [self.obj_r, self.obj_r, .0]

    """
    Pose From Contours, contours processing function
    gets
    cv_img_dlt: OpenCV dilated image from pose_from_img
    returns
    t_list: Cartesian translation dict list {'x', 'y', 'z'[, 'id']}
    r_list: Quaternion rotation dict list {'x', 'y', 'z', 'w', [, 'id']}
    """
    def pose_from_contours(self, cv_img_dlt):
        _, cont, _ = cv2.findContours(cv_img_dlt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        t_list, r_list = [], []
        for cnt in cont:
            area = cv2.contourArea(cnt)
            # Limit area threshold
            if 1250 < area:
                cv_img_ct = np.zeros(cv_img_dlt.shape, np.uint8)
                cv_img_ct = cv2.blur(cv2.drawContours(cv_img_ct, [cnt], -1, 255, -1), (2, 2))
                circles = cv2.HoughCircles(cv_img_ct, cv2.HOUGH_GRADIENT, 1.4, 100, param1=50, param2=30)
                r_flag = True if circles is not None else False
                per = cv2.arcLength(cnt, True)
                apr = cv2.approxPolyDP(cnt, 0.02 * per, True)
                x, y, w, h = cv2.boundingRect(apr)
                # Filter polygons
                if r_flag:
                    pos = np.array([x + w / 2, y + h / 2, x + w, y + h / 2,
                                    x + w / 2, y + h    , x + w, y + h], np.float32).reshape(4, 1, -1)
                    _, r_vec, t_vec = cv2.solvePnP(self.obj_s, pos, self.mtx, self.dist)
                    t_vec[2] += self.obj_r
                    t_res, r_res = self.get_transform(t_vec, r_vec)
                    t_list.append(t_res)
                    r_list.append(r_res)
        return t_list, r_list


""" Square detector class, inherits DetectObject """
class DetectSquare(DetectObject):
    """
    Constructor
    gets
    hsv_low: lower HSV threshold to mask colours [0-180, 0-255, 0-255]
    hsv_high: upper HSV threshold to mask colours [0-180, 0-255, 0-255]
    len: ball to detect radius, in meters
    dist: distortion coefficients vector (4, 5, 8, 12 or 14)
    mtx: input camera matrix (3x3)
    """
    def __init__(self, hsv_low, hsv_high, len, dist, mtx):
        super(DetectSquare, self).__init__(dist, mtx)
        self.hsv_low = np.array(hsv_low).astype(np.uint8)
        self.hsv_high = np.array(hsv_high).astype(np.uint8)
        self.obj_l = len
        self.obj_s = np.array([0, 0, 0, 0, self.obj_l, 0,
                               self.obj_l, self.obj_l, 0, self.obj_l, 0, 0], np.float32).reshape(-1, 3)

    """
    Pose From Contours, contours processing function
    gets
    cv_img_dlt: OpenCV dilated image from pose_from_img
    returns
    t_list: Cartesian translation dict list {'x', 'y', 'z'[, 'id']}
    r_list: Quaternion rotation dict list {'x', 'y', 'z', 'w', [, 'id']}
    """
    def pose_from_contours(self, cv_img_dlt):
        _, cont, _ = cv2.findContours(cv_img_dlt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        t_list, r_list = [], []
        for cnt in cont:
            area = cv2.contourArea(cnt)
            # Limit area threshold
            if 100 < area:
                cv_img_ct = np.zeros(cv_img_dlt.shape, np.uint8)
                cv_img_ct = cv2.drawContours(cv_img_ct, [cnt], -1, 255, -1)
                per = cv2.arcLength(cnt, True)
                apr = cv2.approxPolyDP(cnt, 0.1 * per, True)
                x, y, w, h = cv2.boundingRect(apr)
                # Filter polygons
                if len(apr) == 4:
                    """
                    cv2.putText(cv_img_ct, "A", tuple(apr[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (127, 0, 0), 1)
                    cv2.putText(cv_img_ct, "D", tuple(apr[-1][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (127, 0, 0), 1)
                    cv2.imshow("Image", cv_img_ct)
                    cv2.waitKey(1)
                    """
                    pos = np.array(apr, np.float32).reshape(4, 1, -1)
                    obj_s = self.obj_s if pos[0][0][1] < pos[-1][0][1] else np.roll(self.obj_s, -1, axis=0)
                    _, r_vec, t_vec = cv2.solvePnP(obj_s, pos, self.mtx, self.dist)
                    t_res, r_res = self.get_transform(t_vec, r_vec)
                    t_list.append(t_res)
                    r_list.append(r_res)
        return t_list, r_list


""" ArUco Marker detector class, inherits DetectObject """
class DetectMarker(DetectObject):
    """
    Constructor
    gets
    size_of_marker: square marker length, in mm
    dist: distortion coefficients vector (4, 5, 8, 12 or 14)
    mtx: input camera matrix (3x3)
    """
    def __init__(self, size_of_marker, dist, mtx):
        super(DetectMarker, self).__init__(dist, mtx)
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)
        self.params = aruco.DetectorParameters_create()
        [self.size_of_marker, self.length] = [size_of_marker, 0.05]

    """
    Pose From Image, image processing function, overrides parent
    gets
    cv_raw: OpenCV raw image (BGR or BGRA)
    returns
    t_list: Cartesian translation dict list {'x', 'y', 'z'[, 'id']}
    r_list: Quaternion rotation dict list {'x', 'y', 'z', 'w', [, 'id']}
    """
    def pose_from_img(self, cv_raw):
        # Image pre-processing
        cv_img = cv2.undistort(cv_raw, self.mtx, self.dist, None, self.mtx)
        cv_blr = cv2.blur(cv_img, (2, 2))
        cv_gray = cv2.cvtColor(cv_blr, cv2.COLOR_BGR2GRAY)
        cv_mrk = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2BGR)

        # ArUco markers detect
        corners, ids, _ = aruco.detectMarkers(cv_gray, self.aruco_dict, parameters=self.params)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
        t_list, r_list = [], []
        for i, corner in enumerate(corners):
            cr2 = cv2.cornerSubPix(cv_gray, corner, winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria)
            r_vec, t_vec, _ = aruco.estimatePoseSingleMarkers(cr2, self.size_of_marker, self.mtx, self.dist)
            if type(t_vec) != 'NoneType':
                t_res, r_res = self.get_transform(t_vec, r_vec)
                t_res['id'], r_res['id'] = ids[i, -1], ids[i, -1]
                t_list.append(t_res)
                r_list.append(r_res)
        return t_list, r_list


"""
Marker publisher helper function
gets
pub: Marker msg ROS publisher
t_list: Cartesian translation dict list {'x', 'y', 'z'[, 'id']}
r_list: Quaternion rotation dict list {'x', 'y', 'z', 'w', [, 'id']}
v_mrk: Marker msg object
"""
def publish_marker(pub, t_list, r_list, v_mrk):
    for i, (t_res, r_res) in enumerate(zip(t_list, r_list)):
        v_mrk.id = t_res['id'] if 'id' in t_res else 100  # Arbitrary id = (i + 100) for identified balls
        v_mrk_p, v_mrk_o = v_mrk.pose.position, v_mrk.pose.orientation
        v_mrk_p.x, v_mrk_p.y, v_mrk_p.z = t_res['x'], t_res['y'], t_res['z']
        v_mrk_o.x, v_mrk_o.y, v_mrk_o.z, v_mrk_o.w = r_res['x'], r_res['y'], r_res['z'], r_res['w']
        v_mrk.pose.position, v_mrk.pose.orientation = v_mrk_p, v_mrk_o
        v_mrk.header.stamp = rospy.Time.now()
        if not rospy.is_shutdown():
            pub.publish(v_mrk)


""" Camera reset function """
def reset_cameras():
    print('Resetting cameras...')
    reset_srv = rospy.ServiceProxy('cameras/reset', std_srvs.srv.Empty)
    rospy.wait_for_service('cameras/reset', timeout=10)
    reset_srv()
    print('Done resetting')


"""
Camera open function
gets
camera: camera name, String, default='head_camera'
res: camera resolution, tuple, default=(1280, 800)
"""
def open_camera(camera='head_camera', res=(1280, 800)):
    print('Starting %s' % camera)
    cam = CameraController(camera)
    cam.resolution = res
    cam.open()
    print('%s started' % camera)


def main():
    # Camera parameters
    dist = np.float32([[0.0, 0.0, 0.0, 0.0, 0.0]]).reshape(-1, 1)
    mtx = np.float32([[410.0, 0.0, 640.0], [0.0, 410.0, 400.0], [0.0, 0.0, 1.0]]).reshape(-1, 3)
    bridge = cv_bridge.CvBridge()

    # ball_detector parameters
    radius, hsv_low, hsv_high = .10, [40, 40, 10], [65, 255, 255]  # .11, [6, 100, 30], [15, 255, 255]
    ball_detector = DetectBall(hsv_low, hsv_high, radius, dist, mtx)
    v_ball = Marker()
    v_ball.type = 2
    v_ball.header.frame_id = "/base"
    v_ball.scale.x, v_ball.scale.y, v_ball.scale.z = np.multiply(ball_detector.obj_r * 2, np.array([1, 1, 1]))
    v_ball.color.r, v_ball.color.g, v_ball.color.b, v_ball.color.blank = [.0, 1., .0, 1.]  # [1., .5, .0, 1.]

    # square_detector parameters
    len, hsv_low, hsv_high = .056, [40, 40, 20], [65, 255, 255]
    square_detector = DetectSquare(hsv_low, hsv_high, len, dist, mtx)
    v_square = Marker()
    v_square.type = 1
    v_square.header.frame_id = "/base"
    v_square.scale.x, v_square.scale.y, v_square.scale.z = [len, len, .001]
    v_square.color.r, v_square.color.g, v_square.color.b, v_square.color.blank = [.0, 1., .0, 1.]

    # marker_detector parameters
    size_of_marker = .125
    marker_detector = DetectMarker(size_of_marker, dist, mtx)
    v_mrk = Marker()
    v_mrk.type = 1
    v_mrk.header.frame_id = "/base"
    v_mrk.scale.x, v_mrk.scale.y, v_mrk.scale.z = [.15, .15, .01]
    v_mrk.color.r, v_mrk.color.g, v_mrk.color.b, v_mrk.color.blank = [1., 1., 1., 1.]

    # ROS node, subscribers and publishers
    rospy.init_node("object_camera_subscriber", anonymous=True)
    rospy.Subscriber("/tf", TFMessage, ball_detector.tf_callback)  # Subscriber for head and head_camera tf
    rospy.Subscriber("/tf", TFMessage, square_detector.tf_callback)  # Subscriber for head and head_camera tf
    rospy.Subscriber("/tf", TFMessage, marker_detector.tf_callback)  # Subscriber for head and head_camera tf
    pub = rospy.Publisher("/detected_marker", Marker, queue_size=1)  # Publisher for Marker msg (RViz)

    """
    ROS Image subscriber callback
    gets
    ros_img: ROS Image
    """
    def image_callback(ros_img):
        cv_raw = bridge.imgmsg_to_cv2(ros_img, desired_encoding="passthrough")

        # Pose from ball_detector
        t_list_b, r_list_b = ball_detector.pose_from_img(cv_raw)
        publish_marker(pub, t_list_b, r_list_b, v_ball)

        # Pose from square_detector
        # t_list_s, r_list_s = square_detector.pose_from_img(cv_raw)
        # publish_marker(pub, t_list_s, r_list_s, v_square)

        # Pose from marker_detector
        t_list_m, r_list_m = marker_detector.pose_from_img(cv_raw)
        publish_marker(pub, t_list_m, r_list_m, v_mrk)

    # ROS Image subscriber
    cam_name = "head_camera"
    rospy.Subscriber("/cameras/%s/image" % cam_name, Image, image_callback)

    rospy.spin()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
