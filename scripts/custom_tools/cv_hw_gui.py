#!/usr/bin/env python

import cv2
import numpy as np
import matplotlib.pyplot as plt
import rospy
import time
from baxter_core_msgs.msg import AssemblyState
from baxter_interface import RobotEnable, CHECK_VERSION
from baxter_interface.limb import Limb
from ik_tools import IK_Limb
from img2pose import DST_VEC, CAM_MTX, DetectMarker, reset_cameras, open_camera
from pose_dmp_test import PoseDMP
from sensor_msgs.msg import Image
from tf.transformations import quaternion_multiply as quat_mult
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


def rotate_v_q(v, q):
    v = quat_mult([q[0], q[1], q[2], q[3]], [v[0], v[1], v[2], 0])
    v = quat_mult(v, [-q[0], -q[1], -q[2], q[3]])
    return v[0:3]


def robot_state(flag):
    rs = RobotEnable(CHECK_VERSION)
    while rospy.wait_for_message('/robot/state', AssemblyState).enabled != flag:
        if flag:
            rs.enable()
        else:
            rs.disable()


class DataRecorder(object):
    """
    DMP Handwriting Helper class.
    Main purposes:
    - Record handwriting data
    - Train and play handwriting DMP model
    - Play handwriting DMP model using Baxter robot (TO DO)
    - Save handwriting DMP data as CSV (TO DO)
    """

    def __init__(self, wnd, width=800, height=600):
        """
        DMP Handwriting Data Recorder

        Parameters
        ----------
        wnd : str
            Name of the window
        width : int
            Width of the window
        height : int
            Height of the window
        """
        """ Window variables """
        self.w = width
        self.h = height
        self.wnd = wnd
        self.blank = 255 * np.ones((self.h, self.w, 3), np.uint8)
        self.bg = self.blank
        self.alpha = self.blank
        self.img = np.copy(self.blank)
        self.img_old = np.copy(self.img)
        """ Flags """
        self.str_flag = ''
        self.shutdown = False
        """ Recording data variables """
        self.t = np.empty(0)
        self.t_v = np.empty(0)
        self.t_r = np.empty(0)
        self.trp = np.empty(0)
        self.trp_v = np.empty(0)
        self.trp_r = np.empty(0)
        self.trp_t = np.empty(0)
        self.p = np.empty(2)
        self.tref = 0.0
        """ DMP-related variables """
        self.tau = 1.0
        self.y = np.empty(0)
        self.dy = np.empty(0)
        self.y0 = np.empty(0)
        self.yg = np.empty(0)
        self.q = np.empty(0)
        self.p0 = []
        self.yg_off = np.empty(0)
        self.xyz_off = 0.0
        self.rot_off = 0.0
        self.z_op = np.array([.00, .00, .16])
        self.z_off = self.z_op
        self.dmp_obj = PoseDMP(n=200, alpha_y=8, mode='LIENDO')
        """ ROS-related code """
        rospy.init_node("hadwriting_gui", anonymous=True)
        side = 'left'
        self.l = Limb(side)
        self.ik_l = IK_Limb(side, verbose=True)
        self.init_pose = self.l.joint_angles()
        rospy.on_shutdown(self.shutdown_hook)
        # ArUco marker parameters
        size_of_marker = .125
        self.scale = self.h if self.h > self.w else self.w
        self.mrk_detector = DetectMarker(size_of_marker, DST_VEC, CAM_MTX)
        # TF subscriber
        rospy.Subscriber("/tf", TFMessage, self.mrk_detector.tf_callback)
        # ROS Image subscriber
        cam_name = "head_camera"
        rospy.Subscriber("/cameras/%s/image" % cam_name, Image, self.mrk_detector.img_callback)
        # Reset cameras
        reset_cameras()
        # Open cameras
        open_camera()
        """ RViz Marker """
        # Marker msg (RViz)
        self.v_dot = self.create_vmrk(11, [0., 1., 1., 1.])
        self.r_dot = self.create_vmrk(12, [0., 1., 0., 1.])
        # Publisher for Marker msg (RViz)
        self.mrk_pub = rospy.Publisher("/detected_marker", Marker, queue_size=10000)
        """ OpenCV window """
        cv2.namedWindow(self.wnd)
        cv2.createButton('Train DMP', self.training_callback)
        cv2.createButton('Draw with Baxter', self.play_callback)
        cv2.createButton('Save captured data', self.save_in_callback)
        cv2.createButton('Save DMP GUI data', self.save_v_callback)
        cv2.createButton('Save DMP virtual data', self.save_r_callback)
        cv2.createButton('Save DMP Baxter data', self.save_t_callback)
        cv2.createButton('Load captured data', self.load_callback)
        cv2.createTrackbar('Set Mode (see reference)', self.wnd, 4, 4, self.mode_callback)
        cv2.createTrackbar('Set Tau (0.5 + val * 0.1)', self.wnd, 5, 115, self.tau_callback)
        cv2.createTrackbar('Set y_g offset 1', self.wnd, 0, 9, self.yg_callback)
        cv2.createTrackbar('Set y_g offset 2', self.wnd, 0, 9, self.rg_callback)
        cv2.setMouseCallback(self.wnd, self.on_mouse_event)
        cv2.imshow(self.wnd, self.img)

    @staticmethod
    def create_vmrk(id, color):
        scale = .005
        mrk = Marker()
        mrk.type = Marker.SPHERE_LIST
        mrk.header.stamp = rospy.Time.now()
        mrk.header.frame_id = "/base"
        mrk.pose.position.x, mrk.pose.position.y, mrk.pose.position.z = 0., 0., 0.
        mrk.pose.orientation.x, mrk.pose.orientation.y, mrk.pose.orientation.z, mrk.pose.orientation.w = 0., 0., 0., 0.
        mrk.scale.x, mrk.scale.y, mrk.scale.z = scale, scale, scale
        mrk.id = id
        mrk.color.r, mrk.color.g, mrk.color.b, mrk.color.a = color
        return mrk

    def shutdown_hook(self):
        print("Shutting down...")
        if rospy.wait_for_message('/robot/state', AssemblyState).enabled:
            print("Moving to resting pose...")
            self.l.move_to_joint_positions(self.init_pose)
            robot_state(False)
        print("Done")

    def reset_tr(self):
        """ Reset all data variables"""
        self.t = np.empty(0)
        self.t_v = np.empty(0)
        self.t_r = np.empty(0)
        self.trp = np.empty(0)
        self.trp_v = np.empty(0)
        self.trp_r = np.empty(0)
        self.trp_t = np.empty(0)

    def try_move(self, p):
        return self.ik_l.ik_solve(p, self.q)

    def mode_callback(self, val):
        if val == 0:
            mode = 'IJSPEERT_2002'
        elif val == 1:
            mode = 'IJSPEERT_2013'
        elif val == 2:
            mode = 'KOUTRAS_FREE'
        elif val == 3:
            mode = 'KOUTRAS_PLANE'
        elif val == 4:
            mode = 'LIENDO'
        self.dmp_obj.set_mode(mode)
        print(mode)

    def tau_callback(self, val):
        self.tau = .5 + val * .1
        print(self.tau)

    def yg_callback(self, val):
        self.xyz_off = float(val) / self.scale
        self.get_yg_off()
        print(self.xyz_off, self.yg_off)

    def rg_callback(self, val):
        self.rot_off = (np.pi/90) * float(val)
        self.get_yg_off()
        print(self.rot_off, self.yg_off)

    def save_in_callback(self, *args):
        self.save_callback(self.t, self.trp)

    def save_v_callback(self, *args):
        self.save_callback(self.t_v, self.trp_v, txt='v')

    def save_r_callback(self, *args):
        self.save_callback(self.t_r, self.trp_r, txt='r')

    def save_t_callback(self, *args):
        self.save_callback(self.t_r, self.trp_t, txt='t')

    def save_callback(self, t, trp, txt=''):
        if self.str_flag == '' and t.shape[0] > 0:
            print("Saving...")
            stamp = time.asctime().replace(' ', '').replace(':', '')
            mode = self.dmp_obj.mode
            if not txt == '':
                string = '_'.join([txt, stamp, mode, str(self.tau), str(self.xyz_off), str(self.rot_off)])
            else:
                string = stamp
            name = "t_%s.csv" % string
            np.savetxt(name, np.c_[t.reshape((-1, 1)), trp], delimiter=',')
            print("Done saving")
        else:
            print("Cannot save data")

    def load_callback(self, *args):
        data = np.genfromtxt('trp_thanks.csv', delimiter=',')
        self.reset_tr()
        self.t = data[:,0]
        self.trp = data[:,1:]

        self.alpha = np.copy(self.blank)
        self.img = np.copy(self.blank)
        for i in range(self.trp.shape[0]-1):
            p1 = np.multiply(self.trp[i,0:2], self.scale).ravel().astype('int')
            p2 = np.multiply(self.trp[i+1,0:2], self.scale).ravel().astype('int')
            p1[1] = self.h - p1[1]
            p2[1] = self.h - p2[1]
            cv2.line(self.img, tuple(p1), tuple(p2), 0, 2)
            cv2.line(self.alpha, tuple(p1), tuple(p2), 0, 2)
        self.img_old = np.copy(self.img)

    def detect_marker(self):
        mrk_id = 555
        print("Detecting ArUco marker %i..." % mrk_id)
        p_off = np.empty(0)
        q_off = np.empty(0)
        while p_off.shape[0] < 1:
            for i, (t_res, r_res) in enumerate(zip(self.mrk_detector.t_list, self.mrk_detector.r_list)):
                if 'id' in t_res and t_res['id'] == mrk_id:
                    p_off = np.array([t_res['x'], t_res['y'], t_res['z']])
                    q_off = np.array([r_res['x'], r_res['y'], r_res['z'], r_res['w']])
                    print("ArUco marker %i detected" % mrk_id)
                    break
        return p_off, q_off

    def get_yg_off(self):
        a = self.rot_off
        rot = np.array([np.cos(a),-np.sin(a),0,np.sin(a),np.cos(a),0,0,0,1]).reshape((-1,3))
        yg_off = self.trp[-1, :] - self.trp[0, :]
        yg_off = np.dot(rot, yg_off)
        norm = np.linalg.norm(yg_off)
        if not np.linalg.norm(yg_off) == 0.0:
            self.yg_off = self.xyz_off * yg_off / norm
        else:
            self.yg_off = self.trp[-1, :] + np.array([self.xyz_off, 0, 0])

    def training_callback(self, *args):
        if self.t.shape[0] > 1 and self.str_flag == '':
            self.img = np.copy(self.img_old)

            print("Training model...")
            self.dmp_obj.train_model_p(self.t, self.trp)
            print("Done training")

            print("Fitting virtual model...")
            self.t_v = np.empty(0)
            self.trp_v = np.empty(0)
            self.get_yg_off()
            self.y0 = self.trp[0, :]
            self.yg = self.trp[-1, :] + self.yg_off
            self.y = np.copy(self.y0)
            self.dy = 0 * self.y
            self.dmp_obj.set_pg_p(np.array([0, 0, 1]))

            self.dmp_obj.reset_t()
            self.str_flag = 'PLAY_VIRTUAL'
            self.tref = time.time()
        else:
            print("Not possible to play virtual")

    def play_callback(self, *args):
        if self.t.shape[0] > 1 and self.str_flag == '':
            print("Training model...")
            self.dmp_obj.train_model_p(self.t, self.trp)
            print("Done training")

            p_off, q_off = self.detect_marker()
            sm = self.mrk_detector.size_of_marker
            p_off += rotate_v_q([sm, .05, self.z_off[2]], q_off)
            self.z_off = rotate_v_q(self.z_op, q_off)

            print("Fitting real model...")
            self.t_r = np.empty(0)
            self.trp_r = np.empty(0)
            self.trp_t = np.empty(0)
            self.get_yg_off()
            self.y0 = p_off + rotate_v_q(self.trp[0, :], q_off)
            self.yg = p_off + rotate_v_q(self.trp[-1, :] + self.yg_off, q_off)
            self.y = np.copy(self.y0)
            self.dy = 0 * self.y
            self.dmp_obj.set_pg_p(rotate_v_q([0, 0, 1], q_off))

            q_off = quat_mult(q_off, [0, 0, .707, .707])
            self.q = quat_mult(q_off, [0, .707, 0, .707])

            self.p0.append(p_off + np.array([-.20, .00, .00]) + self.z_off)
            self.p0.append(self.y.ravel() + self.z_off)
            self.p0.append(self.y.ravel())

            self.v_dot = self.create_vmrk(11, [0., 1., 1., 1.])
            self.r_dot = self.create_vmrk(12, [0., 1., 0., 1.])

            robot_state(True)
            for p in self.p0:
                j_a = self.try_move(p)
                if j_a:
                    self.l.move_to_joint_positions(j_a)

            self.dmp_obj.reset_t()
            self.str_flag = 'PLAY_BAXTER'
            self.tref = time.time()
        else:
            print("Not possible to draw with Baxter")

    def on_mouse_event(self, event, x, y, flag, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.str_flag == '':
                self.alpha = np.copy(self.blank)
                self.img = np.copy(self.blank)
                self.str_flag = 'RECORDING'
                self.p = np.array([x, self.h - y])
                self.reset_tr()
                self.tref = time.time()
        if event == cv2.EVENT_MOUSEMOVE and flag == cv2.EVENT_FLAG_LBUTTON:
            if self.str_flag == 'RECORDING':
                self.p[1] = self.h - self.p[1]
                cv2.line(self.img, tuple(self.p), (x, y), 0, 2)
                cv2.line(self.alpha, tuple(self.p), (x, y), 0, 2)
                self.p = np.array([x, self.h - y])
        if event == cv2.EVENT_LBUTTONUP:
            if self.str_flag == 'RECORDING' and self.t.shape[0] > 1:
                self.trp = self.trp.reshape((-1, 2)) / self.scale
                self.trp = np.c_[self.trp, np.zeros(len(self.t))]
                self.t = np.linspace(self.t[0], self.t[-1], num=len(self.t)).ravel()
                self.img_old = np.copy(self.img)
            self.str_flag = ''

    def loop(self):
        """ Infinite loop """
        while True:
            # Current time
            ct = time.time() - self.tref

            if self.str_flag == 'RECORDING':
                """ Recording data """
                self.t = np.append(self.t, ct)
                self.trp =  np.append(self.trp, self.p)

            elif 'PLAY' in self.str_flag:
                if ct < self.t[-1] * 1.2 * self.tau:
                    """ Execution time limit """
                    y_p = np.copy(self.y.ravel())
                    self.y, self.dy, _ = self.dmp_obj.fit_step_p(ct, self.y, self.dy, self.y0, self.yg,
                                                                 tau=self.tau)
                    self.y[0, -1] = 0.0 if np.isnan(self.y[0,-1]) else self.y[0, -1]
                    p = np.copy(self.y.ravel())
                    if 'VIRTUAL' in self.str_flag:
                        """ Playing model virtually """
                        self.t_v = np.append(self.t_v, ct)
                        self.trp_v = np.append(self.trp_v, self.y)
                        y_p *= self.scale
                        p *= self.scale
                        y_p[1] = self.h - y_p[1]
                        p[1] = self.h - p[1]
                        p1 = tuple(np.floor(y_p[0:2]).astype("int"))
                        p2 = tuple(np.floor(p[0:2]).astype("int"))
                        cv2.line(self.img, p1, p2, (0, 0, 255), 2)
                        cv2.line(self.alpha, p1, p2, 0, 2)
                    elif 'BAXTER' in self.str_flag:
                        """ Playing model with Baxter """
                        self.t_r = np.append(self.t_r, ct)
                        self.trp_r = np.append(self.trp_r, self.y)
                        self.p0[1] = self.y.ravel() + self.z_off
                        self.p0[2] = self.y.ravel()
                        j_a = self.try_move(p)
                        if j_a:
                            self.l.set_joint_positions(j_a, raw=True)
                        cp = Point()
                        cp.x, cp.y, cp.z = p
                        pose_t = self.l.endpoint_pose()["position"]
                        p_t = np.array([pose_t.x, pose_t.y, pose_t.z])
                        self.trp_t = np.append(self.trp_t, p_t)
                        self.v_dot.points.append(cp)
                        self.r_dot.points.append(pose_t)
                        self.mrk_pub.publish(self.v_dot)
                        # self.mrk_pub.publish(self.r_dot)
                else:
                    if 'BAXTER' in self.str_flag:
                        self.trp_r = self.trp_r.reshape((-1, 3))
                        self.trp_t = self.trp_t.reshape((-1, 3))
                        self.p0.reverse()
                        for p in self.p0:
                            j_a = self.try_move(p)
                            if j_a:
                                self.l.move_to_joint_positions(j_a)
                        self.l.move_to_joint_positions(self.init_pose)
                        self.p0 = []
                        robot_state(False)
                    elif 'VIRTUAL' in self.str_flag:
                        self.trp_v = self.trp_v.reshape((-1, 3))
                    print("Done")
                    self.str_flag = ''

            # if self.mrk_detector.cv_wrp.shape[0] > 0:
            #     wrp_img = np.copy(self.mrk_detector.cv_wrp)[0:self.h, 0:self.w]
            #     self.bg = cv2.cvtColor(wrp_img, cv2.COLOR_BGRA2BGR)

            alpha = self.alpha / 255
            foreground = cv2.multiply(self.img, 1-alpha)
            background = cv2.multiply(self.bg, alpha)
            result = cv2.add(foreground, background)
            cv2.imshow(self.wnd, result)
            self.shutdown = (cv2.waitKey(1) == ord('q'))

            if self.shutdown:
                break

        cv2.destroyAllWindows()


def main():
    # DataRecorder object
    dr = DataRecorder("Handwriting DMP GUI")
    # DataRecorder Infinite Loop
    dr.loop()


if __name__ == '__main__':
    main()
