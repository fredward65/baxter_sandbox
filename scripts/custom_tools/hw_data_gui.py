#!/usr/bin/env python

from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import baxter_interface
import matplotlib.pyplot as plt
import numpy as np
import pygame
import rospy
from baxter_core_msgs.msg import AssemblyState
from baxter_interface import CHECK_VERSION
from baxter_interface.limb import Limb
from ik_tools import IK_Limb
from pose_dmp import PoseDMP
from tf.transformations import quaternion_multiply, euler_from_quaternion as q_to_eu, quaternion_from_euler as eu_to_q
from visualization_msgs.msg import Marker


def rotate_v_q(v, q):
    v = quaternion_multiply([q[0], q[1], q[2], q[3]], [v[0], v[1], v[2], 0])
    v = quaternion_multiply(v, [-q[0], -q[1], -q[2], q[3]])
    return v[0:3]


def roundline(srf, color, start, end, radius=1):
    pygame.draw.line(srf, color, start, end, width=radius)


def robot_state(flag):
    rs = baxter_interface.RobotEnable(CHECK_VERSION)
    while rospy.wait_for_message('/robot/state', AssemblyState).enabled != flag:
        if flag:
            rs.enable()
        else:
            rs.disable()


def main():
    side = 'left'
    print("Starting node...")
    rospy.init_node("reach_%s_arm" % side, anonymous=True)
    init_flag = rospy.wait_for_message('/robot/state', AssemblyState).enabled

    l = Limb(side)
    ik_l = IK_Limb(side, verbose=True)
    init_pose = l.joint_angles()

    def shutdown_hook():
        print("Killing rospy process")
        robot_state(init_flag)
        pygame.quit()

    rospy.on_shutdown(shutdown_hook)
    print("Node started")

    width = 800
    height = 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Handwriting DMP GUI")
    screen.fill((255, 255, 255))

    draw_on = False

    color = (0, 0, 0)
    radius = 3

    c_pos = []
    last_pos = []

    dt = 0.001
    t = np.empty(0)
    traj = np.empty(0)
    tref = 0

    try:
        while True:
            e = pygame.event.poll()

            if e.type == pygame.QUIT:
                raise StopIteration

            if e.type == pygame.MOUSEBUTTONDOWN:
                draw_on = True
                screen.fill((255, 255, 255))
                c_pos = e.pos
                last_pos = c_pos
                dt = 0.001
                t = np.empty(0)
                traj = np.empty(0)
                tref = np.array(rospy.get_time())

            if e.type == pygame.MOUSEBUTTONUP:
                draw_on = False
                traj = traj.reshape((-1, 2))
                if np.linalg.norm(traj - traj[0, :]) > 0:
                    t = t - t[0]
                    traj = np.c_[traj, np.zeros(len(t))]
                    traj[:, 1] = height - traj[:, 1]
                    traj = np.divide(traj, 100)

                    t = np.linspace(0, t[-1], num=len(t))

                    n = int(np.floor(len(t)*.2))
                    n = n if n < 200 else 200
                    n = n if n > 25 else 25

                    dmp_obj = PoseDMP(n=n, alpha_y=8)

                    print("Training model...")
                    dmp_obj.train_model_p(t, traj)

                    y0 = traj[0, :]
                    yg = traj[-1, :]
                    dy0 = 0 * y0

                    fac = 1.1
                    nt = np.linspace(0, fac * t[-1], num=(fac * len(t)))
                    dt = np.mean(np.diff(nt))

                    print("Fitting virtual model...")
                    y_n = y0
                    dy_n = dy0
                    for i, ct in enumerate(nt):
                        y_p = 100 * np.copy(y_n.ravel())
                        y_n, dy_n, _ = dmp_obj.fit_step_p(ct, y_n, dy_n, y0, yg)
                        p = 100 * np.copy(y_n[0])
                        y_p[1] = height - y_p[1]
                        p[1] = height - p[1]
                        roundline(screen, (255, 0, 0), y_p[0:2], p[0:2], 2)
                        pygame.event.pump()
                        pygame.display.flip()

                    print("Detecting ArUco marker...")
                    while True:
                        vmrk_o = rospy.wait_for_message("/detected_marker", Marker)
                        if vmrk_o.id == 555:
                            break

                    tau = 4

                    p_ = vmrk_o.pose.position
                    q_ = vmrk_o.pose.orientation
                    p_off = np.array([p_.x, p_.y, p_.z])
                    q_off = np.array([q_.x, q_.y, q_.z, q_.w])

                    p_off += rotate_v_q([.12, .18, .10], q_off)

                    y0 = p_off + rotate_v_q(.05 * traj[0, :], q_off)
                    yg = p_off + rotate_v_q(.05 * traj[-1, :], q_off)
                    dy0 = 0 * y0
                    dmp_obj.set_pg_p(rotate_v_q([0, 0, 1], q_off))

                    q_off = quaternion_multiply(q_off, [0, 0, .707, .707])
                    q = quaternion_multiply(q_off, [0, .707, 0, .707])

                    p0 = p_off + np.array([-.14, .00, .14])
                    zoff = .15

                    print("Fitting RL model...")
                    j_angles = ik_l.ik_solve(p0, q)
                    if j_angles:
                        robot_state(True)
                        l.move_to_joint_positions(j_angles)
                        ct = 0.0
                        dmp_obj.reset_t()
                        y_n, dy_n, _ = dmp_obj.fit_step_p(ct, y0, dy0, y0, yg, tau=tau)
                        p = y_n.ravel()
                        p[2] += zoff
                        j_angles = ik_l.ik_solve(p, q)
                        l.move_to_joint_positions(j_angles)
                        p[2] -= zoff
                        j_angles = ik_l.ik_solve(p, q)
                        l.move_to_joint_positions(j_angles)
                        # Time loop
                        tref = rospy.get_time()
                        while ct <= nt[-1] * tau:
                            ct = rospy.get_time() - tref
                            y_n, dy_n, _ = dmp_obj.fit_step_p(ct, y_n, dy_n, y0, yg, tau=tau)
                            p = y_n.ravel()
                            j_angles = ik_l.ik_solve(p, q)
                            if j_angles:
                                l.set_joint_positions(j_angles, raw=True)
                            pygame.event.pump()
                        p[2] += zoff
                        j_angles = ik_l.ik_solve(p, q)
                        l.move_to_joint_positions(j_angles)
                        j_angles = ik_l.ik_solve(p0, q)
                        l.move_to_joint_positions(j_angles)
                        l.move_to_joint_positions(init_pose)
                        robot_state(init_flag)
                    else:
                        rospy.logerr("Possition not possible")

            if e.type == pygame.MOUSEMOTION:
                last_pos = c_pos
                c_pos = e.pos

            if e.type == pygame.KEYDOWN:
                print(e.key)

            if draw_on:
                t = np.append(t, rospy.get_time() - tref)
                roundline(screen, color, c_pos, last_pos, radius)
                traj = np.append(traj, c_pos)

            pygame.display.flip()
            rospy.sleep(dt)

    except StopIteration:
        pass


if __name__ == '__main__':
    main()
