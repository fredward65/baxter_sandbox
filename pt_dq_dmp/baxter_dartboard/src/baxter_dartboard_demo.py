#!/usr/bin/env python

import numpy as np
import rospkg
import rospy
from copy import deepcopy
from custom_tools.ik_tools import LimbManager, Trajectory
from custom_tools.math_tools import dx_dt, quat_rot, twist_from_dq_list, vel_from_twist
from custom_tools.projectile_model import ProjectileModel
from dual_quaternions import DualQuaternion
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import DeleteModel, SetModelState, SpawnModel
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Twist, Vector3
from pt_dq_dmp import PTDQMP
from pyquaternion import Quaternion as quat
from std_msgs.msg import Empty, Header


def load_gazebo_ball(id_=0, pose=Pose(position=Point(x=0.70, y=0.15, z=1.05)), reference_frame="world"):
    # Get Model's Path
    model_path = rospkg.RosPack().get_path('baxter_dartboard')+"/models/"
    # Load Block URDF
    block_xml = ''
    with open (model_path + "ball.urdf", "r") as block_file:
        block_xml = block_file.read().replace('\n', '')
    # Spawn Block URDF
    rospy.wait_for_service('/gazebo/spawn_urdf_model')
    try:
        spawn_urdf = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        resp_urdf = spawn_urdf("ball_%i" % id_, block_xml, "/", pose, reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn URDF service call failed: {0}".format(e))


def delete_gazebo_balls():
    """
    Delete Gazebo Models on ROS Exit
    """
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        gazebo_msg = rospy.wait_for_message("/gazebo/model_states", ModelStates)
        for name in gazebo_msg.name:
            if name.find('ball') >= 0:
                resp_delete = delete_model(name)
    except rospy.ServiceException, e:
        rospy.loginfo("Delete Model service call failed: {0}".format(e))


def get_gazebo_object_pose(obj_name):
    obj_pose = None
    flag_table = False
    while not flag_table:
        gazebo_msg = rospy.wait_for_message("/gazebo/model_states", ModelStates)
        flag_table = True if obj_name in gazebo_msg.name else False
        if flag_table:
            idx = gazebo_msg.name.index(obj_name)
            obj_pose = gazebo_msg.pose[idx]
    return obj_pose


def dual_quaternion_from_geometry_pose(pose):
    q = pose.orientation
    p = pose.position
    dq = DualQuaternion.from_quat_pose_array([q.w, q.x, q.y, q.z, p.x, p.y, p.z])
    return dq


def gen_data():
    t = np.linspace(0, 1, num=500)
    x = .50 * (t ** 2)
    y = .75 + (t * 0)
    z = t * 0
    y_ = np.c_[x, y, z]
    return y_, t


def set_gazebo_state(state):
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state(state)
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e


def launch_ball(dq, vel, id_, trf=DualQuaternion.from_dq_array([1, 0, 0, 0, 0, 0, 0, 0])):
    # Ball ModelState
    ball_state = ModelState()
    ball_state.model_name = 'ball_%i' % id_
    ball_state.reference_frame = 'world'
    # Pose Assignment - Force ball to be at the goal pose
    pos = (trf * dq).translation()
    ball_state.pose.position = Point(x=pos[0], y=pos[1], z=pos[2])
    ball_state.pose.orientation = Quaternion(x=dq.q_r.x, y=dq.q_r.y, z=dq.q_r.z, w=dq.q_r.w)
    # Twist Assignment - Force ball to move at the goal twist
    ball_state.twist.linear = Vector3(x=vel[0], y=vel[1], z=vel[2])
    # ball_state.twist.angular = Vector3(x=tw.q_r.x, y=tw.q_r.y, z=tw.q_r.z)
    load_gazebo_ball(id_=id_, pose=ball_state.pose)
    set_gazebo_state(ball_state)


def main():
    """
    Baxter Dartboard Demo
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    rospy.init_node("baxter_dartboard_demo")

    # Wait for the All Clear from emulator startup
    rospy.wait_for_service('/gazebo/set_model_state')
    rospy.wait_for_message("/robot/sim/started", Empty)
    get_gazebo_object_pose('wall')

    # Get data from file
    data_path = rospkg.RosPack().get_path('baxter_dartboard') + '/resources' + '/demo_throw_left_5.csv'
    data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
    _, d_idx, counts = np.unique(data[:, 1:], axis=0, return_index=True, return_counts=True)
    data = data[np.sort(d_idx), :]

    # Create Limb Manager
    limb = 'left'
    limb_mng = LimbManager(limb, verbose=False)
    limb_mng.gripper_open()
    # World to Base Transformation
    dq_w2b = DualQuaternion.from_quat_pose_array(np.array([1, 0, 0, 0, 0, 0, .93]))

    # Parse data to time, rotation and translation
    t = np.linspace(0, (4/4) * (data[-1, 0] - data[0, 0]), num=data.shape[0])
    r = data[:, 4:]
    p = data[:, 1:4]
    # p = .50 * (data[:, 1:4] - data[0, 1:4]) + data[0, 1:4]  # Scale trajectory

    dmp_obj = PTDQMP(n=100, alpha_y=20)
    dq_list = dmp_obj.dq_from_pose(r, p)
    tw_list = twist_from_dq_list(t, dq_list)

    # Train DMP Model
    dmp_obj.train_model(t, dq_list)

    # rot = quat._from_axis_angle(np.array([0, 0, 1]), -.125 * np.pi)
    # dq_of = DualQuaternion.from_quat_pose_array(np.append(rot.elements, [.0, .0, .0]))
    dq0 = dq_list[0]
    dqg = dq_list[-1]
    # dq0 = dq_list[0] * dq_of
    # dqg = dq0 * dq_list[0].inverse() * dq_list[-1]
    tw0 = DualQuaternion.from_dq_array(np.zeros(8))
    twg = tw_list[-1]

    # Aim towards target [PENDING]
    target = get_gazebo_object_pose('target_1')
    y_0 = np.array(dqg.translation())
    dq_trg = dq_w2b.inverse() * dual_quaternion_from_geometry_pose(target)
    y_l = np.array(dq_trg.translation())
    q_k = dq_trg.q_r
    pm = ProjectileModel()
    dy0, tf = pm.solve(y_0, y_l, q_k)
    dq_g, tw_g = pm.compute_dq(y_0)

    # Assign value for Tau
    tau = 1
    # tau = np.linalg.norm(vel_from_twist(dq_list[-1], tw_list[-1]).elements) /\
    #       np.linalg.norm(vel_from_twist(dq_g, tw_g).elements)
    t_f = t[-1] * tau
    print('Tau : %5.3f -> Period : %5.4f s' % (tau, t_f))

    # Precompute trajectory
    # dq = dq0
    # tw = tw0
    t_ = np.linspace(0, 2, num=100)
    dq_l, tw_l = dmp_obj.fit_model(t_, dq0, tw0, dqg, tau=tau, twg=None)
    t_idx = np.where(t_ < tau * t[-1])[0][-1]
    r_, p_ = dmp_obj.pose_from_dq(dq_l)

    freq = 10  # 1 / np.mean(np.diff(t_))
    print('REP FREQ : %5.5f' % freq)

    print("DEMONSTRATED, ESTIMATED, AND RECONSTRUCTED GOAL")
    print(np.array2string(np.array(dqg.translation()), precision=3))
    print(np.array2string(np.array(dq_g.translation()), precision=3))
    print(np.array2string(np.array(dq_l[t_idx].translation()), precision=3))

    vel_d = quat(vector=dx_dt(t, p)[-1])  # vel_from_twist(dqg, twg)
    vel_l = vel_from_twist(dq_g, tw_g)
    vel_f = quat(vector=dx_dt(t_, p_)[t_idx])  # vel_from_twist(dq_l[t_idx], tw_l[t_idx])
    ang_d = quat_rot(vel_d, vel_f)
    ang_l = quat_rot(vel_l, vel_f)
    rtdf = np.linalg.norm(vel_d.elements) / np.linalg.norm(vel_f.elements)
    rtlf = np.linalg.norm(vel_l.elements) / np.linalg.norm(vel_f.elements)
    print("DEMONSTRATED, ESTIMATED, RECONSTRUCTED")
    print(np.array2string(vel_d.elements[1:], precision=3))
    print(np.array2string(vel_l.elements[1:], precision=3))
    print(np.array2string(vel_f.elements[1:], precision=3))
    print("ANGLE FROM VEL_D TO VEL_F : %5.3f" % ang_d.angle)
    print("ANGLE FROM VEL_L TO VEL_F : %5.3f" % ang_l.angle)
    print("TAU : %5.3f , RATIO D-F : %5.3f, RATIO L-F : %5.3f" % (tau, rtdf, rtlf))

    plt.plot(t, p, '--', t_, p_)
    plt.show()

    # Launch Ball from Reconstructed Conditions
    launch_ball(dq_l[t_idx], vel_f.elements[1:], 500, trf=dq_w2b)

    # Launch Ball from Estimated Conditions
    # launch_ball(dq_g, tw_g, 501, trf=dq_w2b)

    # Precompute angles from trajectory
    grp_val = 20
    grp_list = []
    jnt_list = []
    for dq_i, t_i in zip(dq_l, t_):
        jnt_i = limb_mng.solve_pose(limb_mng.pose_from_dq(dq_i))
        jnt_list.append(jnt_i)
        if t_i < t_[t_idx]:
            grp_list.append(grp_val)
        else:
            grp_list.append(100)

    # traj = Trajectory('left')
    # rospy.on_shutdown(traj.stop)
    # for t_i, jnt_i in zip(t_, jnt_list):
    #     traj.add_point(jnt_i.values(), t_i)
    # traj.start()
    # traj.wait(100)
    # return

    # vel_list = limb_mng.vel_from_joints(jnt_list, t_)

    # Remove models from the scene on shutdown
    rospy.on_shutdown(delete_gazebo_balls)

    num = 3
    p_final_arr = np.empty(num, dtype=object)

    limb_mng.move_to_start()
    starting_pose = limb_mng.pose_from_dq(dq0)

    for i_ in range(num):
        dmp_obj.reset_t()

        # limb_mng.restart_ik()
        limb_mng.move_to_joint_position(jnt_list[0])
        print("INITIAL POSE SET")

        # Compute Ball Pose
        limb_endpose = limb_mng.get_limb_pose()  # get_gazebo_object_pose('block')
        ball_pose = Pose(position=limb_endpose['position'], orientation=limb_endpose['orientation'])
        ball_pos = ball_pose.position
        ball_rot = ball_pose.orientation
        ball_pos = np.array([ball_pos.x, ball_pos.y, ball_pos.z])
        ball_rot = np.array([ball_rot.w, ball_rot.x, ball_rot.y, ball_rot.z])
        dq_ball = DualQuaternion.from_quat_pose_array(np.append(ball_rot, ball_pos))
        dq_end = dq_w2b * dq_ball
        ball_p = dq_end.translation()
        ball_r = dq_end.q_r
        ball_end = Pose(Point(x=ball_p[0], y=ball_p[1], z=ball_p[2]),
                        Quaternion(x=ball_r[1], y=ball_r[2], z=ball_r[3], w=ball_r[0]))
        # Load Ball Gazebo Model via Spawning Services
        id_ = i_ + 1
        load_gazebo_ball(id_=id_, pose=ball_end)
        limb_mng.gripper_close(val=grp_val)

        rospy.sleep(0.5)

        print("STARTING RECONSTRUCTION...")
        p_final = []
        pos_f = limb_mng.get_limb_pose()['position']
        thrown = False
        t0 = rospy.get_time()
        tl = t_[t_idx] + t0
        current_t = t0
        rate = rospy.Rate(freq)
        # for jnt_i, vel_i, val_i, t_i in zip(jnt_list, vel_list, grp_list, t_ + t0):
        for t_i, jnt_i, val_i in zip(t_ + t0, jnt_list, grp_list):
        # for t_i, vel_i, val_i in zip(t_ + t0, vel_list, grp_list):
            # while current_t < t_i:
            #     current_t = rospy.get_time()
            limb_mng.set_joint_position(jnt_i)
            # limb_mng.set_velocities(vel_i)
            p_final.append(limb_mng.get_limb_pose()['position'])
            # limb_mng.gripper_close(val=val_i)
            if not t_i < tl and not thrown:
                limb_mng.gripper_open()
            #     pos_f = limb_mng.get_limb_pose()['position']
            #     t_f = current_t - t0
            #     thrown = True
            rate.sleep()
        rospy.sleep(0.5)

        for i, p_i in enumerate(p_final):
            p_final[i] = np.array([p_i.x, p_i.y, p_i.z])
        p_final = np.array(p_final).reshape((-1, 3))
        p_final_arr[i_] = p_final

        # print("BALL LAUNCHED AT %5.4f s" % t_f)
        # pos_f = np.array([pos_f.x, pos_f.y, pos_f.z])
        # pos_g = np.array(dq_l[t_idx].translation())
        # print("GOAL POS : " + np.array2string(pos_g, precision=3))
        # print("TRUE POS : " + np.array2string(pos_f, precision=3))
        # print("ERROR NORM : %5.3f " % np.linalg.norm(pos_g - pos_f))

    # Plot
    plt.axvline(tau * t[-1])
    plt.plot(t, p, ':', t_, p_, '--')
    for p_final in p_final_arr:
        plt.plot(t_, p_final)
    plt.show()

    # Finishing
    limb_mng.move_to_start()
    limb_mng.disable()


if __name__ == '__main__':
    main()
