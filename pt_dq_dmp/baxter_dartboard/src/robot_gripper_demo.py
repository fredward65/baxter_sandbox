#!/usr/bin/env python

import numpy as np
import rospy
import rospkg
from copy import deepcopy
from custom_tools.math_tools import dx_dt, quat_rot, twist_from_dq_list, vel_from_twist
from custom_tools.projectile_model import ProjectileModel
from dual_quaternions import DualQuaternion
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import DeleteModel, SetModelState, SpawnModel
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Twist, Vector3
from pt_dq_dmp import PTDQMP
from pyquaternion import Quaternion as quat
from std_msgs.msg import Empty, Header, Float64
from std_srvs.srv import Empty as EmptySrv


class GripperManager(object):
    def __init__(self):
        self.pub_plan = rospy.Publisher('/cmd_vel', Twist, queue_size=0)
        topic = '/robot_gripper/joint_%s_position_controller/command'
        self.pub_left = rospy.Publisher(topic % 'left', Float64, queue_size=0)
        self.pub_right = rospy.Publisher(topic % 'right', Float64, queue_size=0)
        self.msg = Float64()

    def open(self):
        self.msg.data = -0.5
        self.pub_left.publish(self.msg)
        self.pub_right.publish(self.msg)

    def close(self):
        self.msg.data = 0.0
        self.pub_left.publish(self.msg)
        self.pub_right.publish(self.msg)

    def cmd_vel(self, vel_vec, omg_vec):
        vel = Twist()
        vel.linear.x = vel_vec[0]
        vel.linear.y = vel_vec[1]
        vel.linear.z = vel_vec[2]
        vel.angular.x = omg_vec[0]
        vel.angular.y = omg_vec[1]
        vel.angular.z = omg_vec[2]
        self.pub_plan.publish(vel)


def load_gazebo_ball(id_=0, pose=Pose(position=Point(x=0.70, y=0.15, z=1.05)), reference_frame="world"):
    # Get Model's Path
    model_path = rospkg.RosPack().get_path('baxter_dartboard')+"/models/"
    # Load Block URDF
    block_xml = ''
    with open (model_path + "cube.urdf", "r") as block_file:
        block_xml = block_file.read().replace('\n', '')
    # Spawn Block URDF
    rospy.wait_for_service('/gazebo/spawn_urdf_model')
    try:
        spawn_urdf = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        resp_urdf = spawn_urdf("ball_%i" % id_, block_xml, "/", pose, reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn URDF service call failed: {0}".format(e))


def pause_gazebo():
    try:
        pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', EmptySrv)
        resp_urdf = pause_physics.call()
    except rospy.ServiceException, e:
        rospy.logerr("Pause Physics service call failed: {0}".format(e))


def unpause_gazebo():
    try:
        pause_physics = rospy.ServiceProxy('/gazebo/unpause_physics', EmptySrv)
        resp_urdf = pause_physics.call()
    except rospy.ServiceException, e:
        rospy.logerr("Pause Physics service call failed: {0}".format(e))


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


def set_gazebo_state(state):
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state(state)
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e


def move_gripper(dq,
                 off=DualQuaternion.from_quat_pose_array([1, 0, 0, 0, .00, .00, .00]),
                 tw=DualQuaternion.from_dq_array(np.zeros(8))):
    set_state('robot_gripper', dq * off.inverse(), tw)


def set_state(name, dq, tw):
    # Model ModelState
    model_state = ModelState()
    model_state.model_name = name
    model_state.reference_frame = 'world'
    # Pose Assignment
    pos = dq.translation()
    rot = dq.q_r
    model_state.pose.position = Point(x=pos[0], y=pos[1], z=pos[2])
    model_state.pose.orientation = Quaternion(x=rot.x, y=rot.y, z=rot.z, w=rot.w)
    # Twist Assignment
    vel = vel_from_twist(dq, tw).elements[1:]
    omg = tw.q_r
    model_state.twist.linear = Vector3(x=vel[0], y=vel[1], z=vel[2])
    # model_state.twist.angular = Vector3(x=omg.x, y=omg.y, z=omg.z)
    set_gazebo_state(model_state)



def main():
    """
    Robot Gripper Dartboard Demo
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    rospy.init_node("robot_gripper_demo")

    # Wait for the All Clear from emulator startup
    rospy.wait_for_service('/gazebo/set_model_state')
    get_gazebo_object_pose('robot_gripper')

    gripper = GripperManager()

    q_a = quat._from_axis_angle(np.array([1, 0, 0]),-.5 * np.pi)
    q_b = quat._from_axis_angle(np.array([0, 1, 0]), .5 * np.pi)
    q_c = quat._from_axis_angle(np.array([0, 1, 0]), .0 * np.pi)
    q_off = q_b * q_a
    dq_off = DualQuaternion.from_quat_pose_array(np.append(q_off.elements, [.00, .20, .00]))
    dq_off = DualQuaternion.from_quat_pose_array(np.append(q_c.elements, [0, 0, 0])) * dq_off

    # Get data from file
    data_path = rospkg.RosPack().get_path('baxter_dartboard') + '/resources' + '/demo_throw_left_5.csv'
    data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
    _, d_idx, counts = np.unique(data[:, 1:], axis=0, return_index=True, return_counts=True)
    data = data[np.sort(d_idx), :]

    # World to Base Transformation
    dq_w2b = DualQuaternion.from_quat_pose_array(np.array([1, 0, 0, 0, 0, 0, .93]))

    # Parse data to time, rotation and translation
    t = np.linspace(0, (data[-1, 0] - data[0, 0]), num=data.shape[0])
    r = data[:, 4:]
    p = data[:, 1:4]
    # p = .50 * (data[:, 1:4] - data[0, 1:4]) + data[0, 1:4]  # Scale trajectory

    dmp_obj = PTDQMP(n=100, alpha_y=20)
    dq_list = dmp_obj.dq_from_pose(r, p)
    tw_list = twist_from_dq_list(t, dq_list)

    # Train DMP Model
    dmp_obj.train_model(t, dq_list)

    dq0 = DualQuaternion.from_quat_pose_array([1, 0, 0, 0, 0, 0, .5]) * dq_list[0]
    dqg = DualQuaternion.from_quat_pose_array([1, 0, 0, 0, 0, 0, .5]) * dq_list[-1]
    # rot = quat._from_axis_angle(np.array([0, 1, 0]), -(1/2) * np.pi)
    # dq_of = DualQuaternion.from_quat_pose_array(np.append(rot.elements, [.0, .0, .0]))
    # dq0 = dq_list[0] * dq_of
    # dqg = dq0 * dq_list[0].inverse() * dq_list[-1]
    tw0 = DualQuaternion.from_dq_array(np.zeros(8))
    twg = tw_list[-1]

    # Aim towards target [PENDING]
    # target = get_gazebo_object_pose('target_1')
    # y_0 = np.array(dqg.translation())
    # dq_trg = dq_w2b.inverse() * dual_quaternion_from_geometry_pose(target)
    # y_l = np.array(dq_trg.translation())
    # q_k = dq_trg.q_r
    # pm = ProjectileModel()
    # dy0, tf = pm.solve(y_0, y_l, q_k)
    # dq_g, tw_g = pm.compute_dq(y_0)

    # Assign value for Tau
    tau = 1
    # tau = np.linalg.norm(vel_from_twist(dq_list[-1], tw_list[-1]).elements) /\
    #       np.linalg.norm(vel_from_twist(dq_g, tw_g).elements)
    print('Tau : %5.3f -> Period : %5.4f s' % (tau, t[-1] * tau))

    # Precompute trajectory
    t_ = np.linspace(0, 2, num=400)
    dq_l, tw_l = dmp_obj.fit_model(t_, dq0, tw0, dqg, tau=tau)
    t_idx = np.where(t_ <= tau * t[-1])[0][-1]
    r_, p_ = dmp_obj.pose_from_dq(dq_l)

    freq = 1 / np.mean(np.diff(t_))
    print('REP FREQ : %5.5f' % freq)

    print("DEMONSTRATED, ESTIMATED, AND RECONSTRUCTED GOAL")
    print(np.array2string(np.array(dqg.translation()), precision=3))
    # print(np.array2string(np.array(dq_g.translation()), precision=3))
    print(np.array2string(np.array(dq_l[t_idx].translation()), precision=3))

    vel_d = quat(vector=dx_dt(t, p)[-1])  # vel_from_twist(dqg, twg)
    # vel_l = vel_from_twist(dq_g, tw_g)
    vel_f = quat(vector=dx_dt(t_, p_)[t_idx])  # vel_from_twist(dq_l[t_idx], tw_l[t_idx])
    ang_d = quat_rot(vel_d, vel_f)
    # ang_l = quat_rot(vel_l, vel_f)
    rtdf = np.linalg.norm(vel_d.elements) / np.linalg.norm(vel_f.elements)
    # rtlf = np.linalg.norm(vel_l.elements) / np.linalg.norm(vel_f.elements)
    print("DEMONSTRATED, ESTIMATED, RECONSTRUCTED")
    print(np.array2string(vel_d.elements[1:], precision=3))
    # print(np.array2string(vel_l.elements[1:], precision=3))
    print(np.array2string(vel_f.elements[1:], precision=3))
    print("ANGLE FROM VEL_D TO VEL_F : %5.3f" % ang_d.angle)
    # print("ANGLE FROM VEL_L TO VEL_F : %5.3f" % ang_l.angle)
    print("TAU : %5.3f , RATIO D-F : %5.3f" % (tau, rtdf))
    # print("TAU : %5.3f , RATIO L-F : %5.3f" % (tau, rtlf))

    # Remove models from the scene on shutdown
    rospy.on_shutdown(delete_gazebo_balls)

    num = 3
    t_f = []
    dq_f = []
    p_ball = np.empty(num, dtype=object)

    for i in range(num):
        # Load Ball Gazebo Model via Spawning Services
        id_ = i
        gripper.open()
        rospy.sleep(3)

        # Compute Ball Pose
        gripper_pose = get_gazebo_object_pose('robot_gripper')
        grp_pos = gripper_pose.position
        grp_rot = gripper_pose.orientation
        dq0 = DualQuaternion.from_quat_pose_array([grp_rot.w, grp_rot.x, grp_rot.y, grp_rot.z,
                                                   grp_pos.x, grp_pos.y, grp_pos.z])
        dq0 = dq0 * dq_off
        ball_p = dq0.translation()
        ball_r = dq0.q_r.elements
        ball_end = Pose(Point(x=ball_p[0], y=ball_p[1], z=ball_p[2]),
                        Quaternion(x=ball_r[1], y=ball_r[2], z=ball_r[3], w=ball_r[0]))

        # pause_gazebo()
        load_gazebo_ball(id_=id_, pose=ball_end)
        gripper.close()
        rospy.sleep(3)
        # unpause_gazebo()
        gripper.cmd_vel([0, 5, 0], [0, 0, 0])
        rospy.sleep(.2)
        gripper.open()
        gripper.cmd_vel([0, 0, 0], [0, 0, 0])
        rospy.sleep(.5)
        gripper.cmd_vel([0, 0, 0], [0, 0, 4 * np.pi])
        rospy.sleep(1)
        gripper.cmd_vel([0, 0, 0], [0, 0, 0])


def main2():
    if True:
        print("STARTING RECONSTRUCTION...")
        cball_p = []
        dq = 1 * dq0
        tw = 1 * tw0
        dmp_obj.reset_t()
        pos_f = 1 * dq
        thrown = False
        t0 = rospy.get_time()
        tl = t_[t_idx] + t0
        tf = t_[-1] + t0
        tf_ = t0
        current_t = rospy.get_time()
        while current_t < tf:
            ti = current_t - t0
            dq, tw = dmp_obj.fit_step(ti, dq, tw, dqg, tau=tau)
            move_gripper(dq, off=dq_off, tw=(1/tau) * tw)
            # if not thrown:
            #     set_state('ball_%i' % id_, dq, (1/tau) * tw)
            if current_t > tl and not thrown:
                gripper.open()
                pos_f = 1 * dq
                tf_ = ti
                thrown = True
            # t_f.append(ti)
            # dq_f.append(dq)
            # bp = get_gazebo_object_pose('ball_%i' % id_).position
            # cball_p.append([bp.x, bp.y, bp.z])
            current_t = rospy.get_time()

        # cball_p = np.array(cball_p).reshape((-1, 3))
        # p_ball[i] = cball_p

        print("BALL LAUNCHED AT %5.4f s" % tf_)
        pos_f = np.array(pos_f.translation())
        pos_g = np.array(dq_l[t_idx].translation())
        print("GOAL POS : " + np.array2string(pos_g, precision=3))
        print("TRUE POS : " + np.array2string(pos_f, precision=3))
        print("ERROR NORM : %5.3f " % np.linalg.norm(pos_g - pos_f))

    # t_f = np.array(t_f)
    # dq_f = np.array(dq_f, dtype=DualQuaternion)
    # r_f, p_f = dmp_obj.pose_from_dq(dq_f)

    # plt.figure()
    # plt.plot(t, p, ':', t_, p_, '--', t_f, p_f)
    # plt.figure()
    # for bp in p_ball:
    #     for i in range(3):
    #         plt.subplot(310+i+1)
    #         plt.plot(bp[:, i])
    # plt.show()

    print("END OF SCRIPT")


if __name__ == '__main__':
    main()
