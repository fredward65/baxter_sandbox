#!/usr/bin/env python

import baxter_interface
import matplotlib.pyplot as plt
import numpy as np
import rospy
import rospkg
import tf2_ros
from baxter_core_msgs.msg import AssemblyState
from copy import deepcopy
from custom_tools.ik_tools import IKLimb
# from custom_tools.math_tools import npa_to_dql, dql_to_npa
from custom_tools.projectile_model import ProjectileModel
from dual_quaternions import DualQuaternion
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import DeleteModel, SetModelState, SpawnModel
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Twist, Vector3
from pt_dq_dmp import PTDQMP
from pyquaternion import Quaternion as quat
from std_msgs.msg import Empty, Header
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose


def load_gazebo_ball(pose=Pose(position=Point(x=0.00, y=0.00, z=0.00)), reference_frame="world"):
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
        resp_urdf = spawn_urdf("ball", block_xml, "/", pose, reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn URDF service call failed: {0}".format(e))


def delete_gazebo_ball():
    """
    Delete Gazebo Models on ROS Exit
    """
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp_delete = delete_model("ball")
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


def set_gazebo_state(state):
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state(state)
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e


def main():
    """
    Gazebo Ball Throwing Dartboard Demo
    """
    rospy.init_node("throw_ball_demo")

    load_gazebo_ball()
    # Remove model from the scene on shutdown
    rospy.on_shutdown(delete_gazebo_ball)
    rospy.sleep(2.0)

    ball = get_gazebo_object_pose('ball')
    target = get_gazebo_object_pose('target')
    pm = ProjectileModel()
    y_0 = np.array([ball.position.x, ball.position.y, ball.position.z])
    y_l = np.array([target.position.x, target.position.y, target.position.z])
    q_k = quat([target.orientation.w, target.orientation.x,
                target.orientation.y, target.orientation.z])
    dy0, tf = pm.solve(y_0, y_l, q_k)

    rospy.wait_for_service('/gazebo/set_model_state')

    ball_state = ModelState()
    ball_state.model_name  = 'ball'
    ball_state.reference_frame = 'world'
    # Pose Assignment
    ball_state.pose.position.x = y_0[0]
    ball_state.pose.position.y = y_0[1]
    ball_state.pose.position.z = y_0[2]
    # Twist Assignment
    ball_state.twist.linear.x = dy0[0]
    ball_state.twist.linear.y = dy0[1]
    ball_state.twist.linear.z = dy0[2]
    print(ball_state)

    set_gazebo_state(ball_state)
    print("BALL LAUNCHED", dy0)

    rospy.sleep(2.0)


if __name__ == '__main__':
    main()
