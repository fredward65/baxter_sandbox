#!/usr/bin/env python

import actionlib
import baxter_interface
import numpy as np
import rospy
import tf2_ros
from baxter_core_msgs.msg import AssemblyState
from baxter_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from copy import copy
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from math_tools import dx_dt
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose
from trajectory_msgs.msg import JointTrajectoryPoint


class IKLimb(object):
    """
    Baxter's Inverse Kinematics helper class
    """
    def __init__(self, limb, verbose=False):
        """
        Baxter's Inverse Kinematics helper object

        :param str limb: Limb to be queried, left or right
        :param bool verbose: Verbose mode. True to enable rospy.loginfo messages
        """
        ns = "/ExternalTools/%s/PositionKinematicsNode/IKService" % limb
        rospy.wait_for_service(ns, timeout=None)
        self._limb = limb
        self._iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        self._verbose = verbose
        self._seed = None

    @staticmethod
    def build_pose(pos, rot):
        """
        Build pose from pose data

        :param numpy.ndarray pos: Cartesian position data, [x, y, z]
        :param numpy.ndarray rot: Quaternion orientation data, [w, x, y, z]
        :return: ROS PoseStamped pose
        :rtype: PoseStamped
        """
        p = Point(x=pos[0], y=pos[1], z=pos[2])
        o = Quaternion(rot[3], rot[0], rot[1], rot[2])
        b_pose = IKLimb.build_pose_from_pq(p, o)
        return b_pose

    @staticmethod
    def build_pose_from_pq(point, quaternion):
        """
        Build pose from Point and Quaternion

        :param Point point: geometry_msg Point
        :param Quaternion quaternion: geometry_msg Quaternion
        :return: Stamped Pose
        :rtype: PoseStamped
        """
        b_pose = PoseStamped()
        b_pose.header = Header(stamp=rospy.Time.now(), frame_id='base')
        b_pose.pose = Pose(position=point, orientation=quaternion)
        return b_pose

    def ik_solve(self, pos, rot, auto=True):
        """
        Solve Inverse Kinematics for a given position and orientation

        :param numpy.ndarray pos: Cartesian position data, [x, y, z]
        :param numpy.ndarray rot: Quaternion orientation data, [w, x, y, z]
        :param bool auto: Use Auto Seed
        :return: Joint angles from pose
        :rtype: dict
        """
        pose = self.build_pose(pos, rot)
        joint_angles = self.ik_solve_from_pose(pose, auto=auto)
        return joint_angles

    def ik_solve_from_pq(self, point, quaternion, auto=True):
        """
        Solve Inverse Kinematics for a given position and orientation

        :param Point point: geometry_msg Point
        :param Quaternion quaternion: geometry_msg Quaternion
        :param bool auto: Use Auto Seed
        :return: Joint angles from pose
        :rtype: dict
        """
        pose = self.build_pose_from_pq(point, quaternion)
        joint_angles = self.ik_solve_from_pose(pose, auto=auto)
        return joint_angles

    def ik_solve_from_pose(self, pose, auto=True):
        """
        Solve Inverse Kinematics for a given stamped pose

        :param PoseStamped pose: Stamped Pose
        :param bool auto: Use Auto Seed
        :return: Joint angles from pose
        :rtype: dict
        """
        ikreq = SolvePositionIKRequest()
        if not auto:
            ikreq.seed_mode = ikreq.SEED_CURRENT
        if self._seed:
            ikreq.seed_mode = ikreq.SEED_USER
            ikreq.seed_angles.append(self._seed)
        ikreq.pose_stamp.append(pose)

        joint_angles = None
        try:
            resp = self._iksvc(ikreq)
            if resp.isValid[0]:
                if self._verbose:
                    rospy.loginfo("Success! Valid Joint Solution")
                self._seed = JointState()
                self._seed.name = resp.joints[0].name
                self._seed.position = resp.joints[0].position
                joint_angles = dict(zip(resp.joints[0].name, resp.joints[0].position))
            else:
                rospy.logerr("Error: No Valid Joint Solution")
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr("Service call failed: %s" % (e,))
        return joint_angles

    def restart_service(self):
        """
        Restart IKService

        :return:
        """
        # self.iksvc.close()
        ns = "/ExternalTools/%s/PositionKinematicsNode/IKService" % self._limb
        self._iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        self._seed = None


class LimbManager(object):
    def __init__(self, limb, verbose=True, raw=True):
        self._limb_name = limb
        self._verbose = verbose
        self._limb = baxter_interface.Limb(limb)
        self._limb.set_joint_position_speed(1)
        self._limb.set_command_timeout(0.5)
        self._gripper = baxter_interface.Gripper(limb)
        self._ik_tool = IKLimb(limb)
        self._raw = raw
        # verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        self.enable()

    def enable(self):
        print("Enabling robot... ")
        while not rospy.wait_for_message('/robot/state', AssemblyState).enabled:
            self._rs.enable()

    def disable(self):
        print("Enabling robot... ")
        while rospy.wait_for_message('/robot/state', AssemblyState).enabled:
            self._rs.disable()

    def move_to_start(self):
        print("Moving the %s arm to neutral pose..." % self._limb_name)
        self._limb.move_to_neutral()
        rospy.sleep(1.0)
        print("Running. Ctrl-c to quit")

    def set_joint_position(self, joint_angles):
        self._guarded_set_joint_positions(joint_angles)

    def move_to_joint_position(self, joint_angles):
        self._guarded_move_to_joint_position(joint_angles)

    def _guarded_move_to_joint_position(self, joint_angles):
        if joint_angles:
            self._limb.move_to_joint_positions(joint_angles)
        elif self._verbose:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")

    def _guarded_set_joint_positions(self, joint_angles):
        if joint_angles:
            self._limb.set_joint_positions(joint_angles, raw=self._raw)
        elif self._verbose:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")

    def gripper_open(self):
        self._gripper.open(block=False)
        # rospy.sleep(1.0)

    def gripper_close(self, val=None):
        if val and 0 <= val <= 100:
            self._gripper.command_position(val)
        else:
            self._gripper.close()
        # rospy.sleep(1.0)

    def _servo_to_pose(self, pose):
        # servo down to release
        joint_angles = self._ik_tool.ik_solve_from_pose(pose)
        self._guarded_move_to_joint_position(joint_angles)
        return joint_angles

    def solve_pose(self, pose):
        joint_angles = self._ik_tool.ik_solve_from_pose(pose)
        return joint_angles

    def move_to_pose(self, pose):
        return self._servo_to_pose(pose)

    def set_pose(self, pose, auto=False):
        joint_angles = self._ik_tool.ik_solve_from_pose(pose, auto=auto)
        self._guarded_set_joint_positions(joint_angles)

    def set_velocities(self, velocities):
        self._limb.set_joint_velocities(velocities)

    def restart_ik(self):
        self._ik_tool.restart_service()

    @staticmethod
    def pose_from_dq(dq):
        ik_pose = Pose()
        p = dq.translation()
        q = dq.q_r
        ik_pose.position.x = p[0]
        ik_pose.position.y = p[1]
        ik_pose.position.z = p[2]
        ik_pose.orientation.w = q.w
        ik_pose.orientation.x = q.x
        ik_pose.orientation.y = q.y
        ik_pose.orientation.z = q.z
        return PoseStamped(Header(stamp=rospy.Time.now(), frame_id='base'), ik_pose)

    @staticmethod
    def vel_from_joints(j_ang_list, t):
        # Get joint positions from dictionary list
        j_arr = np.empty((len(j_ang_list), len(j_ang_list[0].values())))
        for i, j_an_i in enumerate(j_ang_list):
            j_arr[i] = j_an_i.values()
        # Compute joint velocities
        v_arr = dx_dt(t, j_arr)
        # Organize as dictionary
        vel_list = []
        for v_i, j_i in zip(v_arr, j_ang_list):
            vel_i = {}
            j_keys = j_i.keys()
            for val_i, key_i in zip(v_i, j_keys):
                vel_i[key_i] = val_i
            print(vel_i)
            vel_list.append(vel_i)
        return vel_list

    def get_limb_pose(self):
        return self._limb.endpoint_pose()

    def get_limb_twist(self):
        return self._limb.endpoint_velocity()


class Trajectory(object):
    import sys
    def __init__(self, limb, tol=0.1):
        ns = 'robot/limb/' + limb + '/'
        self._client = actionlib.SimpleActionClient(
            ns + "follow_joint_trajectory",
            FollowJointTrajectoryAction,
        )
        self._goal = FollowJointTrajectoryGoal()
        self._goal_time_tolerance = rospy.Time(tol)
        self._goal.goal_time_tolerance = self._goal_time_tolerance
        server_up = self._client.wait_for_server(timeout=rospy.Duration(5.0))
        if not server_up:
            rospy.logerr("Timed out waiting for Joint Trajectory"
                         " Action Server to connect. Start the action server"
                         " before running example.")
            rospy.signal_shutdown("Timed out waiting for Action Server")
            self.sys.exit(1)
        self.clear(limb)

    def add_point(self, positions, time):
        point = JointTrajectoryPoint()
        point.positions = copy(positions)
        point.time_from_start = rospy.Duration(time)
        self._goal.trajectory.points.append(point)

    def start(self):
        self._goal.trajectory.header.stamp = rospy.Time.now()
        self._client.send_goal(self._goal)

    def stop(self):
        self._client.cancel_goal()

    def wait(self, timeout=1.0):
        self._client.wait_for_result(timeout=rospy.Duration(timeout))

    def result(self):
        return self._client.get_result()

    def clear(self, limb):
        joints = ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']
        self._goal = FollowJointTrajectoryGoal()
        self._goal.goal_time_tolerance = self._goal_time_tolerance
        self._goal.trajectory.joint_names = [limb + '_' + joint for joint in joints]


class TransformListener(object):
    def __init__(self):
        self.exceptions = (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException)
        self.tf_bfr = tf2_ros.Buffer()
        self.tf_bfr_in = tf2_ros.BufferInterface()
        self.listener = tf2_ros.TransformListener(self.tf_bfr)
        self.trf = None

    def get_transform(self, name='base'):
        trf = None
        while not trf:
            try:
                trf = self.tf_bfr.lookup_transform(name, 'world', rospy.Time())
            except self.exceptions:
                pass
        self.trf = trf
        return trf

    def apply_transform(self, obj_pose):
        obj_pose_stamped = PoseStamped(Header(stamp=rospy.Time.now(), frame_id='base'), obj_pose)
        trf_obj = do_transform_pose(obj_pose_stamped, self.trf)
        return trf_obj


def main():
    pass


if __name__ == '__main__':
    main()
