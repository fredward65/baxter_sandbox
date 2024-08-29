#!/usr/bin/env python

import numpy as np
import rospy
from baxter_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from std_msgs.msg import Header


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
        self.iksvc = rospy.ServiceProxy(ns, SolvePositionIK, persistent=True)
        self.verbose = verbose

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
        b_pose = PoseStamped()
        b_pose.header = Header(stamp=rospy.Time.now(), frame_id='base')
        b_pose.pose = Pose(position=p, orientation=o)
        return b_pose

    def ik_solve(self, pos, rot):
        """
        Solve Inverse Kinematics for a given pose

        :param numpy.ndarray pos: Cartesian position data, [x, y, z]
        :param numpy.ndarray rot: Quaternion orientation data, [w, x, y, z]
        :return: Joint angles from pose
        :rtype: numpy.ndarray
        """
        pose = self.build_pose(pos, rot)

        ikreq = SolvePositionIKRequest()
        ikreq.pose_stamp.append(pose)

        joint_angles = None
        try:
            resp = self.iksvc(ikreq)
            if resp.isValid[0]:
                if self.verbose:
                    rospy.loginfo("Success! Valid Joint Solution")
                joint_angles = dict(zip(resp.joints[0].name, resp.joints[0].position))
            else:
                rospy.logerr("Error: No Valid Joint Solution")
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % (e,))
        return joint_angles


def main():
    pass


if __name__ == '__main__':
    main()
