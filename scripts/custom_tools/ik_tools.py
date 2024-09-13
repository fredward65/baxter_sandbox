#!/usr/bin/env python

import numpy as np
import rospy
from baxter_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from std_msgs.msg import Header


class IK_Limb:
    """
    Baxter's Inverse Kinematics helper class
    """
    def __init__(self, limb, verbose=False):
        """
        Constructor

        Parameters
        ----------
        limb : str
            Limb to be queried, left or right
        verbose : bool
            Verbose mode. True to enable rospy.loginfo messages
        """
        ns = "/ExternalTools/%s/PositionKinematicsNode/IKService" % limb
        rospy.wait_for_service(ns, timeout=None)
        self.iksvc = rospy.ServiceProxy(ns, SolvePositionIK, persistent=True)
        self.verbose = verbose

    def build_pose(self, pos, rot):
        """
        Builds pose from pose data

        Parameters
        ----------
        pos : numpy.ndarray
            Cartesian position data, [x, y, z]
        rot : numpy.ndarray
            Quaternion orientation data, [x, y, z, w]

        Returns
        -------
        b_pose : PoseStamped
            ROS PoseStamped pose
        """
        p = Point(x=pos[0], y=pos[1], z=pos[2])
        o = Quaternion(rot[0], rot[1], rot[2], rot[3])
        b_pose = PoseStamped()
        b_pose.header = Header(stamp=rospy.Time.now(), frame_id='base')
        b_pose.pose = Pose(position=p, orientation=o)
        return b_pose

    def ik_solve(self, pos, rot):
        """
        Solve Inverse Kinematics for a given pose

        Parameters
        ----------
        pos : numpy.ndarray
            Cartesian position data, [x, y, z]
        rot : numpy.ndarray
            Quaternion orientation data, [x, y, z, w]

        Returns
        -------
        joint_angles : numpy.ndarray
            Joint angles from pose
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


def try_float(x):
    try:
        return float(x)
    except ValueError:
        return None


def clean_line(line, names):
    line = [try_float(x) for x in line.rstrip().split(',')]
    combined = zip(names[1:], line[1:])
    cleaned = [x for x in combined if x[1] is not None]
    command = dict(cleaned)
    return command, line


def map_file(filename):
    print("Mapping file %s" % filename)
    with open(filename, 'r') as f:
        lines = f.readlines()
    keys = lines[0].rstrip().split(',')
    i = 0
    print("Starting...")
    _cmd, _raw = clean_line(lines[1], keys)
    bib = np.empty([len(lines[1:]), len(_raw)])
    for line in lines[1:]:
        i += 1
        cmd, values = clean_line(line, keys)
        bib[i - 1, :] = values
    print("Finished")
    return keys, bib


def main():
    pass


if __name__ == '__main__':
    main()
