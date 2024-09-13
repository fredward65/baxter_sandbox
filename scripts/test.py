#!/usr/bin/env python

import rospy, baxter_interface, argparse, numpy as np

from baxter_core_msgs.msg import AssemblyState, EndpointState
from baxter_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
from baxter_interface import CHECK_VERSION
from baxter_interface.limb import Limb
from baxter_interface.gripper import Gripper
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from tf.transformations import quaternion_from_euler as eu_to_q
from std_msgs.msg import Header
from sensor_msgs.msg import JointState

def tic():
    import time
    global timer
    timer = time.time()

def toc():
    import time
    if 'timer' in globals():
        print "Module runtime: %5.3f s"%(time.time() - timer)
    else:
        print 'Error: Timer not set'

def state(flag):
    while rospy.wait_for_message('/robot/state', AssemblyState).enabled <> flag:
        rs = baxter_interface.RobotEnable(CHECK_VERSION)
        if flag:
            rs.enable()
        else:
            rs.disable()

def build_pose(px,py,pz,q):
    o = Quaternion(q[0],q[1],q[2],q[3])
    pose_stamped = PoseStamped()
    hdr = Header(stamp=rospy.Time.now(),frame_id='base')
    pose = Pose(position = Point(x=px,y=py,z=pz),orientation = o)
    pose_stamped.header = hdr
    pose_stamped.pose = pose
    return pose_stamped

def main():
    parser = argparse.ArgumentParser(description="Arm Selector")
    parser.add_argument('-l', '--left', const='left',
                        dest='actions', action='append_const', help='Use left arm')
    parser.add_argument('-r', '--right', const='right',
                        dest='actions', action='append_const', help='Use right arm')
    arg = parser.parse_args(rospy.myargv()[1:])
    if arg.actions == None:
        parser.print_usage()
        parser.exit(0, "No arm selected.\n")
    try:
        for act in arg.actions:
            if act == 'right':
                m = -1
                side = "right"
            else:
                m = 1
                side = "left"
    except Exception, e:
        rospy.logerr(e.strerror)

    print "Starting node..."
    rospy.init_node('arm_control', anonymous=True)
    print "Node started"
    state(True)
    l = Limb(side)
    l.move_to_neutral()    
    g = Gripper(side)
    if g.error():
            g.reset()
    if (not g.calibrated() and g.type() != 'custom'):
        g.calibrate()
    g.open(True)
    #preva = l.joint_angles()
    print "Current position:\n", rospy.wait_for_message("/robot/limb/%s/endpoint_state"%side, EndpointState).pose.position
    rospy.sleep(1)

    print "Starting sequence..."
    ns = "/ExternalTools/%s/PositionKinematicsNode/IKService" % side
    rospy.wait_for_service(ns)
    iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
    eu = {'a': 0, 'b': np.pi, 'c': m*0}
    q = eu_to_q(eu['a'],eu['b'],eu['c'])
    goal_world = [build_pose( 0.60, 0.50*m, -0.15,q),
                  build_pose( 0.60, 0.50*m,  0.30,q),
                  build_pose( 0.60, 0.20*m,  0.30,q),
                  build_pose( 0.60, 0.20*m, -0.15,q)]
    gripper_state = [False, False, False, True]
    ikreq = SolvePositionIKRequest()
    for goal in goal_world:
        ikreq.pose_stamp.append(goal)
    tic()
    try:
        resp = iksvc(ikreq)
    except rospy.ServiceException,e:
        rospy.loginfo("Service call failed: %s" % (e,))
    for i, sresp in enumerate(resp.isValid):
        if (resp.isValid[i]):
            rospy.loginfo("Success! Valid Joint Solution")
            toc()
            print "Current position:\n", goal.pose.position
            tic()
            joint_angles = dict(zip(resp.joints[i].name,resp.joints[i].position))
            l.move_to_joint_positions(joint_angles)
            toc()
            if gripper_state[i]:
                g.open(True)
            else:
                g.command_position(50, True)
        else:
            rospy.loginfo("Error: No Valid Joint Solution")
            toc()
        rospy.sleep(0.5)
    print "Sequence finished"
    
    rospy.sleep(1)
    l.move_to_neutral() 
    # l.move_to_joint_positions(preva)
    state(False)
    print "End of script"

if __name__ == '__main__':
    main()
