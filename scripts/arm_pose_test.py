#!/usr/bin/env python

import argparse
import numpy as np
import rospy
from scipy.spatial.transform import Rotation as Rot
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker

global v_mrk, tf, ref, m_ref, mtxf_tf, side, targetid
m_ref = "hand"
v_mrk = Marker()
v_mrk.type = 1
v_mrk.header.frame_id = "/base"
[v_mrk.scale.x, v_mrk.scale.y, v_mrk.scale.z] = [0.1, 0.1, 0.1]
[v_mrk.color.r, v_mrk.color.g, v_mrk.color.b, v_mrk.color.blank] = [0.0, 1.0, 1.0, 0.5]


def head_callback(ref_msg):
    global v_mrk, ref, m_ref, mtxf_tf
    tflist = ref_msg.transforms
    for _tf in tflist:
        if _tf.child_frame_id == ref:
            ref = _tf.header.frame_id
            c_tf = _tf.transform
            tc_tf = np.float32([c_tf.translation.x, c_tf.translation.y, c_tf.translation.z]).ravel().reshape(3, 1)
            rc_tf = Rot.from_quat([c_tf.rotation.x, c_tf.rotation.y, c_tf.rotation.z, c_tf.rotation.w]).as_dcm()
            mtxc_tf = np.concatenate((np.c_[rc_tf, tc_tf], [[0, 0, 0, 1]]), axis=0)
            mtxf_tf = np.dot(mtxc_tf, mtxf_tf)
            if _tf.header.frame_id == "base":
                # print ref #_tf.header.frame_id, _tf.child_frame_id
                # print mtxf_tf
                ref = "%s_%s" % (side, m_ref)
                t_true = mtxf_tf[0:3, 3]
                r_true = mtxf_tf[0:3, 0:3]
                r_true = Rot.from_dcm(r_true).as_quat()
                mtxf_tf = np.diag([1, 1, 1, 1])
                vmrk.id = targetid
                vmrk.header.stamp = rospy.Time.now()
                vmrk.pose.position.x = t_true[0]
                vmrk.pose.position.y = t_true[1]
                vmrk.pose.position.z = t_true[2]
                vmrk.pose.orientation.x = r_true[0]
                vmrk.pose.orientation.y = r_true[1]
                vmrk.pose.orientation.z = r_true[2]
                vmrk.pose.orientation.w = r_true[3]


def main():
    parser = argparse.ArgumentParser(description="Arm Selector")
    parser.add_argument('-l', '--left', const='left',
                        dest='actions', action='append_const', help='Use left arm')
    parser.add_argument('-r', '--right', const='right',
                        dest='actions', action='append_const', help='Use right arm')
    arg = parser.parse_args(rospy.myargv()[1:])
    if arg.actions is None:
        parser.print_usage()
        parser.exit(0, "No arm selected.\n")
    try:
        global side, targetid
        for act in arg.actions:
            if act == 'right':
                targetid = 1
                side = "right"
            else:
                targetid = 2
                side = "left"
    except Exception, e:
        rospy.logerr(e.strerror)

    global ref, m_ref, mtxf_tf, v_mrk
    ref = "%s_%s" % (side, m_ref)
    mtxf_tf = np.diag([1, 1, 1, 1])
    print "Starting node..."
    rospy.init_node("end_effector_pose_node", anonymous=True)
    print "Node started"
    rospy.Subscriber("/tf", TFMessage, head_callback)
    pub = rospy.Publisher("/end_effector_pose", Marker, queue_size=1)
    print "Publishing end-effector marker for %s side..." % side
    while not rospy.is_shutdown():
        pub.publish(vmrk)
    print("Node stopped")


if __name__ == "__main__":
    main()
