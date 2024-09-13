#!/usr/bin/env python

import argparse
import rospy
import baxter_interface
from baxter_interface import CHECK_VERSION, Limb
from baxter_core_msgs.msg import DigitalIOState
from tf.transformations import euler_from_quaternion as q_to_eu


class Button(object):
    def __init__(self, btn):
        self.flag = False
        rospy.Subscriber(btn, DigitalIOState, self.callback)

    def callback(self, msg):
        self.flag = msg.state

    def set_flag(self, val):
        self.flag = val

    def wait(self):
        while not self.flag:
            pass


def main():
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt, description=main.__doc__)
    required = parser.add_argument_group('required arguments')
    required.add_argument('-f', '--file', dest='filename', required=True,
                          help='the file name to record to')
    required.add_argument('-l', '--limb', dest='limb', required=True,
                          help='the limb to record, either left or right')
    parser.add_argument('-r', '--record-rate', type=int, default=250, dest='rate',
                        help='rate at which to record (default: 250)')
    args = parser.parse_args(rospy.myargv()[1:])
    if args.limb is (not "left" and not "right") or None:
        parser.print_usage()
        parser.exit(0, "No arm selected.\n")

    sbutton1 = '/robot/digital_io/%s_button_ok/state' % args.limb
    sbutton2 = '/robot/digital_io/%s_button_back/state' % args.limb
    btn_1 = Button(sbutton1)
    btn_2 = Button(sbutton2)

    print("Initializing node... ")
    rospy.init_node("end_effector_pose_recorder")
    print("Getting robot state... ")
    rs = baxter_interface.RobotEnable(CHECK_VERSION)

    def stop_callback():
        rs.disable()

    _limb = Limb(args.limb)
    rospy.on_shutdown(stop_callback)

    print("Enabling robot... ")
    rs.enable()

    _rate = rospy.Rate(args.rate)

    with open(args.filename, 'w') as f:
        f.write('time,posx,posy,posz,rotx,roty,rotz,rotw\n')
        print("Press OK button to start recording")
        btn_1.wait()
        print("Recording. Press Ctrl-C or Back button to stop.")
        _start_time = rospy.get_time()
        while not rospy.is_shutdown() and not btn_2.flag:
            mrk = _limb.endpoint_pose()
            pos, rot = mrk["position"], mrk["orientation"]
            ti = rospy.get_time() - _start_time
            vlist = [ti, pos.x, pos.y, pos.z, rot.x, rot.y, rot.z, rot.w]
            f.write(','.join(map(str, vlist)) + '\n')
            _rate.sleep()

    print("\nDone.")


if __name__ == '__main__':
    main()
