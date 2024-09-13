#!/usr/bin/env python

import argparse
import baxter_interface
import matplotlib.pyplot as plt
import numpy as np
import rospy
from baxter_core_msgs.msg import AssemblyState, DigitalIOState
from custom_tools import dmp
from baxter_interface import CHECK_VERSION, Limb
from mpl_toolkits.mplot3d import Axes3D
from tf.transformations import euler_from_quaternion as q_to_eu


def state(s_flag):
    while rospy.wait_for_message('/robot/state', AssemblyState).enabled <> s_flag:
        rs = baxter_interface.RobotEnable(CHECK_VERSION)
        if s_flag:
            rs.enable()
        else:
            rs.disable()


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
    print("Playing back: %s" % (filename,))
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
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt)
    parser.add_argument('-f', '--file', metavar='PATH', required=True, help='path to input file')
    parser.add_argument('-l', '--limb', dest='limb', required=True, help='the limb to record, either left or right')
    args = parser.parse_args(rospy.myargv()[1:])
    if args.limb is (not "left" and not "right") or None:
        parser.print_usage()
        parser.exit(0, "No arm selected.\n")

    print("Initializing node... ")
    rospy.init_node("joint_position_dmp_modeling")
    print("Getting robot state... ")
    rs = baxter_interface.RobotEnable(CHECK_VERSION)
    print rs.state().enabled

    names, bib = map_file(args.file)
    t = bib[:, 0]
    bib2 = np.empty(bib.shape)
    bib2[:, 0] = t
    alphay = 4
    betay = alphay / 4
    n = 50

    flag = False
    state(True)
    print "Move robot to end pose and press button"
    while not flag:
        flag = rospy.wait_for_message('/robot/digital_io/torso_left_button_ok/state', DigitalIOState).state
    print "Button pressed"

    mrk = Limb(args.limb).endpoint_pose()
    pos = mrk["position"]
    erot = mrk["orientation"]
    rot = q_to_eu([erot.x, erot.y, erot.z, erot.w])
    endpose = {'posx': pos.x, 'posy': pos.y, 'posz': pos.z,
               'rota': rot[0], 'rotb': rot[0], 'rotc': rot[0]}
    state(False)

    for i in range(len(bib[0, 1:])):
        print "Modeling %s..." % names[i + 1]
        y = bib[:, i + 1]
        fn, x, dt = dmp.get_model(t, y, n, alphay)

        g = endpose[names[i + 1]]
        yp = y[0]
        dyp = 0
        y0 = yp
        yf = np.empty([len(y), 3])
        for j in range(len(t)):
            ddyp = alphay * (betay * (g - yp) - dyp) + (fn[j] * x[j] * (g - y0))
            dyp = dyp + ddyp * dt
            yp = yp + dyp * dt
            yf[j] = [yp, dyp, ddyp]

        plt.plot(t, yf[:, 0])
        plt.plot(t, y, '--')
        plt.plot(t, np.abs(y - yf[:, 0]), ':')
        plt.show()

        bib2[:, i + 1] = yf[:, 0]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(bib2[:, 1], bib2[:, 2], bib2[:, 3])
    plt.show()

    _filename = "dmp_%s" % args.file
    with open(_filename, 'w') as f:
        f.write(','.join([str(x) for x in names]) + '\n')
        for i in range(len(t)):
            f.write(','.join([str(x) for x in bib2[i]]) + '\n')
    print "DMP model done"


if __name__ == '__main__':
    main()
