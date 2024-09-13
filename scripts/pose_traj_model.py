#!/usr/bin/env python

import argparse
import baxter_interface
import matplotlib.pyplot as plt
import numpy as np
import rospy
from baxter_core_msgs.msg import AssemblyState, DigitalIOState
from custom_tools.dmp_pos import DMP, qlog, qprod, qconj
from baxter_interface import CHECK_VERSION, Limb
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as Rot


def state(s_flag):
    while rospy.wait_for_message('/robot/state', AssemblyState).enabled is not s_flag:
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


def read_pose(limb, sbutton):
    flag = False
    print "Move %s arm to initial pose and press button" % limb
    while not flag:
        flag = rospy.wait_for_message(sbutton, DigitalIOState).state
    print "Button pressed"
    mrk = Limb(limb).endpoint_pose()
    pos, rot = mrk["position"], mrk["orientation"]
    pose = {'posx': pos.x, 'posy': pos.y, 'posz': pos.z,
            'rotx': rot.x, 'roty': rot.y, 'rotz': rot.z, 'rotw': rot.w}
    while rospy.wait_for_message(sbutton, DigitalIOState).state:
        pass
    return pose


def main():
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt)
    required = parser.add_argument_group('required arguments', description=main.__doc__)
    required.add_argument('-f', '--file', metavar='PATH', required=True, help='path to input file')
    required.add_argument('-l', '--limb', dest='limb', required=True, help='the limb to use, either left or right')
    parser.add_argument('-i', '--init', const=True, dest='iflag', action='append_const', help='assign initial pose')
    parser.add_argument('-g', '--goal', const=True, dest='gflag', action='append_const', help='assign goal pose')
    parser.add_argument('-n', '--ngaus', dest='n', help='number of gaussian kernels, default n = 50')
    parser.add_argument('-t', '--tau', dest='tau', help='time scale factor, default tau = 1')
    args = parser.parse_args(rospy.myargv()[1:])
    if args.limb is (not "left" and not "right") or None:
        parser.print_usage()
        parser.exit(0, "No arm selected.\n")

    print("Initializing node... ")
    rospy.init_node("joint_position_dmp_modeling")
    print("Getting robot state... ")
    rs = baxter_interface.RobotEnable(CHECK_VERSION)
    print rs.state().enabled

    print "Mapping file %s" % args.file
    names, bib = map_file(args.file)
    t = bib[:, 0]
    dt = np.mean(np.diff(t))
    print "File mapped"

    y0, yg = bib[0, 1:4], bib[-1, 1:4]
    q0, qg = bib[0, 4:], bib[-1, 4:]

    sbutton = '/robot/digital_io/torso_%s_button_ok/state' % args.limb
    rpose = [{'posx', 'posy', 'posz', 'rotx', 'roty', 'rotz', 'rotw'},
             {'posx', 'posy', 'posz', 'rotx', 'roty', 'rotz', 'rotw'}]
    if args.iflag:
        state(True)
        rpose[0] = read_pose(args.limb, sbutton)
        state(False)
    if args.gflag:
        state(True)
        rpose[1] = read_pose(args.limb, sbutton)
        state(False)
    ini_pose, end_pose = rpose

    n = int(args.n) if args.n else 50
    tau = float(args.tau) if args.tau else 1
    alphay = 4

    """ Position DMP """
    print "Modeling %s..." % names[1:4]
    y = bib[:, 1:4]

    dmp_obj = DMP(n, alphay)
    wi, x = dmp_obj.get_model_p(t, y)

    y0 = np.array([ini_pose[names[1]], ini_pose[names[2]], ini_pose[names[3]]]) \
         if args.iflag else y[0, :]
    dy0 = np.diff(y[0:2, :], axis=0)
    yg = [end_pose[names[1]], end_pose[names[2]], end_pose[names[3]]] \
         if args.gflag else y[-1, :]
    tn, yf = dmp_obj.fit_model_p(t, tau, yg, y0, dy0, wi)

    plt.plot(tn, yf[:, 0:3])
    plt.plot(t, y, '--')
    # plt.plot(t, np.abs(y - yf[:, 0:3]), ':')
    plt.show()

    """ Orientation DMP """
    print "Modeling %s..." % names[4:]
    q = bib[:, 4:]

    wi, x = dmp_obj.get_model_q(t, q)

    q0 = np.array([ini_pose[names[4]], ini_pose[names[5]], ini_pose[names[6]], ini_pose[names[7]]]).reshape((1, 4)) \
        if args.iflag else q[0, :].reshape((1, 4))
    eq = 2 * qlog(qprod(q[-1, :], qconj(q)))
    deq0 = np.diff(eq[0:2, :], axis=0) / dt
    qg = np.array([end_pose[names[4]], end_pose[names[5]], end_pose[names[6]], end_pose[names[7]]]) \
        if args.gflag else q[-1, :]
    tn, qf = dmp_obj.fit_model_q(t, tau, qg, q0, deq0, wi)

    """ Plot """
    plt.plot(tn, qf[:, 4:8])
    plt.plot(t, eq, '--')
    plt.show()

    plt.plot(tn, qf[:, 0:4])
    plt.plot(t, q, '--')
    # plt.plot(t, np.abs(y - yf[:, 0:3]), ':')
    plt.show()

    bib2 = np.empty((len(tn), bib.shape[1]))
    bib2[:, 0] = tn
    bib2[:, 4:] = qf[:, 0:4]
    bib2[:, 1:4] = yf[:, 0:3]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(bib2[:, 1], bib2[:, 2], bib2[:, 3])
    ax.plot(bib[:, 1], bib[:, 2], bib[:, 3], '--')
    plt.show()

    _filename = "%s_dmp" % args.file
    with open(_filename, 'w') as f:
        f.write(','.join([str(x) for x in names]) + '\n')
        for i in range(len(tn)):
            f.write(','.join([("%.12f" % x) for x in bib2[i]]) + '\n')
    print "DMP model done"


if __name__ == '__main__':
    main()
