#!/usr/bin/env python

import argparse, rospy, baxter_interface, numpy as np, matplotlib.pyplot as plt
from baxter_interface import CHECK_VERSION
from baxter_interface.digital_io import DigitalIO
from baxter_interface.limb import Limb
from baxter_core_msgs.msg import AssemblyState
from custom_tools import dmp

global flag, endpose
flag = False

def state(flag):
    while rospy.wait_for_message('/robot/state', AssemblyState).enabled <> flag:
        rs = baxter_interface.RobotEnable(CHECK_VERSION)
        if flag:
            rs.enable()
        else:
            rs.disable()

def update_state(*args):
    if args[0] == True:
        global flag, endpose, side
        l = Limb(side)
        endpose = l.joint_angles()
        flag = args[0]
        print "Button pressed"

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
    left_command = dict((key, command[key]) for key in command.keys()
                        if key[:-2] == 'left_')
    right_command = dict((key, command[key]) for key in command.keys()
                         if key[:-2] == 'right_')
    return (command, left_command, right_command, line)

def map_file(filename):
    print("Playing back: %s" % (filename,))
    with open(filename, 'r') as f:
        lines = f.readlines()
    keys = lines[0].rstrip().split(',')
    i = 0
    print("Starting...")
    _cmd, lcmd_start, rcmd_start, _raw = clean_line(lines[1], keys)
    bib = np.empty([len(lines[1:]),len(_raw)])
    for line in lines[1:]:
        i += 1
        cmd, lcmd, rcmd, values = clean_line(line, keys)
        bib[i-1,:] = values
    print("Finished")
    return keys, bib

def main():
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt)
    parser.add_argument('-f', '--file', metavar='PATH', required=True, help='path to input file')
    args = parser.parse_args(rospy.myargv()[1:])

    print("Initializing node... ")
    rospy.init_node("joint_position_dmp_modeling")
    print("Getting robot state... ")
    rs = baxter_interface.RobotEnable(CHECK_VERSION)
    print rs.state().enabled

    names, bib = map_file(args.file)
    t = bib[:,0]
    bib2 = np.empty(bib.shape)
    bib2[:,0] = t
    alphay = 4; betay = alphay/4; alphax = alphay/3
    n = 50

    global side
    side = None
    if names[1].find('left') > -1 and names[1].find('left') < 6:
        side = 'left'
    elif names[1].find('right') > -1 and names[1].find('right') < 6:
        side = 'right'

    if not side == None:
        button = DigitalIO("torso_%s_button_ok" % side)
        button.state_changed.connect(update_state)
        state(True)
        print "Move robot to end pose and press button"
        while flag == False:
            pass

        for i in range(len(bib[0,1:])):
            y = bib[:,i+1]
            fn, x, dt = dmp.get_model(t, y, n, alphay, betay, alphax)

            if names[i+1] <> "%s_gripper" % side:
                g = endpose[names[i+1]]
            else:
                g = y[len(y)-1]
            yp = y[0]; dyp = 0; y0 = yp
            yf = np.empty([len(y), 3])
            for j in range(len(t)):
                ddyp = alphay*(betay*(g-yp)-dyp) + (fn[j]*x[j]*(g-y0))
                dyp = dyp + ddyp*dt
                yp = yp + dyp*dt
                yf[j] = [yp, dyp, ddyp]
            
            plt.plot(t, yf[:, 0])
            plt.plot(t, y, '--')
            plt.plot(t, np.abs(y-yf[:, 0]), ':')
            plt.show()
            
            bib2[:,i+1] = yf[:, 0]

        _filename = "dmp_%s" % args.file
        with open(_filename, 'w') as f:
            f.write(','.join([str(x) for x in names]) + '\n')
            for i in range(len(t)):            
                f.write(','.join([str(x) for x in bib2[i]]) + '\n')
        state(False)
    else:
        print "Error: There is an issue with the trajectory file"

if __name__ == '__main__':
    main()
