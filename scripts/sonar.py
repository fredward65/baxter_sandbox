#!/usr/bin/env python

import rospy, baxter_interface, numpy

from std_msgs.msg import UInt16
from sensor_msgs.msg import PointCloud

def callback(msg):
    x = []
    y = []
    var = msg.channels
    l = len(var[0].values)
    for i in range(12):
        a = (i + 1) * numpy.pi/6
        r = 0
        for j in range(l):
            if var[0].values[j] == i:
                r = var[1].values[j]
        x.append(r*numpy.cos(a))
        y.append(r*numpy.sin(a))
        print [i, r]
    
def state(flag):
    val = UInt16()
    if flag:
        val.data = 65535
    else:
        val.data = 0
    rospy.sleep(0.5)
    pub.publish(val)

def sdhook():    
    print "Closing node..."
    state(False)
    

def main():
    print "Starting node..."
    rospy.init_node("sonar_reader", anonymous=True)
    print "Node started"
    global pub
    pub = rospy.Publisher("/robot/sonar/head_sonar/set_sonars_enabled",UInt16,queue_size=3)
    state(True)
    rospy.Subscriber("/robot/sonar/head_sonar/state",PointCloud,callback)
    rospy.on_shutdown(sdhook)
    rospy.spin()
    
if __name__ == '__main__':
    main()
