#!/usr/bin/env python

import rospy
from baxter_core_msgs.srv import OpenCamera,OpenCameraRequest,OpenCameraResponse

class ExCamera():
	def __init__(self):
		rospy.init_node("camera")
		print "Waiting for service"
		rospy.wait_for_service("/cameras/open")
		print "Service found"
		self.srv_proxy_open_camera = rospy.ServiceProxy("/cameras/open",OpenCamera)
		req = OpenCameraRequest()
		req.name = "head_camera"
		req.settings.width = 1280
		req.settings.height = 800
		req.settings.fps = 10
		response = self.srv_proxy_open_camera(req)
		print "response error:",response.err


		self.srv_open_camera = rospy.Service(
			"/fake_open_camera",
			OpenCamera,
			self.cb_fake_open_camera
		)

		rospy.spin()

	def cb_fake_open_camera(self, request):
		print "request", request
		response = OpenCameraResponse()
		response.err = 666
		return response

if __name__=="__main__":
	ex = ExCamera()
