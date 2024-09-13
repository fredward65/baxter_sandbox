#!/usr/bin/env python

from charuco import calibrate_charuco
from utils import load_coefficients, save_coefficients
import cv2

IMAGES_DIR = './src/baxter_sandbox/scripts/aruco_tools/'
IMAGES_FORMAT = 'jpg'
MARKER_LENGTH = 0.27
SQUARE_LENGTH = 0.32

ret, mtx, dist, rvecs, tvecs = calibrate_charuco(IMAGES_DIR, IMAGES_FORMAT, MARKER_LENGTH, SQUARE_LENGTH)

save_coefficients(mtx, dist, IMAGES_DIR+'calibration_charuco.yml')
print "Done! ", mtx, dist
