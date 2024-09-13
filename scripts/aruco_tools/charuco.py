#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2, pathlib, cv2.aruco as aruco, numpy as np

def calibrate_charuco(dirpath, image_format, marker_length, square_length):
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)
    board = aruco.CharucoBoard_create(5, 7, square_length, marker_length, aruco_dict)
    arucoParams = aruco.DetectorParameters_create()
    counter, corners_list, id_list = [],[],[]
    img_dir = pathlib.Path(dirpath)
    first = 0    
    for img in img_dir.glob('*'+str(image_format)):
        print('using image '+str(img))
        image = cv2.imread(str(img))
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(img_gray, aruco_dict, parameters=arucoParams)
        resp, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(markerCorners=corners, markerIds=ids, image=img_gray, board=board)
        if resp > 20:
            corners_list.append(charuco_corners)
            id_list.append(charuco_ids)
    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(charucoCorners=corners_list, charucoIds=id_list, board=board, imageSize=img_gray.shape, cameraMatrix=None, distCoeffs=None)
    return [ret, mtx, dist, rvecs, tvecs]
