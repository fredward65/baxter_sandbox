#!/usr/bin/env python

import cv2
import numpy as np


def draw_eyes(img, t_mrk=[0, 0, 1]):
    size = 120
    poslx = int(img.shape[1]*(0.3 - np.exp(-1.25*t_mrk[2])*(t_mrk[0] - 0.05)))
    posly = int(img.shape[0]*(0.5 + np.exp(-1.25*t_mrk[2])*(t_mrk[1])))
    posrx = int(img.shape[1]*(0.7 - np.exp(-1.25*t_mrk[2])*(t_mrk[0] + 0.05)))
    posry = int(img.shape[0]*(0.5 + np.exp(-1.25*t_mrk[2])*(t_mrk[1])))
    posl1 = (poslx, posly)
    posr1 = (posrx, posry)
    posl2 = (int(poslx + 1*size*(poslx - img.shape[1]*0.3)/img.shape[1]),
             int(posly + 2*size*(posly - img.shape[0]*0.5)/img.shape[0]))
    posr2 = (int(posrx + 1*size*(posrx - img.shape[1]*0.7)/img.shape[1]),
             int(posry + 2*size*(posry - img.shape[0]*0.5)/img.shape[0]))
    img = cv2.circle(img, posl1, size, (0, 0, 0), 10)
    img = cv2.circle(img, posl2, int(size*0.3), (0, 0, 0), -1)
    img = cv2.circle(img, posr1, size, (0, 0, 0), 10)
    img = cv2.circle(img, posr2, int(size*0.3), (0, 0, 0), -1)
    return img


def main():
    pass


if __name__ == "__main__":
    main()
