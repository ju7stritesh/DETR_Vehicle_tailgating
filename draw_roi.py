import xlrd
import ast
import json
import operator
import re
import math

import cv2
import numpy as np
import copy

drawing = False     # true if mouse is pressed
mode = True         # if True, draw rectangle.
ix, iy = -1, -1
drawn_box = []
img = []

# mouse callback function
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode, drawn_box, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode != True:
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
            drawn_box.append([iy,ix,y,x])
            cv2.imwrite("test5.jpg", img)

        else:
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

def draw_roi(img1):
    global mode, drawn_box, img
    drawn_box = []
    img = img1
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)


    while (1):
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == 113:
            break

    cv2.destroyAllWindows()
    return drawn_box
