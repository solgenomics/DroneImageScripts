# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/Rotate.py --image_path /folder/mypic.png --outfile_path /export/mychoppedimages/outimage.png --angle -0.12

# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
import math

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path", required=True, help="image path")
ap.add_argument("-o", "--outfile_path", required=True, help="file path directory where the output will be saved")
ap.add_argument("-a", "--angle", required=True, help="angle of rotation. positive is counter-clockwise and negative is clockwise")
ap.add_argument("-c", "--centered", required=False, help="whether to rotate around the center")
args = vars(ap.parse_args())

input_image = args["image_path"]
outfile_path = args["outfile_path"]
centered = args["centered"]
angle = float(args["angle"])

img = cv2.imread(input_image)
height, width = img.shape[:2]

image_center = (width/2,height/2)
if centered:
    image_center = (0,0)

rotation_mat = cv2.getRotationMatrix2D(image_center,angle,1)

abs_cos = abs(rotation_mat[0,0]) 
abs_sin = abs(rotation_mat[0,1])

bound_w = int(height * abs_sin + width * abs_cos)
bound_h = int(height * abs_cos + width * abs_sin)

if centered:
    rotation_mat[0, 2] += image_center[0]
    rotation_mat[1, 2] += image_center[1]
else:
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

src = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h))

# tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
# b, g, r = cv2.split(src)
# rgba = [b,g,r, alpha]
# dst = cv2.merge(rgba,4)
# print(dst.shape)
# print(dst.dtype)

#cv2.imshow("Result", dst)
#cv2.waitKey(0)
cv2.imwrite(outfile_path, src)
