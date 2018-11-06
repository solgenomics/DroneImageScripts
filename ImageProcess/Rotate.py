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
args = vars(ap.parse_args())

input_image = args["image_path"]
outfile_path = args["outfile_path"]
angle = float(args["angle"])

img = cv2.imread(input_image)
rows,cols,d = img.shape

M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
dst = cv2.warpAffine(img,M,(cols,rows))

#cv2.imshow("Result", dst)
#cv2.waitKey(0)
cv2.imwrite(outfile_path, dst)
