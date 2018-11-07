# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/RemoveBackground.py --image_path /folder/mypic.png --outfile_path /export/mychoppedimages/outimage.png

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
ap.add_argument("-t", "--threshold", required=True, help="threshold value to remove from image")
args = vars(ap.parse_args())

input_image = args["image_path"]
outfile_path = args["outfile_path"]
thresh = args["threshold"]

src = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)

th, dst = cv2.threshold(src, int(float(thresh)), 255, cv2.THRESH_TOZERO)

#cv2.imshow("Result", dst)
cv2.imwrite(outfile_path, dst)
#cv2.waitKey(0)
