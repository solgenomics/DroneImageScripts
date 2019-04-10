# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/Denoise.py --image_path /folder/mypic.png --outfile_path /export/mychoppedimages/outimage.png

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
args = vars(ap.parse_args())

input_image = args["image_path"]
outfile_path = args["outfile_path"]

img = cv2.imread(input_image)

# Denoising
dst = cv2.fastNlMeansDenoising(img,None,1,7,21)
#dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

cv2.imwrite(outfile_path, dst)
#cv2.imshow("Result", dst)
#cv2.waitKey(0)
