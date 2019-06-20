# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/VegetativeIndex/VARI.py --image_path /folder/mypic.png --outfile_path /export/myimages/tgi.png

# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
import math

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path", required=True, help="image path")
ap.add_argument("-o", "--outfile_path", required=True, help="file path where the output will be saved")
args = vars(ap.parse_args())

input_image = args["image_path"]
outfile_path = args["outfile_path"]

img = cv2.imread(input_image)
#cv2.imshow("img", img)
b,g,r = cv2.split(img)

numerator = g - r
denominator = g + r - b
#print(img.dtype)
#print(numerator.dtype)
#print(denominator.dtype)
#cv2.imshow("numerator", numerator)
#cv2.imshow("denominator", denominator)
vari = np.divide(numerator, denominator)
vari[np.isnan(vari)] = 0

#print(vari.shape)
#print(vari.dtype)

vari = vari * 255
vari = vari.astype(np.uint8)

#print(vari.shape)
#print(vari.dtype)
#cv2.imshow("Result", vari)

cv2.imwrite(outfile_path, vari)
#cv2.waitKey(0)
