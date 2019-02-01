# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/VegetativeIndex/VARI.py --image_path /folder/mypic.png --outfile_path /export/mychoppedimages/tgi.png

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
cv2.imshow("img", img)
b,g,r = cv2.split(img)

numerator = g - r
denominator = g + r - b
cv2.imshow("numerator", numerator)
cv2.imshow("denominator", denominator)
tgi = np.divide(numerator, denominator)
tgi[np.isnan(tgi)] = 0
#tgi = tgi.astype(int)
cv2.imshow("tgi_o", tgi)
cv2.imshow("tgi0inv", cv2.bitwise_not(tgi))
max_arr = np.ones_like(tgi) * 255
cv2.imshow("255", max_arr)
print(max_arr)
print(tgi)
cv2.imshow("tgi1inv", cv2.bitwise_not(tgi))
tgi = max_arr - tgi
print(tgi)
cv2.imshow("numinv", cv2.bitwise_not(numerator))
cv2.imshow("tgi2inv", cv2.bitwise_not(tgi))
cv2.imshow("Result", tgi)
cv2.imwrite(outfile_path, tgi)
cv2.waitKey(0)
