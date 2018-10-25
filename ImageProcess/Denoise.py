# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/Denoise.py --image_path /folder/mypic.png --outfile_path /export/mychoppedimages/outimage.png

# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
import urllib.request
import math
from matplotlib import pyplot as plt

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path", required=True, help="image path")
ap.add_argument("-o", "--outfile_path", required=True, help="file path directory where the output will be saved")
args = vars(ap.parse_args())

input_image = args["image_path"]
outfile_path = args["outfile_path"]

img = cv2.imread(input_image)
b,g,r = cv2.split(img)           # get b,g,r
rgb_img = cv2.merge([r,g,b])     # switch it to rgb

# Denoising
dst = cv2.fastNlMeansDenoising(img,None,7,7,21)
#dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

#cv2.imshow("Result", dst)
cv2.imwrite(outfile_path, dst)

# b,g,r = cv2.split(dst)           # get b,g,r
# rgb_dst = cv2.merge([r,g,b])     # switch it to rgb
# 
# plt.subplot(211),plt.imshow(rgb_img)
# plt.subplot(212),plt.imshow(rgb_dst)
# plt.show()

cv2.waitKey(0)

