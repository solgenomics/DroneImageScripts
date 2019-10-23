# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/CompareTwoImagesPixelValues.py --image_path1 /folder/mypic.png --image_path2 /folder/mypic2.png --outfile_path /export/mychoppedimages/outimage.png

# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path1", required=True, help="image path for first image")
ap.add_argument("-j", "--image_type1", required=True, help="image type for first image")
ap.add_argument("-a", "--image_path2", required=True, help="image path for second image")
ap.add_argument("-k", "--image_type2", required=True, help="image type for second image")
ap.add_argument("-o", "--outfile_path", required=True, help="file path directory where the output will be saved")
args = vars(ap.parse_args())

input_image1 = args["image_path1"]
input_image_type1 = args["image_type1"]
input_image2 = args["image_path2"]
input_image_type2 = args["image_type2"]
outfile_path = args["outfile_path"]

img1 = cv2.imread(input_image1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread(input_image2, cv2.COLOR_BGR2GRAY)

x = []
y = []
for r in img1:
    for c in r:
        x.append(c)
for r in img2:
    for c in r:
        y.append(c)

plt.plot(x,y, 'bo')
plt.xlabel(input_image_type1)
plt.ylabel(input_image_type2)
# plt.show()
plt.savefig(outfile_path)
