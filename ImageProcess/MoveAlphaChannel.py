# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/MoveAlphaChannel.py --image_path /folder/mypic.png --outfile_path /export/mychoppedimages/

# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
import math
from PIL import Image

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path", required=True, help="image path")
ap.add_argument("-o", "--outfile_path", required=True, help="where output image will be saved")
args = vars(ap.parse_args())

input_image = args["image_path"]
outfile_path = args["outfile_path"]

print(input_image)
img = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)
print(img.shape)
print(img.dtype)

print(img)
print(type(img))
print(img.min())
print(img.max())

img = img*255
img = img.astype(np.uint8)

print(img)
print(type(img))
print(img.min())
print(img.max())

cv2.imwrite(outfile_path, img)
