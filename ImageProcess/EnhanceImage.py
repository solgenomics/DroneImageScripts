# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/EnhanceImage.py --image_path /folder/mypic.png --outfile_path /export/mychoppedimages/

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

rgb = img/255

gaussian_rgb = cv2.GaussianBlur(rgb, (9,9), 10.0)
gaussian_rgb[gaussian_rgb<0] = 0
gaussian_rgb[gaussian_rgb>1] = 1
unsharp_rgb = cv2.addWeighted(rgb, 1.5, gaussian_rgb, -0.5, 0)
unsharp_rgb[unsharp_rgb<0] = 0
unsharp_rgb[unsharp_rgb>1] = 1

# Apply a gamma correction to make the render appear closer to what our eyes would see
gamma = 1.4
gamma_corr_rgb = unsharp_rgb**(1.0/gamma)

cv2.imwrite(outfile_path, gamma_corr_rgb*255)
