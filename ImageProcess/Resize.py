# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/Resize.py --image_path /folder/mypic.png --outfile_path /export/mychoppedimages/outimage.png --width 12000

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
ap.add_argument("-w", "--width", required=False, help="the width to resize to. will automatically rescale height if no height given")
ap.add_argument("-x", "--height", required=False, help="the height to resize to. will automatically rescale width if no height given")
args = vars(ap.parse_args())

conv = lambda i : i or None

input_image = args["image_path"]
outfile_path = args["outfile_path"]
width = conv(args["width"])
height = conv(args["height"])

if width is not None:
    width = int(width)

if height is not None:
    height = int(height)

img = cv2.imread(input_image)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is not None and height is not None:
        dim = (width, height)
    elif width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation = inter)

    return resized

resized = image_resize(img, width = width, height = height)

cv2.imwrite(outfile_path, resized)
#cv2.imshow("Result", dst)
#cv2.waitKey(0)
