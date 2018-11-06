# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/SplitChannels.py --image_path /folder/mypic.png --outfile_dir /export/mychoppedimages/

# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
import math

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path", required=True, help="image path")
ap.add_argument("-o", "--outfile_dir", required=True, help="file directory where the output (separate r, g, b images) will be saved")
args = vars(ap.parse_args())

input_image = args["image_path"]
outfile_dir = args["outfile_dir"]

img = cv2.imread(input_image)
b,g,r = cv2.split(img)

cv2.imwrite(outfile_dir+'b.png', b)
cv2.imwrite(outfile_dir+'g.png', g)
cv2.imwrite(outfile_dir+'r.png', r)
