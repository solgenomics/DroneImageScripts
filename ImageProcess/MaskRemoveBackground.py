# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/MaskRemoveBackground.py --image_path /folder/mypic.png --mask_image_path /folder/mymaskpic.png --outfile_path /export/mychoppedimages/outimage.png

# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
import math

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path", required=True, help="image path")
ap.add_argument("-m", "--mask_image_path", required=True, help="image path for masked image. masked image should be black everywhere except where the input image will show.")
ap.add_argument("-o", "--outfile_path", required=True, help="file path directory where the output will be saved")
args = vars(ap.parse_args())

input_image = args["image_path"]
mask_input_image = args["mask_image_path"]
outfile_path = args["outfile_path"]

src = cv2.imread(input_image)
mask = cv2.imread(mask_input_image, 0)

#print(mask.shape)
#print(src.shape)

#cv2.imshow("src", src)
#cv2.imshow("Mask", mask)

dst = cv2.bitwise_or(src, src, mask=mask)

cv2.imwrite(outfile_path, dst)
#cv2.imshow("Result", dst)
#cv2.waitKey(0)
