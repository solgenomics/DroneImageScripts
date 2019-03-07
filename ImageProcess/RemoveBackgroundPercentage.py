# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/RemoveBackgroundPercentage.py --image_path /folder/mypic.png --outfile_path /export/mychoppedimages/backgroundremoved.png --lower_percentage 20 --upper_percentage 20

# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
import math


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path", required=True, help="image path for image with NIR as first channel and Red as second channel.")
ap.add_argument("-l", "--lower_percentage", required=True, help="percentage of pixels in lower bound to remove.")
ap.add_argument("-u", "--upper_percentage", required=True, help="percentage of pixels in upper bound to remove.")
ap.add_argument("-o", "--outfile_path", required=True, help="file path where the output will be saved")
args = vars(ap.parse_args())

image_path = args["image_path"]
outfile_path = args["outfile_path"]
lower_percentage = float(args["lower_percentage"])/100
upper_percentage = float(args["upper_percentage"])/100

histSize = [256]
input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

hist = cv2.calcHist([input_image],[0],None,histSize,[0,255])

#print(hist)
#print(input_image.shape)

total_pixels = input_image.shape[0] * input_image.shape[1]
summing = 0
drone_imagery_remove_background_lower_percentage_threshold = 0
drone_imagery_remove_background_upper_percentage_threshold = 0
for i in range(0, histSize[0]):
    binVal = hist[i][0]
    summing = summing + binVal
    percentage = summing / total_pixels
    if percentage >= lower_percentage:
        drone_imagery_remove_background_lower_percentage_threshold = i
        break

summing = 0;
for i in range(0, histSize[0]):
    binVal = hist[i][0]
    summing = summing + binVal
    percentage = summing / total_pixels
    if percentage >= 1-upper_percentage:
        drone_imagery_remove_background_upper_percentage_threshold = i
        break

lower_thresh = int(float(drone_imagery_remove_background_lower_percentage_threshold))
upper_thresh = int(float(drone_imagery_remove_background_upper_percentage_threshold))
th, dst = cv2.threshold(input_image, lower_thresh, upper_thresh, cv2.THRESH_TOZERO)

cv2.imwrite(outfile_path, dst)
#cv2.waitKey(0)
