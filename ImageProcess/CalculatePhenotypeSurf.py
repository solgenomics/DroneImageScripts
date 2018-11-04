# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/CalculatePhenotypeSurf.py --image_paths /folder/mypic1.png,/folder/mypic2.png --outfile_paths /export/mychoppedimages/outimage2.png,/export/mychoppedimages/outimage2.png

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
ap.add_argument("-i", "--image_paths", required=True, help="image path")
ap.add_argument("-o", "--outfile_paths", required=True, help="file path directory where the output will be saved")
ap.add_argument("-r", "--results_outfile_path", required=True, help="file path where results will be saved")
args = vars(ap.parse_args())

input_images = args["image_paths"]
outfile_paths = args["outfile_paths"]
results_ourfile = args["results_outfile_path"]
images = input_images.split(",")
outfiles = outfile_paths.split(",")

count = 0
for image in images:
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    orb = cv2.xfeatures2d.SURF_create()
    kp, des = orb.detectAndCompute(img, None)
    print(kp)
    #print(des)

    kpsimage = cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imshow('image'+str(count),kpsimage)
    cv2.imwrite(outfiles[count], kpsimage)
    
    count += 1

#cv2.waitKey(0)

