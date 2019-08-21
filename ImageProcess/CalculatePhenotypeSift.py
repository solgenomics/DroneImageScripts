# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/CalculatePhenotypeSift.py --image_paths /folder/mypic1.png,/folder/mypic2.png --outfile_paths /export/mychoppedimages/outimage2.png,/export/mychoppedimages/outimage2.png

# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import csv

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_paths", required=True, help="image paths comma separated")
ap.add_argument("-o", "--outfile_paths", required=True, help="file path directory where the output images will be saved")
ap.add_argument("-r", "--results_outfile_path", required=True, help="file path where results will be saved")
args = vars(ap.parse_args())

input_images = args["image_paths"]
outfile_paths = args["outfile_paths"]
results_outfile_path = args["results_outfile_path"]
images = input_images.split(",")
outfiles = outfile_paths.split(",")

count = 0
result_file_lines = []
for image in images:
    img = cv2.imread(image)
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(img, None)
    print(features)
    result_file_lines.append([len(kps)])

    kpsimage = cv2.drawKeypoints(img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imshow('image'+str(count),kpsimage)
    cv2.imwrite(outfiles[count], kpsimage)
    
    count += 1

#cv2.waitKey(0)

if results_outfile_path is not None:
    with open(results_outfile_path, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(result_file_lines)

    writeFile.close()
