# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageContours/GetContours.py --image_url https://myserver.com/image/myimage.png --outfile_path /export/mychoppedimages/outimage.png

# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
import urllib.request
import math
from ImageContours.ShapeDetection.ShapeDetector import ShapeDetector
from ImageCropping.CropPolygons.CropPolygonsToSingleImage import CropPolygonsToSingleImage
import csv

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_url", required=True, help="url path the link")
ap.add_argument("-o", "--outfile_path", required=True, help="file path directory where the output will be saved")
ap.add_argument("-r", "--results_outfile_path", required=False, help="file path where results will be saved")
args = vars(ap.parse_args())

input_image_url = args["image_url"]
outfile_path = args["outfile_path"]
results_outfile_path = args["results_outfile_path"]

with urllib.request.urlopen(input_image_url) as url_response:
    image = np.asarray(bytearray(url_response.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    #image = cv2.resize(image,(480, 320))

imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
imgray = cv2.blur(imgray,(15,15))

#ret,thresh = cv2.threshold(imgray, math.floor(np.average(imgray)), 255, cv2.THRESH_BINARY_INV)
#thresh = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
thresh = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#cv2.imshow("Threshold", thresh)

#dilated=cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)))
dilated=cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)))
#cv2.imshow("Dilated", dilated)

if imutils.is_cv3():
    im2, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
else:
    contours, hierarchy = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

print(hierarchy)
ratio = 1
sd = ShapeDetector()

for i in range(len(contours)):
    c = contours[i]
    h = hierarchy[0]
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    shape = sd.detect(c)
    print(shape)
 
    if shape == 'circle':
        print(h[i])
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        cv2.drawContours(image, contours, i, (0, 255, 0))
        # cv2.imshow("Img"+str(i), image)
        # cv2.waitKey(0)

cv2.imwrite(outfile_path, image)

result_file_lines = [
    [len(contours)]
]

if results_outfile_path is not None:
    with open(results_outfile_path, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(result_file_lines)

    writeFile.close()
