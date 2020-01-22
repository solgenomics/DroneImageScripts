# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/GetLargestContour.py --image_url https://myserver.com/image/myimage.png --outfile_path /export/mychoppedimages/outimage.png --results_outfile_path /file/myresults.csv

# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
import math
from ImageContours.ShapeDetection.ShapeDetector import ShapeDetector
from ImageCropping.CropPolygons.CropPolygonsToSingleImage import CropPolygonsToSingleImage
import csv

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path", required=True, help="url path the link")
ap.add_argument("-o", "--outfile_path", required=True, help="file path directory where the output will be saved")
ap.add_argument("-r", "--results_outfile_path", required=False, help="file path where results will be saved")
args = vars(ap.parse_args())

input_image_path = args["image_path"]
outfile_path = args["outfile_path"]
results_outfile_path = args["results_outfile_path"]

image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)

if (len(image.shape) == 2):
    empty_mat = np.ones(image.shape, dtype=image.dtype) * 0
    image = cv2.merge((image, empty_mat, empty_mat))

# BGR
lower = [180, 180, 180]
upper = [255, 255, 255]

# create NumPy arrays from the boundaries
lower = np.array(lower, dtype="uint8")
upper = np.array(upper, dtype="uint8")

mask = cv2.inRange(image, lower, upper)
output = cv2.bitwise_and(image, image, mask=mask)

ret,thresh = cv2.threshold(mask, 40, 255, 0)
#dilated=cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))

if imutils.is_cv3():
    im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
else:
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#print(hierarchy)
ratio = 1
sd = ShapeDetector()

c = max(contours, key = cv2.contourArea)

x,y,w,h = cv2.boundingRect(c)
hull = [cv2.convexHull(c)]
cv2.drawContours(output, hull, -1, (0,255,0), 2)
largestcontourarea = cv2.contourArea(hull[0])

polygon = []
for p in hull[0]:
    polygon.append({'x':p[0][0], 'y':p[0][1]})
#cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)

sd = CropPolygonsToSingleImage()
finalImage = sd.crop(output, [polygon])
grayimage = cv2.cvtColor(finalImage,cv2.COLOR_BGR2GRAY)
nonzeropixelcount = cv2.countNonZero(grayimage)
print(nonzeropixelcount)

for i in range(len(contours)):
    c = contours[i]
    h = hierarchy[0]
    # M = cv2.moments(c)
    # cX = int((M["m10"] / M["m00"]) * ratio)
    # cY = int((M["m01"] / M["m00"]) * ratio)
    # shape = sd.detect(c)
    # print(shape)
    #
    # if shape == 'circle':
    # print(h[i])
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    cv2.drawContours(output, contours, i, (255, 255, 0), 1)
    # cv2.imshow("Img"+str(i), image)
    # cv2.waitKey(0)

cv2.imwrite(outfile_path, finalImage)

result_file_lines = [
    [str(round(((1 - nonzeropixelcount/largestcontourarea)*100),2))+'%']
]

if results_outfile_path is not None:
    with open(results_outfile_path, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(result_file_lines)

    writeFile.close()
