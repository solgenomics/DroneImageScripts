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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_url", required=True, help="url path the link")
ap.add_argument("-o", "--outfile_path", required=True, help="file path directory where the output will be saved")
args = vars(ap.parse_args())

input_image_url = args["image_url"]
outfile_path = args["outfile_path"]

with urllib.request.urlopen(input_image_url) as url_response:
    image = np.asarray(bytearray(url_response.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    #image = cv2.resize(image,(480, 320))

imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
imgray = cv2.blur(imgray,(15,15))

#ret,thresh = cv2.threshold(imgray, math.floor(np.average(imgray)), 255, cv2.THRESH_BINARY_INV)
thresh = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
#thresh = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#cv2.imshow("Threshold", thresh)

dilated=cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)))
#dilated=cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)))
#cv2.imshow("Dilated", dilated)

im2, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

ratio = 1
sd = ShapeDetector()

for c in contours:
    # compute the center of the contour, then detect the name of the
    # shape using only the contour
    #M = cv2.moments(c)
    #cX = int((M["m10"] / M["m00"]) * ratio)
    #cY = int((M["m01"] / M["m00"]) * ratio)
    #shape = sd.detect(c)
 
    # multiply the contour (x, y)-coordinates by the resize ratio,
    # then draw the contours and the name of the shape on the image
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    #cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# create hull array for convex hull points
hull = []
# calculate points for each contour
for i in range(len(contours)):
    # creating convex hull object for each contour
    hull.append(cv2.convexHull(contours[i], False))

# create an empty black image
drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
 
# draw contours and hull points
for i in range(len(contours)):
    color_contours = (0, 255, 0) # green - color for contours
    color = (255, 0, 0) # blue - color for convex hull
    # draw ith contour
    cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
    # draw ith convex hull object
    cv2.drawContours(drawing, hull, i, color, 1, 8)

# MAYBE USE CROPTOPOLYGON TO REMOVE ALL NON CONTOUR AREAS FROM IMAGE

#polygons = []
#for i in contours:
#    polygon = []
#    for contour_block in i:
#        polygon.append({'x':contour_block[0][0], 'y':contour_block[0][1]})
#    polygons.append(polygon)

#sd = CropPolygonsToSingleImage()
#finalImage = sd.crop(image, polygons)
#cv2.imshow("Cut", finalImage)

#cv2.imshow("Image", drawing)

cv2.imwrite(outfile_path, dilated)
cv2.waitKey(0)

