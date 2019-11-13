# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/GetContour.py --image_url https://myserver.com/image/myimage.png --outfile_path /export/mychoppedimages/outimage.png --results_outfile_path /file/myresults.csv

# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
import math
import ShapeDetection.ShapeDetector as ShapeDetector
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
img_shape = image.shape
if len(img_shape) == 3:
    if img_shape[2] == 3:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
imgray = cv2.blur(image,(15,15))

#ret,thresh = cv2.threshold(imgray, math.floor(np.average(imgray)), 255, cv2.THRESH_BINARY_INV)
#thresh = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
thresh = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)
#cv2.imshow("Threshold", thresh)

#dilated=cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)))
dilated=cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)))
#cv2.imshow("Dilated", dilated)

if imutils.is_cv3():
    im2, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
else:
    contours, hierarchy = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#print(hierarchy)
ratio = 1
sd = ShapeDetector.ShapeDetector()

for i in range(len(contours)):
    c = contours[i]
    h = hierarchy[0]
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    shape = sd.detect(c)
    #print(shape)
 
    if shape == 'circle':
        #print(h[i])
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        cv2.drawContours(image, contours, i, (255, 255, 0))
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
