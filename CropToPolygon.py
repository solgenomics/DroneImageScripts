# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageCropping/CropToPolygon.py --inputfile_path /export/archive/mystitchedimage.png --outputfile_path /export/mychoppedimages/polygon.png --polygon_json '[{x:10, y:10}, {x:15, y:20}, {x:1, y:10}, {x:1, y:25}]'

# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
import json
from ImageCropping.CropPolygons.CropPolygonsToSingleImage import CropPolygonsToSingleImage

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--inputfile_path", required=True, help="complete file path to the image you want to crop to a polygon")
ap.add_argument("-o", "--outputfile_path", required=True, help="complete file path to where to cropped polygon will be saved")
ap.add_argument("-p", "--polygon_json", required=True, help="json string that is an array of length 4, with the x,y coordinates of the polygon")
args = vars(ap.parse_args())

inputfile_path = args["inputfile_path"]
outputfile_path = args["outputfile_path"]
polygon_json = args["polygon_json"]
polygons = json.loads(polygon_json)
print(polygons)

input_image = cv2.imread(inputfile_path)

sd = CropPolygonsToSingleImage()
finalImage = sd.crop(input_image, polygons)

cv2.imwrite(outputfile_path, finalImage)
#cv2.imshow("Result", finalImage)
#cv2.waitKey(0)
