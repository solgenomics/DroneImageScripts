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
ap.add_argument("-b", "--image_band_index", required=False, help="if a specific channel should be returned, the band index 0,1,2 can be given")
args = vars(ap.parse_args())

inputfile_path = args["inputfile_path"]
outputfile_path = args["outputfile_path"]
image_band_index = args["image_band_index"]
polygon_json = args["polygon_json"]
polygons = json.loads(polygon_json)
print(polygons)

img = cv2.imread(inputfile_path, cv2.IMREAD_UNCHANGED)
img_shape = img.shape

if len(img_shape) == 3:
    if img_shape[2] == 3:
        b,g,r = cv2.split(img)
        if image_band_index is not None:
            image_band_index = int(image_band_index)
            if image_band_index == 0:
                print('Get Band1')
                img = b
            if image_band_index == 1:
                print('Get Band2')
                img = g
            if image_band_index == 2:
                print('Get Band3')
                img = r

sd = CropPolygonsToSingleImage()
finalImage = sd.crop(img, polygons)

cv2.imwrite(outputfile_path, finalImage)
#cv2.imshow("Result", finalImage)
#cv2.waitKey(0)
