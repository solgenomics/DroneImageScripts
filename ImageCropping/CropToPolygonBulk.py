# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/CropToPolygonBulk.py --inputfile_path /export/archive/input.csv

# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
import json
import csv
import pandas as pd
import CropPolygons.CropPolygonsToSingleImage as CropPolygonsToSingleImage
import CropPolygonsSquareRectangles.CropPolygonsToSingleSquareRectangularImage as CropPolygonsToSingleSquareRectangularImage

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--inputfile_path", required=True, help="complete file path to the image you want to crop to a polygon")
args = vars(ap.parse_args())

inputfile_path = args["inputfile_path"]

input_image_file_data = pd.read_csv(inputfile_path, sep="\t", header=None)

for index, row in input_image_file_data.iterrows():
    inputfile_path = row[0]
    outputfile_path = row[1]
    polygon_json = row[2]
    polygon_type = row[3]
    image_band_index = row[4]
    polygons = json.loads(polygon_json)

    img = cv2.imread(inputfile_path, cv2.IMREAD_UNCHANGED)
    img_shape = img.shape

    if len(img_shape) == 3:
        if img_shape[2] == 3:
            b,g,r = cv2.split(img)
            if image_band_index is not None and not np.isnan(image_band_index):
                image_band_index = int(image_band_index)
                if image_band_index == 0:
                    img = b
                if image_band_index == 1:
                    img = g
                if image_band_index == 2:
                    img = r
    if polygon_type == 'rectangular_square':
        sd = CropPolygonsToSingleSquareRectangularImage.CropPolygonsToSingleSquareRectangularImage()
        finalImage = sd.crop(img, polygons)
    elif polygon_type == 'rectangular_polygon':
        sd = CropPolygonsToSingleImage.CropPolygonsToSingleImage()
        finalImage = sd.crop(img, polygons)

    cv2.imwrite(outputfile_path, finalImage)
    #cv2.imshow("Result", finalImage)
    #cv2.waitKey(0)
