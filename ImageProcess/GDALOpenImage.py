# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/GDALOpenImage.py --image_path /folder/mypic.png --outfile_path_b1 /export/mychoppedimages/ --outfile_path_b2 /export/mychoppedimages/ --outfile_path_b3 /export/mychoppedimages/ --outfile_path_b4 /export/mychoppedimages/ --outfile_path_b5 /export/mychoppedimages/

# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
import math
from PIL import Image
from osgeo import gdal

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path", required=True, help="image path")
ap.add_argument("-o", "--outfile_path_b1", required=True, help="where output image will be saved for band 1")
ap.add_argument("-j", "--outfile_path_b2", required=True, help="where output image will be saved for band 2")
ap.add_argument("-k", "--outfile_path_b3", required=True, help="where output image will be saved for band 3")
ap.add_argument("-l", "--outfile_path_b4", required=True, help="where output image will be saved for band 4")
ap.add_argument("-m", "--outfile_path_b5", required=True, help="where output image will be saved for band 5")
args = vars(ap.parse_args())

input_image = args["image_path"]
outfile_path_b1 = args["outfile_path_b1"]
outfile_path_b2 = args["outfile_path_b2"]
outfile_path_b3 = args["outfile_path_b3"]
outfile_path_b4 = args["outfile_path_b4"]
outfile_path_b5 = args["outfile_path_b5"]

print(input_image)
dataset = gdal.Open(input_image, gdal.GA_ReadOnly)
print(dataset.GetMetadata())

band1 = dataset.GetRasterBand(1).ReadAsArray() * 255
band2 = dataset.GetRasterBand(2).ReadAsArray() * 255
band3 = dataset.GetRasterBand(3).ReadAsArray() * 255
band4 = dataset.GetRasterBand(4).ReadAsArray() * 255
band5 = dataset.GetRasterBand(5).ReadAsArray() * 255

#cv2.imshow("Result", band1.ReadAsArray())
#cv2.waitKey(0)

cv2.imwrite(outfile_path_b1, band1)
cv2.imwrite(outfile_path_b2, band2)
cv2.imwrite(outfile_path_b3, band3)
cv2.imwrite(outfile_path_b4, band4)
cv2.imwrite(outfile_path_b5, band5)
