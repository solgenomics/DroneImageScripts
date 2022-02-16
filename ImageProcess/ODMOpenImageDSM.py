# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/ODMOpenImageDSM.py --image_path_dsm /folder/dsm.tif --image_path_dtm /folder/dtm.tif --outfile_path_dsm /export/dsm.png --outfile_path_dtm /export/dtm.png --outfile_path_subtract /export/subtract.png --band_number 1

# import the necessary packages
import argparse
import numpy as np
from osgeo import gdal
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path_dsm", required=True, help="image path DSM")
ap.add_argument("-j", "--image_path_dtm", required=True, help="image path DTM")
ap.add_argument("-o", "--outfile_path_dsm", required=True, help="where output PNG image of DSM will be saved")
ap.add_argument("-p", "--outfile_path_dtm", required=True, help="where output PNG image of DTM will be saved")
ap.add_argument("-s", "--outfile_path_subtract", required=True, help="where output PNG image of DSM minus DTM will be saved")
ap.add_argument("-b", "--band_number", required=True, help="band number to save")
args = vars(ap.parse_args())

input_image_dsm = args["image_path_dsm"]
input_image_dtm = args["image_path_dtm"]
outfile_path_dsm = args["outfile_path_dsm"]
outfile_path_dtm = args["outfile_path_dtm"]
outfile_path_subtract = args["outfile_path_subtract"]
band_number = args["band_number"]

options_list = [
    '-ot Byte',
    '-of PNG',
    '-b '+band_number,
    '-scale'
]

options_string = " ".join(options_list)

gdal.Translate(
    outfile_path_dsm,
    input_image_dsm,
    options=options_string
)

gdal.Translate(
    outfile_path_dtm,
    input_image_dtm,
    options=options_string
)

img_dtm = cv2.imread(outfile_path_dtm)
img_dsm = cv2.imread(outfile_path_dsm)
img_substract = img_dtm - img_dsm
cv2.imwrite(outfile_path_subtract, img_substract*(0.05*255))
