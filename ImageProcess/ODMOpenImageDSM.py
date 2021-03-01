# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/ODMOpenImageDSM.py --image_path /folder/mypic.tif --outfile_path /export/b1.png

# import the necessary packages
import argparse
import numpy as np
from osgeo import gdal

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path", required=True, help="image path")
ap.add_argument("-o", "--outfile_path", required=True, help="where output PNG image will be saved")
args = vars(ap.parse_args())

input_image = args["image_path"]
outfile_path = args["outfile_path"]

options_list = [
    '-ot Byte',
    '-of PNG',
    '-b 1',
    '-scale'
]

options_string = " ".join(options_list)
    
gdal.Translate(
    outfile_path,
    input_image,
    options=options_string
)
