# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/GDALOpenSingleChannelImageGeoTiff.py --image_path $file --outfile_path_image $outfile_image --outfile_path_geo_params $outfile_geoparams

# import the necessary packages
import argparse
import numpy as np
import gdal
import cv2
import osr
import osgeo
import csv

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path", required=True, help="image path DSM")
ap.add_argument("-o", "--outfile_path_image", required=True, help="where output PNG image of RGB will be saved")
ap.add_argument("-g", "--outfile_path_geo_params", required=True, help="where output GeoTiff params will be saved")
ap.add_argument("-p", "--outfile_path_geo_projection", required=True, help="where output GeoTiff projection will be saved")
args = vars(ap.parse_args())

input_image = args["image_path"]
outfile_path_image = args["outfile_path_image"]
outfile_path_geo_params = args["outfile_path_geo_params"]
outfile_path_geo_projection = args["outfile_path_geo_projection"]

driver = gdal.GetDriverByName('GTiff')
dataset = gdal.Open(input_image)
transform = dataset.GetGeoTransform()
projection = str(dataset.GetProjection())
print(transform)
print(projection)

with open(outfile_path_geo_params, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows([transform])
writeFile.close()

with open(outfile_path_geo_projection, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows([projection])
writeFile.close()

options_list = [
    '-ot Byte',
    '-of PNG',
    '-b 1',
    '-scale'
]
options_string = " ".join(options_list)

gdal.Translate(
    outfile_path_image,
    input_image,
    options=options_string
)

band = cv2.imread(outfile_path_image, cv2.IMREAD_UNCHANGED)

# cv2.imshow("Result", band)
# cv2.waitKey(0)
cv2.imwrite(outfile_path_image, band)
