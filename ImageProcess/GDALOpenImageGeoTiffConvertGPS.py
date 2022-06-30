# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/GDALOpenImageGeoTiffConvertGPS.py --image_path $file --outfile_path_image $outfile_image --outfile_path_geo_params $outfile_geoparams

#tif = gdal.Open(my_tiff)
#gt = tif.GetGeotransform()
#x_min = gt[0]
#x_size = gt[1]
#y_min = gt[3]
#y_size = gt[5]
#mx, my = 500, 600  #coord in map units, as in question
#px = mx * x_size + x_min #x pixel
#py = my * y_size + y_min #y pixel

# import the necessary packages
import argparse
import numpy as np
from osgeo import gdal
import cv2
from osgeo import osr
import osgeo
import csv
import pandas as pd

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path", required=True, help="image path GeoTiff")
ap.add_argument("-j", "--input_coords_path", required=True, help="file where input coords are (e.g. [312290.6173, 1567248.5259])")
ap.add_argument("-k", "--output_coords_path", required=True, help="file where output coords are (e.g. [14.170542407723804, 121.2603596771053])")
args = vars(ap.parse_args())

input_image = args["image_path"]
input_coords_path = args["input_coords_path"]
output_coords_path = args["output_coords_path"]

def ReprojectCoords(coords, src_srs, tgt_srs):
    """ Reproject a list of x,y coordinates. """
    trans_coords = []
    transform = osr.CoordinateTransformation(src_srs, tgt_srs)
    for plot_name,x,y in coords:
        x,y,z = transform.TransformPoint(x,y)
        trans_coords.append([plot_name, x, y])
    return trans_coords

driver = gdal.GetDriverByName('GTiff')
dataset = gdal.Open(input_image)
transform = dataset.GetGeoTransform()
projection = str(dataset.GetProjection())
print(transform)
print(projection)

src_srs = osr.SpatialReference()
src_srs.ImportFromWkt(dataset.GetProjection())
#tgt_srs = osr.SpatialReference()
#tgt_srs.ImportFromEPSG(4326)
tgt_srs = src_srs.CloneGeogCS()

input_coords_data = pd.read_csv(input_coords_path, sep="\t", header=None)

input_coords = []
for index, row in input_coords_data.iterrows():
    plot_name = row[0]
    point_x = row[1]
    point_y = row[2]
    input_coords.append([plot_name, point_x, point_y])

geo_coords = ReprojectCoords(input_coords, src_srs, tgt_srs)
# print(geo_coords)

with open(output_coords_path, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(geo_coords)
writeFile.close()
