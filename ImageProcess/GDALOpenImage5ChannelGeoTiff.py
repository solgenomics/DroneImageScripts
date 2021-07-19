# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/GDALOpenImage5ChannelGeoTiff.py --image_path $file --outfile_path_image $outfile_image --outfile_path_image_1 $outfile_image_r --outfile_path_image_2 $outfile_image_g --outfile_path_image_3 $outfile_image_b --outfile_path_image_4 $outfile_image_n --outfile_path_image_5 $outfile_image_re --outfile_path_geo_params $outfile_geoparams --outfile_path_geo_projection $outfile_path_geo_projection

# import the necessary packages
import argparse
import numpy as np
import gdal
import osr
import osgeo
import cv2
import csv

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path", required=True, help="image path of GeoTIFF")
ap.add_argument("-i", "--image_path", required=True, help="image path")
ap.add_argument("-p", "--outfile_path_image_1", required=True, help="B image")
ap.add_argument("-q", "--outfile_path_image_2", required=True, help="G image")
ap.add_argument("-r", "--outfile_path_image_3", required=True, help="R image")
ap.add_argument("-s", "--outfile_path_image_4", required=True, help="NIR image")
ap.add_argument("-u", "--outfile_path_image_5", required=True, help="RE image")
ap.add_argument("-g", "--outfile_path_geo_params", required=True, help="where output Geo params will be saved")
ap.add_argument("-j", "--outfile_path_geo_projection", required=True, help="where output GeoTiff projection will be saved")
ap.add_argument("-o", "--outfile_path_image", required=False, help="where output PNG image of RGB will be saved")
args = vars(ap.parse_args())

input_image = args["image_path"]
outfile_path_image = args["outfile_path_image"]
outfile_path_image_1 = args["outfile_path_image_1"]
outfile_path_image_2 = args["outfile_path_image_2"]
outfile_path_image_3 = args["outfile_path_image_3"]
outfile_path_image_4 = args["outfile_path_image_4"]
outfile_path_image_5 = args["outfile_path_image_5"]
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

options_list_r = [
    '-ot Byte',
    '-of PNG',
    '-b 1',
    '-scale'
]
options_list_g = [
    '-ot Byte',
    '-of PNG',
    '-b 2',
    '-scale'
]
options_list_b = [
    '-ot Byte',
    '-of PNG',
    '-b 3',
    '-scale'
]
options_list_n = [
    '-ot Byte',
    '-of PNG',
    '-b 4',
    '-scale'
]
options_list_re = [
    '-ot Byte',
    '-of PNG',
    '-b 5',
    '-scale'
]

options_string_r = " ".join(options_list_r)
options_string_g = " ".join(options_list_g)
options_string_b = " ".join(options_list_b)
options_string_n = " ".join(options_list_n)
options_string_re = " ".join(options_list_re)

gdal.Translate(
    outfile_path_image_3,
    input_image,
    options=options_string_r
)
gdal.Translate(
    outfile_path_image_2,
    input_image,
    options=options_string_g
)
gdal.Translate(
    outfile_path_image_1,
    input_image,
    options=options_string_b
)
gdal.Translate(
    outfile_path_image_4,
    input_image,
    options=options_string_n
)
gdal.Translate(
    outfile_path_image_5,
    input_image,
    options=options_string_re
)

if outfile_path_image is not None:
    band1 = cv2.imread(outfile_path_image_1, cv2.IMREAD_UNCHANGED)
    band2 = cv2.imread(outfile_path_image_2, cv2.IMREAD_UNCHANGED)
    band3 = cv2.imread(outfile_path_image_3, cv2.IMREAD_UNCHANGED)

    merged = cv2.merge((band1, band2, band3))

    # cv2.imshow("Result", merged)
    # cv2.waitKey(0)
    cv2.imwrite(outfile_path_image, merged)
