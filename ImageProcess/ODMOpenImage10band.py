# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/ODMOpenImage10band.py --image_path /folder/mypic.tif --outfile_path_b1 /export/b1.png --outfile_path_b2 /export/b2.png --outfile_path_b3 /export/b3.png --outfile_path_b4 /export/b4.png --outfile_path_b5 /export/b5.png --outfile_path_b6 /export/b6.png --outfile_path_b7 /export/b7.png --outfile_path_b8 /export/b8.png --outfile_path_b9 /export/b9.png --outfile_path_b10 /export/b10.png --odm_radiocalibrated FALSE

# import the necessary packages
import argparse
import numpy as np
import math
import cv2
import rasterio

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path", required=True, help="image path")
ap.add_argument("-o", "--outfile_path_b1", required=True, help="where output image will be saved for band 1")
ap.add_argument("-j", "--outfile_path_b2", required=True, help="where output image will be saved for band 2")
ap.add_argument("-k", "--outfile_path_b3", required=True, help="where output image will be saved for band 3")
ap.add_argument("-l", "--outfile_path_b4", required=True, help="where output image will be saved for band 4")
ap.add_argument("-m", "--outfile_path_b5", required=True, help="where output image will be saved for band 5")
ap.add_argument("-n", "--outfile_path_b6", required=True, help="where output image will be saved for band 6")
ap.add_argument("-p", "--outfile_path_b7", required=True, help="where output image will be saved for band 7")
ap.add_argument("-s", "--outfile_path_b8", required=True, help="where output image will be saved for band 8")
ap.add_argument("-t", "--outfile_path_b9", required=True, help="where output image will be saved for band 9")
ap.add_argument("-u", "--outfile_path_b10", required=True, help="where output image will be saved for band 10")
ap.add_argument("-r", "--odm_radiocalibrated", required=False, help="if the image was radiocalibrated by ODM set this to True")
args = vars(ap.parse_args())

input_image = args["image_path"]
outfile_path_b1 = args["outfile_path_b1"]
outfile_path_b2 = args["outfile_path_b2"]
outfile_path_b3 = args["outfile_path_b3"]
outfile_path_b4 = args["outfile_path_b4"]
outfile_path_b5 = args["outfile_path_b5"]
outfile_path_b6 = args["outfile_path_b6"]
outfile_path_b7 = args["outfile_path_b7"]
outfile_path_b8 = args["outfile_path_b8"]
outfile_path_b9 = args["outfile_path_b9"]
outfile_path_b10 = args["outfile_path_b10"]
odm_radiocalibrated = args["odm_radiocalibrated"]

print(input_image)
dataset = rasterio.open(input_image)
print(dataset.count)
print(dataset.tags())

band1 = dataset.read(1)
band2 = dataset.read(2)
band3 = dataset.read(3)
band4 = dataset.read(4)
band5 = dataset.read(5)
band6 = dataset.read(6)
band7 = dataset.read(7)
band8 = dataset.read(8)
band9 = dataset.read(9)
band10 = dataset.read(10)

if np.max(band4) < 1:
    odm_radiocalibrated = 'True'

if odm_radiocalibrated == 'True':
    band1 = band1*255
    band2 = band2*255
    band3 = band3*255
    band4 = band4*255
    band5 = band5*255
    band6 = band6*255
    band7 = band7*255
    band8 = band8*255
    band9 = band9*255
    band10 = band10*255

#cv2.imshow("Result", band1.ReadAsArray())
#cv2.waitKey(0)

cv2.imwrite(outfile_path_b1, band1)
cv2.imwrite(outfile_path_b2, band2)
cv2.imwrite(outfile_path_b3, band3)
cv2.imwrite(outfile_path_b4, band4)
cv2.imwrite(outfile_path_b5, band5)
cv2.imwrite(outfile_path_b6, band6)
cv2.imwrite(outfile_path_b7, band7)
cv2.imwrite(outfile_path_b8, band8)
cv2.imwrite(outfile_path_b9, band9)
cv2.imwrite(outfile_path_b10, band10)
