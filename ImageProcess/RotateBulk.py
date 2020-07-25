# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/RotateBulk.py --input_path /folder/myinput.csv

# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
import math
import csv
import pandas as pd

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_path", required=True, help="input file")
args = vars(ap.parse_args())

input_path = args["input_path"]

input_image_file_data = pd.read_csv(input_path, sep="\t", header=None)

for index, row in input_image_file_data.iterrows():
    input_image = row[0]
    output_image = row[1]
    angle_input = row[2]

    angle = float(angle_input)

    img = cv2.imread(input_image)
    height, width = img.shape[:2]

    image_center = (width/2,height/2)

    rotation_mat = cv2.getRotationMatrix2D(image_center,angle,1)

    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    src = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h))

    cv2.imwrite(output_image, src)
