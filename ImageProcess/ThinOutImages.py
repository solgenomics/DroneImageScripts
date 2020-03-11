import sys
import cv2
import numpy as np
import argparse
import os, glob
import imutils
import statistics
import csv
from shutil import copyfile

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_dir", required=True, help="directory where all images are")
ap.add_argument("-o", "--image_out_dir", required=True, help="directory where thinned out images will go")
ap.add_argument("-t", "--thin", required=True, help="skip every other image")
args = vars(ap.parse_args())

image_dir = args["image_dir"]
image_out_dir = args["image_out_dir"]
thin = args["thin"]

image_names = glob.glob(os.path.join(image_dir,'*.tif'))

if not os.path.exists(image_out_dir):
    os.makedirs(image_out_dir)

thinned = image_names[0::thin]
for img in thinned:
    name = os.path.basename(img)
    copyfile(img, os.path.join(image_out_dir, name))
