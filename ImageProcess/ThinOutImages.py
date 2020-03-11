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
thin = int(args["thin"])

blue_image_names = glob.glob(os.path.join(image_dir,'*1.tif'))
green_image_names = glob.glob(os.path.join(image_dir,'*2.tif'))
red_image_names = glob.glob(os.path.join(image_dir,'*3.tif'))
nir_image_names = glob.glob(os.path.join(image_dir,'*4.tif'))
red_edge_image_names = glob.glob(os.path.join(image_dir,'*5.tif'))

if not os.path.exists(image_out_dir):
    os.makedirs(image_out_dir)

thinned_blue = blue_image_names[0::thin]
thinned_green = green_image_names[0::thin]
thinned_red = red_image_names[0::thin]
thinned_nir = nir_image_names[0::thin]
thinned_red_edge = red_edge_image_names[0::thin]

thinned_imgs_list = [thinned_blue, thinned_green, thinned_red, thinned_nir, thinned_red_edge]
thinned_imgs = [y for x in thinned_imgs_list for y in x]

for img in thinned_imgs:
    name = os.path.basename(img)
    copyfile(img, os.path.join(image_out_dir, name))
