# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/GetMicasenseImageGPS.py --input_image_file  /folder/myimage.tiff --outfile_path /export/myresults.csv

# import the necessary packages
import sys
import argparse
import csv
import imutils
import cv2
import numpy as np
import math
from PIL import Image
import micasense.imageutils as imageutils
import micasense.plotutils as plotutils
from micasense.image import Image
from micasense.panel import Panel
import micasense.utils as msutils
from micasense.capture import Capture

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--log_file_path", required=False, help="file path to write log to. useful for using from the web interface")
ap.add_argument("-i", "--input_image_file", required=True, help="file path for image")
ap.add_argument("-o", "--outfile_path", required=True, help="file path where the output will be saved")
args = vars(ap.parse_args())

log_file_path = args["log_file_path"]
input_file = args["input_image_file"]
outfile_path = args["outfile_path"]

if sys.version_info[0] < 3:
    raise Exception("Must use Python3. Use python3 in your command line.")

if log_file_path is not None:
    sys.stderr = open(log_file_path, 'a')

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

img = Image(input_file)

with open(outfile_path, 'w') as writeFile:
    writer = csv.writer(writeFile)
    # GSD resolution for Micasenes camera in m/p
    writer.writerows([[img.latitude, img.longitude, img.altitude, 0.06857*100*img.altitude/10]])

writeFile.close()
