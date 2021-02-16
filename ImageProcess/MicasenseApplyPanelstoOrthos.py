# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/MicasenseApplyPanelstoOrthos.py --panel_image_path /folder/panels --infile_path_b1 /export/b1.png --infile_path_b2 /export/b2.png --infile_path_b3 /export/b3.png --infile_path_b4 /export/b4.png --infile_path_b5 /export/b5.png --outfile_path_b1 /export/b1.png --outfile_path_b2 /export/b2.png --outfile_path_b3 /export/b3.png --outfile_path_b4 /export/b4.png --outfile_path_b5 /export/b5.png

import sys
from micasense.capture import Capture
import cv2
import numpy as np
import matplotlib.pyplot as plt
import micasense.imageutils as imageutils
import micasense.plotutils as plotutils
import argparse
import os, glob
import imutils
import statistics
import matplotlib.pyplot as plt
from micasense.image import Image
from micasense.panel import Panel
import micasense.utils as msutils
import csv
import math

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--panel_image_path", required=True, help="image path to directory with all 5 panel images inside of it. e.g. /home/nmorales/MicasenseTest/000")
ap.add_argument("-a", "--infile_path_b1", required=True, help="input file for band 1 blue orthophoto")
ap.add_argument("-b", "--infile_path_b2", required=True, help="input file for band 2 green orthophoto")
ap.add_argument("-c", "--infile_path_b3", required=True, help="input file for band 3 red orthophoto")
ap.add_argument("-d", "--infile_path_b4", required=True, help="input file for band 4 nir orthophoto")
ap.add_argument("-e", "--infile_path_b5", required=True, help="input file for band 5 rededge orthophoto")
ap.add_argument("-o", "--outfile_path_b1", required=True, help="where output image will be saved for band 1")
ap.add_argument("-j", "--outfile_path_b2", required=True, help="where output image will be saved for band 2")
ap.add_argument("-k", "--outfile_path_b3", required=True, help="where output image will be saved for band 3")
ap.add_argument("-l", "--outfile_path_b4", required=True, help="where output image will be saved for band 4")
ap.add_argument("-m", "--outfile_path_b5", required=True, help="where output image will be saved for band 5")
args = vars(ap.parse_args())

panel_image_path = args["panel_image_path"]
infile_path_b1 = args["infile_path_b1"]
infile_path_b2 = args["infile_path_b2"]
infile_path_b3 = args["infile_path_b3"]
infile_path_b4 = args["infile_path_b4"]
infile_path_b5 = args["infile_path_b5"]
outfile_path_b1 = args["outfile_path_b1"]
outfile_path_b2 = args["outfile_path_b2"]
outfile_path_b3 = args["outfile_path_b3"]
outfile_path_b4 = args["outfile_path_b4"]
outfile_path_b5 = args["outfile_path_b5"]

panelNames = glob.glob(os.path.join(panel_image_path,'*.tif'))
panelCap = Capture.from_filelist(panelNames)
if panelCap.panel_albedo() is not None:
    panel_reflectance_by_band = panelCap.panel_albedo()
else:
    panel_reflectance_by_band = [0.58, 0.59, 0.59, 0.54, 0.58] #RedEdge band_index order b,g,r,nir,rededge
panel_irradiance = panelCap.panel_irradiance(panel_reflectance_by_band)

band1 = cv2.imread(infile_path_b1) * math.pi / panel_irradiance[0]
band2 = cv2.imread(infile_path_b2) * math.pi / panel_irradiance[1]
band3 = cv2.imread(infile_path_b3) * math.pi / panel_irradiance[2]
band4 = cv2.imread(infile_path_b4) * math.pi / panel_irradiance[3]
band5 = cv2.imread(infile_path_b5) * math.pi / panel_irradiance[4]

cv2.imwrite(outfile_path_b1, band1)
cv2.imwrite(outfile_path_b2, band2)
cv2.imwrite(outfile_path_b3, band3)
cv2.imwrite(outfile_path_b4, band4)
cv2.imwrite(outfile_path_b5, band5)
