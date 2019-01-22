# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageStitching/PanoramaStitchZipfile.py --zipfile_path /home/myimagezipfile.zip --extract_path /home/unzip/here/ --outfile_path /export/archive/mystitchedimage.png

# import the necessary packages
from Panorama.Panorama import Stitcher
import os
import argparse
import imutils
import cv2
import zipfile
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--zipfile_path", required=True, help="complete file path to zipfile with images")
ap.add_argument("-e", "--extract_path", required=True, help="file path where to extract images")
ap.add_argument("-o", "--outfile_path", required=True, help="complete file path for output stitched image")
args = vars(ap.parse_args())

zipfile_path = args["zipfile_path"]
extract_path = args["extract_path"]
outfile_path = args["outfile_path"]

images = []
zfile = zipfile.ZipFile(zipfile_path)
zfile.extractall(path=extract_path)
for finfo in zfile.infolist():
    file_name = extract_path + finfo.filename
    if os.path.isfile(file_name):
        print(file_name)
        image = cv2.imread(file_name)
        images.append(image)
    
stitcher = cv2.createStitcher(True) if imutils.is_cv3() else cv2.Stitcher_create(True) #Try GPU #Stitcher::SCANS or Stitcher::PANORAMA
stitch_result = stitcher.stitch(images)
status = stitch_result[0]
print(status)
# OK = 0
# ERR_NEED_MORE_IMGS = 1
# ERR_HOMOGRAPHY_EST_FAIL = 2
# ERR_CAMERA_PARAMS_ADJUST_FAIL = 3
result = stitch_result[1]

cv2.imwrite(outfile_path, result)
#cv2.imshow("Result", result)
#cv2.imwrite("streakedimage.png", result)
#cv2.waitKey(0)

