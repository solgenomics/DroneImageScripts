# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageCropping/CropToChunks.py --inputfile_path /export/archive/mystitchedimage.png --output_path /export/mychoppedimages/ --width 2000 --height 1000

# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--inputfile_path", required=True, help="complete file path to the image you want to cut into chunks")
ap.add_argument("-o", "--output_path", required=True, help="file path directory where the cut images will be saved")
ap.add_argument("-x", "--width", required=True, help="the width of the output image chunks in px")
ap.add_argument("-y", "--height", required=True, help="the height of the output image chunks in px")
args = vars(ap.parse_args())

inputfile_path = args["inputfile_path"]
output_path = args["output_path"]
width = int(args["width"])
height = int(args["height"])

input_image = cv2.imread(inputfile_path, cv2.IMREAD_COLOR)
input_image_size = input_image.shape
print(input_image_size)
input_image_height = input_image_size[0]
input_image_width = input_image_size[1]

current_width = 0
current_height = 0
width_overlap = 500
height_overlap = 500

crops = []
for col in range(0, input_image_width, width-width_overlap):
    for row in range(0, input_image_height, height-height_overlap):
        cropped = input_image[col:col+width, row:row+height]
        if (cropped.shape[0] != 0 and cropped.shape[1] != 0):
            crops.append(cropped)

count = 1
for i in crops:
    # cv2.imshow("Result", i)
    # cv2.waitKey(0)
    cv2.imwrite(output_path+'image'+str(count)+'.png', i)
    count += 1
