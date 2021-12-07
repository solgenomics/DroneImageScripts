# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/VegetativeIndex/CCC.py --image_path /folder/mypic.png --outfile_path /export/myimages/tgi.png

# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
import math

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path", required=True, help="image path")
ap.add_argument("-o", "--outfile_path", required=True, help="file path where the output will be saved")
args = vars(ap.parse_args())

input_image = args["image_path"]
outfile_path = args["outfile_path"]

#check1 = r/g #lt
#check2 = b/g #lt
#check3 = 2*g - r - b #gt

def threshold_ccc(image_b, image_g, image_r):
    theta1 = 0.95
    theta2 = 0.95
    theta3 = 20

    # grab the image dimensions
    h = image_b.shape[0]
    w = image_b.shape[1]

    image = np.zeros((h,w,1), dtype=np.uint8)

    # loop over the image
    for y in range(0, h):
        for x in range(0, w):
            # threshold the pixel
            image[y, x] = 255 if image_r[y, x]/image_g[y, x] < theta1 and image_b[y, x]/image_g[y, x] < theta2 and 2*image_g[y, x] - image_r[y, x] - image_b[y, x] > theta3 else 0

    # return the thresholded image
    return image

img = cv2.imread(input_image)
b,g,r = cv2.split(img)

ccc = threshold_ccc(b,g,r)

cv2.imwrite(outfile_path, ccc)
