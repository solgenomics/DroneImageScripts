# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/MergeChannels.py --image_path /folder/mypic.png --outfile_path /export/mychoppedimages/outimage.png

# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
import math

def align_images(moving, fixed_im):
    print("ALIGN IMAGES")
    MIN_MATCH_COUNT = 10

    moving_im = cv2.imread(moving, cv2.IMREAD_UNCHANGED)  # image to be distorted
    moving_im_shape = moving_im.shape
    print(moving_im_shape)
    if len(moving_im_shape) == 3:
        if moving_im_shape[2] == 3:
            b,g,r = cv2.split(moving_im)
            moving_im = b

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    print("GET SIFT")

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(moving_im, None)
    kp2, des2 = sift.detectAndCompute(fixed_im, None)

    # use FLANN method to match keypoints. Brute force matches not appreciably better
    # and added processing time is significant.
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    print("GET MATCH")
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches following Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        print("GET HOMOGRAPHY")
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w = fixed_im.shape  # shape of input images, needs to remain the same for output

        print("WARP PERSPECTIVE")
        outimg = cv2.warpPerspective(moving_im, M, (w, h))

        return outimg

    else:
        print("Not enough matches are found for moving image")
        matchesMask = None

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--image_path_band_1", required=True, help="image path band 1")
ap.add_argument("-b", "--image_path_band_2", required=True, help="image path band 2")
ap.add_argument("-c", "--image_path_band_3", required=True, help="image path band 3")
ap.add_argument("-o", "--outfile_path", required=True, help="file path directory where the output will be saved")
ap.add_argument("-d", "--alignment_needed", default=False, help="if alignment is needed")
ap.add_argument("-e", "--is_tiff", default=False, help="if alignment is needed")
args = vars(ap.parse_args())

input_image_band_1 = args["image_path_band_1"]
input_image_band_2 = args["image_path_band_2"]
input_image_band_3 = args["image_path_band_3"]
outfile_path = args["outfile_path"]
alignment_needed = args["alignment_needed"]
is_tiff = args["is_tiff"]

band1 = cv2.imread(input_image_band_1, cv2.IMREAD_UNCHANGED)
band_1_shape = band1.shape
print(band_1_shape)
if len(band_1_shape) == 3:
    if band_1_shape[2] == 3:
        b,g,r = cv2.split(band1)
        band1 = b

if alignment_needed is True:
    band2 = align_images(input_image_band_2, band1)
    if band2 is None:
        band2 = cv2.imread(input_image_band_2, cv2.IMREAD_UNCHANGED)

    band3 = align_images(input_image_band_3, band1)
    if band3 is None:
        band3 = cv2.imread(input_image_band_3, cv2.IMREAD_UNCHANGED)
else:
    band2 = cv2.imread(input_image_band_2, cv2.IMREAD_UNCHANGED)
    band3 = cv2.imread(input_image_band_3, cv2.IMREAD_UNCHANGED)

#print("MERGE IMAGES")
#print(band1.shape)
#print(band2.shape)
#print(band3.shape)
#cv2.imshow("Band1", band1)
#cv2.imshow("Band2", band2)
#cv2.imshow("Band3", band3)
#cv2.waitKey(0)

if is_tiff:
    merged = cv2.merge((band1*255, band2*255, band3*255))
else:
    merged = cv2.merge((band1, band2, band3))

#cv2.imshow("Result", dst)
#cv2.waitKey(0)
cv2.imwrite(outfile_path, merged)
