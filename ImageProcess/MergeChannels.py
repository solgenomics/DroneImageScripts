# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/MergeChannels.py --image_path /folder/mypic.png --outfile_path /export/mychoppedimages/outimage.png

# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
import math

def align_images(moving, fixed):
    print("ALIGN IMAGES")
    MIN_MATCH_COUNT = 10

    moving_im = cv2.imread(moving, 0)  # image to be distorted
    fixed_im = cv2.imread(fixed, 0)  # image to be matched

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

        h, w = moving_im.shape  # shape of input images, needs to remain the same for output

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
args = vars(ap.parse_args())

input_image_band_1 = args["image_path_band_1"]
input_image_band_2 = args["image_path_band_2"]
input_image_band_3 = args["image_path_band_3"]
outfile_path = args["outfile_path"]

band1 = cv2.imread(input_image_band_1, 0)
band2 = align_images(input_image_band_2, input_image_band_1)
band3 = align_images(input_image_band_3, input_image_band_1)

print("MERGE IMAGES")
merged = cv2.merge((band1, band2, band3))

#cv2.imshow("Result", dst)
#cv2.waitKey(0)
cv2.imwrite(outfile_path, merged)

