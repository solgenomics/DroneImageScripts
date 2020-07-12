# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/MatchAndAlignImages.py --image_path1 /folder/mypic1.png --image_path2 /folder/mypic2.png --outfile_path /export/myalignedimages/outimage.png

# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
import math
import csv

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path1", required=True, help="image path1")
ap.add_argument("-j", "--image_path2", required=True, help="image path2")
ap.add_argument("-o", "--outfile_match_path", required=True, help="file path directory where the match output will be saved")
ap.add_argument("-p", "--outfile_path", required=True, help="file path directory where the output will be saved")
ap.add_argument("-r", "--results_outfile_path_src", required=True, help="file path directory where the output match points will be saved")
ap.add_argument("-q", "--results_outfile_path_dst", required=True, help="file path directory where the output match points will be saved")
ap.add_argument("-m", "--max_features", required=True, help="maximum number of match points")
args = vars(ap.parse_args())

input_image1 = args["image_path1"]
input_image2 = args["image_path2"]
outfile_match_path = args["outfile_match_path"]
outfile_path = args["outfile_path"]
results_outfile_path = args["results_outfile_path_src"]
results_outfile_path_2 = args["results_outfile_path_dst"]
MAX_FEATURES = int(args["max_features"])

GOOD_MATCH_PERCENT = 0.02 #0.15

img1 = cv2.imread(input_image2, cv2.IMREAD_UNCHANGED)
img2 = cv2.imread(input_image1, cv2.IMREAD_UNCHANGED)

# orb = cv2.ORB_create(MAX_FEATURES)
orb = cv2.xfeatures2d.SIFT_create(MAX_FEATURES)
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

# matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
# matches = matcher.match(descriptors1, descriptors2, None)
# 
# matches.sort(key=lambda x: x.distance, reverse=False)
# numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
# matches = matches[:numGoodMatches]

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors1,descriptors2)
matches = sorted(matches, key = lambda x:x.distance)

imMatches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)
cv2.imwrite(outfile_match_path, imMatches)

result_file_lines = []
result_file_lines_2 = []

points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)
for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
    result_file_lines.append(keypoints2[match.trainIdx].pt)
    result_file_lines_2.append(keypoints1[match.queryIdx].pt)

#print(points1)
#print(points2)

# h, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 4)
# height, width = img2.shape
# im1Reg = cv2.warpPerspective(img1, h, (width, height))
# 
# cv2.imwrite(outfile_path, im1Reg)

with open(results_outfile_path, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(result_file_lines)
writeFile.close()

with open(results_outfile_path_2, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(result_file_lines_2)
writeFile.close()
