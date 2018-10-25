import numpy as np
import imutils
import cv2

class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3()

    def stitch(self, images, ratio=0.75, reprojThresh=4.0):
        # unpack the images, then detect keypoints and extract local invariant descriptors from them
        # imageB = images[2]
        # imageA = images[0]
        imageA = images.pop(0)
        count = 1
        (kpsA, kpsB, result, images_non_used) = self.stitch_pair(imageA, images, ratio, reprojThresh, count, 2)
        
        for image in images:
            count += 1
            (kpsA, kpsB, result_stitch, images_non_used) = self.stitch_pair(result, images_non_used, ratio, reprojThresh, count, 2)
            if result_stitch is not None:
                result = result_stitch

        return (kpsA, kpsB, result)

    def stitch_pair(self, imageA, images, ratio, reprojThresh, count, type):
        (kpsA, kpsNumpyA, featuresA) = self.detectAndDescribe(imageA)
        kpsAimage=cv2.drawKeypoints(imageA, kpsA, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('first_image'+str(count),kpsAimage)

        most_matches = 0
        best_match = None
        image_index = 0
        best_match_index = 0
        for image in images:
            (kpsB, kpsNumpyB, featuresB) = self.detectAndDescribe(image)

            # match features between the two images
            if type == 1:
                (matches, H, status) = self.matchKeypoints(kpsNumpyA, kpsNumpyB, featuresA, featuresB, ratio, reprojThresh)
            if type == 2:
                (matches, H, status) = self.matchKeypoints(kpsNumpyB, kpsNumpyA, featuresB, featuresA, ratio, reprojThresh)
            if len(matches) > most_matches:
                most_matches = len(matches)
                best_match = [matches, H, status, image, kpsB]
                best_match_index = image_index
            
            # increment current image index
            image_index += 1

        if best_match is not None:
            kpsMatch=cv2.drawKeypoints(best_match[3], best_match[4], None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow('matched_image'+str(count),kpsMatch)            
            if type == 1:
                result = cv2.warpPerspective(imageA, best_match[1], (imageA.shape[1] + best_match[3].shape[1], imageA.shape[0]))
                cv2.imshow('warped image'+str(count),result)
                result[0:best_match[3].shape[0], 0:best_match[3].shape[1]] = best_match[3]
            if type == 2:
                result = cv2.warpPerspective(best_match[3], best_match[1], (imageA.shape[1] + best_match[3].shape[1], best_match[3].shape[0]))
                cv2.imshow('warped image'+str(count),result)
                result[0:imageA.shape[0], 0:imageA.shape[1]] = imageA
            used_image = images.pop(best_match_index)

            # Mask of non-black pixels (assuming image has a single channel).
            # mask = result > 0
            # # Coordinates of non-black pixels.
            # coords = np.argwhere(mask)
            # # Bounding box of non-black pixels.
            # (x0, y0, z0) = coords.min(axis=0)
            # (x1, y1, z1) = coords.max(axis=0) + 1   # slices are exclusive at the top
            # 
            # # Get the contents of the bounding box.
            # result = result[x0:x1, y0:y1, z0:z1]
            cv2.imshow('stitch result'+str(count),result)
            return (kpsA, kpsB, result, images)
        else:
            return (None, None, None, images)

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Only works in OpenCV 3.X
        # detect and extract features from the image
        descriptor = cv2.xfeatures2d.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(image, None)

        # convert the keypoints from KeyPoint objects to NumPy arrays
        kps_numpy = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return (kps, kps_numpy, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            # return the matches along with the homograpy matrix and status of each matched point
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        return (matches, None, None)
