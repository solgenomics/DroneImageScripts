import argparse
import os, glob
from multiprocessing import Process, freeze_support
import imutils
import statistics

# panelNames = []
# imageNames = []
# outpathNames = []

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--image_path", required=True, default="/home/nmorales/MicasenseTest/000", help="image path")
ap.add_argument("-o", "--output_path", required=True, help="output image path")
ap.add_argument("-i", "--do_pairwise_stitch", required=False, help="do dumb pairwise stitching, no GPS info")
# ap.add_argument("-a", "--image_path_band_1", required=True, help="image path band 1")
# ap.add_argument("-b", "--image_path_band_2", required=True, help="image path band 2")
# ap.add_argument("-c", "--image_path_band_3", required=False, help="image path band 3")
# ap.add_argument("-d", "--image_path_band_4", required=False, help="image path band 4")
# ap.add_argument("-e", "--image_path_band_5", required=False, help="image path band 5")
# ap.add_argument("-f", "--outpath_aligned_image_path_band_1", required=False, help="outpath for aligned image path band 1")
# ap.add_argument("-g", "--outpath_aligned_image_path_band_2", required=False, help="outpath for aligned image path band 2")
# ap.add_argument("-p", "--outpath_aligned_image_path_band_3", required=False, help="outpath for aligned image path band 3")
# ap.add_argument("-i", "--outpath_aligned_image_path_band_4", required=False, help="outpath for aligned image path band 4")
# ap.add_argument("-j", "--outpath_aligned_image_path_band_5", required=False, help="outpath for aligned image path band 5")
# ap.add_argument("-k", "--panel_image_path_band_1", required=False, help="panel image path band 1")
# ap.add_argument("-l", "--panel_image_path_band_2", required=False, help="panel image path band 2")
# ap.add_argument("-m", "--panel_image_path_band_3", required=False, help="panel image path band 3")
# ap.add_argument("-n", "--panel_image_path_band_4", required=False, help="panel image path band 4")
# ap.add_argument("-o", "--panel_image_path_band_5", required=False, help="panel image path band 5")
args = vars(ap.parse_args())

image_path = args["image_path"]
output_path = args["output_path"]
do_pairwise_stitch = args["do_pairwise_stitch"]
# input_image_bands = [args["image_path_band_1"], args["image_path_band_2"], args["image_path_band_3"], args["image_path_band_4"], args["image_path_band_5"]]
# outpath_aligned_image_bands = [args["outpath_aligned_image_path_band_1"], args["outpath_aligned_image_path_band_2"], args["outpath_aligned_image_path_band_3"], args["outpath_aligned_image_path_band_4"], args["outpath_aligned_image_path_band_5"]]
# panel_image_bands = [args["panel_image_path_band_1"], args["panel_image_path_band_2"], args["panel_image_path_band_3"], args["panel_image_path_band_4"], args["panel_image_path_band_5"]]

panelNames = None

# imagePath = "Downloads/MicasenseExample5AlignmentImages"
# imageNames = glob.glob(os.path.join(imagePath,'IMG_0085_*.jpg'))

# imagePath = "Downloads/rededge-m"
# imageNames = glob.glob(os.path.join(imagePath,'img01_*.tif'))

# imagePath = "Downloads/MicasenseTest/Panels"
# imageNames = glob.glob(os.path.join(imagePath,'IMG_0432_*.tif'))

# imagePath = "/home/nmorales/MicasenseTest/000"
imageNamesAll = glob.glob(os.path.join(image_path,'*.tif'))
imageNamesDict = {}
for i in imageNamesAll:
    s = i.split("_")
    k = s[2].split(".")
    if s[1] not in imageNamesDict:
        imageNamesDict[s[1]] = {}
    imageNamesDict[s[1]][k[0]] = i

imageNameCaptures = []
for i in sorted (imageNamesDict.keys()):
    im = []
    for j in sorted (imageNamesDict[i].keys()):
        #print(imageNamesDict[i][j])
        im.append(imageNamesDict[i][j])
    if len(im) > 0:
        imageNameCaptures.append(im)

# imagePath = "Downloads/NickKExample5AlignmentImages"
# imageNames = glob.glob(os.path.join(imagePath,'IMG_0999_*.jpg'))

# for i in input_image_bands:
#     if i is not None:
#         imageNames.append(i)
# 
# for i in outpath_aligned_image_bands:
#     if i is not None:
#         outpathNames.append(i)
# 
# for i in panel_image_bands:
#     if i is not None:
#         panelNames.append(i)

def run():
    import micasense.capture as capture
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import micasense.imageutils as imageutils
    import micasense.plotutils as plotutils
    
    freeze_support()

    if panelNames is not None:
        panelCap = capture.Capture.from_filelist(panelNames)
    else:
        panelCap = None

    captures = []
    captureGPSDict = {}
    counter = 0
    for i in imageNameCaptures:
        im = capture.Capture.from_filelist(i)
        captures.append(im)
        latitudes = []
        longitudes = []
        altitudes = []
        for i,img in enumerate(im.images):
            latitudes.append(img.latitude)
            longitudes.append(img.longitude)
            altitudes.append(img.altitude)
        captureGPSDict[counter] = [round(statistics.mean(latitudes), 4), round(statistics.mean(longitudes), 4), statistics.mean(altitudes)]
        counter = counter + 1

    GPSsorter = {}
    for counter, loc in captureGPSDict.items(): 
        if loc[0] not in GPSsorter:
            GPSsorter[loc[0]] = {}
        GPSsorter[loc[0]][loc[1]] = counter

    imageCaptureSets = []
    if do_pairwise_stitch == 1:
        imageCaptureSets = captures
    else:
        for i in sorted (GPSsorter.keys()):
            im = []
            for j in sorted (GPSsorter[i].keys()):
                #print(GPSsorter[i][j])
                im.append(captures[GPSsorter[i][j]])
            if len(im) > 0:
                imageCaptureSets.append(im)

    if panelCap is not None:
        if panelCap.panel_albedo() is not None:
            panel_reflectance_by_band = panelCap.panel_albedo()
        else:
            panel_reflectance_by_band = [0.67, 0.69, 0.68, 0.61, 0.67] #RedEdge band_index order
        panel_irradiance = panelCap.panel_irradiance(panel_reflectance_by_band)    
        img_type = "reflectance"
    #    capture1.plot_undistorted_reflectance(panel_irradiance)
    else:
        if captures[0].dls_present():
            img_type='reflectance'
    #        capture1.plot_undistorted_reflectance(capture1.dls_irradiance())
        else:
            img_type = "radiance"
    #        capture1.plot_undistorted_radiance()

    ## Alignment settings
    match_index = 0 # Index of the band 
    max_alignment_iterations = 1000
    warp_mode = cv2.MOTION_HOMOGRAPHY # MOTION_HOMOGRAPHY or MOTION_AFFINE. For Altum images only use HOMOGRAPHY
    pyramid_levels = None # for images with RigRelatives, setting this to 0 or 1 may improve alignment

    print(img_type)
    print("Alinging images. Depending on settings this can take from a few seconds to many minutes")
    # Can potentially increase max_iterations for better results, but longer runtimes
    warp_matrices, alignment_pairs = imageutils.align_capture(captures[0],
                                                              ref_index = match_index,
                                                              max_iterations = max_alignment_iterations,
                                                              warp_mode = warp_mode,
                                                              pyramid_levels = pyramid_levels,
                                                              multithreaded = True)

    print("Finished Aligning, warp matrices={}".format(warp_matrices))


    if do_pairwise_stitch == 1:
        print("DOING PAIRWISE..")
        cropped_dimensions, edges = imageutils.find_crop_bounds(captures[0], warp_matrices, warp_mode=warp_mode)
        im_aligned = imageutils.aligned_capture(captures[0], warp_matrices, warp_mode, cropped_dimensions, match_index, img_type=img_type)
        print(im_aligned.shape)

        i1 = im_aligned[:,:,[0,1,2]]
        img1 = np.uint8(i1*255)

        i2 = im_aligned[:,:,[2,3,4]]
        img2 = np.uint8(i2*255)

        for i in range(1, len(captures)):
            cropped_dimensions, edges = imageutils.find_crop_bounds(captures[i], warp_matrices, warp_mode=warp_mode)
            im_aligned = imageutils.aligned_capture(captures[i], warp_matrices, warp_mode, cropped_dimensions, match_index, img_type=img_type)
            print(im_aligned.shape)

            i1 = im_aligned[:,:,[0,1,2]]
            image1 = np.uint8(i1*255)

            i2 = im_aligned[:,:,[2,3,4]]
            image2 = np.uint8(i2*255)

            stitcher = cv2.createStitcher(True) if imutils.is_cv3() else cv2.Stitcher_create(True) #Try GPU #Stitcher::SCANS or Stitcher::PANORAMA

            stitch_result1 = stitcher.stitch([img1, image1])
            print(stitch_result1[0])
            print(stitch_result1[1])
            img1 = stitch_result1[1]

            stitcher = cv2.createStitcher(True) if imutils.is_cv3() else cv2.Stitcher_create(True) #Try GPU #Stitcher::SCANS or Stitcher::PANORAMA

            stitch_result2 = stitcher.stitch([img2, image2])
            print(stitch_result2[0])
            print(stitch_result2[1])
            img2 = stitch_result2[1]

            cv2.imwrite(output_path+"/resultstostitch1_"+str(count)+".png", img1)
            cv2.imwrite(output_path+"/resultstostitch2_"+str(count)+".png", img2)

        final_result_img1 = img1
        final_result_img2 = img2
    else:
        resultsToStitch1 = []
        resultsToStitch2 = []
        count = 1
        for x in imageCaptureSets:
            images_to_stich1 = []
            images_to_stich2 = []
            for i in x:
                cropped_dimensions, edges = imageutils.find_crop_bounds(i, warp_matrices, warp_mode=warp_mode)
                im_aligned = imageutils.aligned_capture(i, warp_matrices, warp_mode, cropped_dimensions, match_index, img_type=img_type)
                print(im_aligned.shape)

                i1 = im_aligned[:,:,[0,1,2]]
                image1 = np.uint8(i1*255)
                images_to_stich1.append(image1)

                i2 = im_aligned[:,:,[2,3,4]]
                image2 = np.uint8(i2*255)
                images_to_stich2.append(image2)

            stitcher = cv2.createStitcher(True) if imutils.is_cv3() else cv2.Stitcher_create(True) #Try GPU #Stitcher::SCANS or Stitcher::PANORAMA

            stitch_result1 = stitcher.stitch(images_to_stich1)
            print(stitch_result1[0])
            print(stitch_result1[1])

            stitcher = cv2.createStitcher(True) if imutils.is_cv3() else cv2.Stitcher_create(True) #Try GPU #Stitcher::SCANS or Stitcher::PANORAMA

            stitch_result2 = stitcher.stitch(images_to_stich2)
            print(stitch_result2[0])
            print(stitch_result2[1])
            resultsToStitch1.append(stitch_result1[1])
            resultsToStitch2.append(stitch_result2[1])

            cv2.imwrite(output_path+"/resultstostitch1_"+str(count)+".png", stitch_result1[1])
            cv2.imwrite(output_path+"/resultstostitch2_"+str(count)+".png", stitch_result2[1])

            count = count + 1

        stitcher = cv2.createStitcher(True) if imutils.is_cv3() else cv2.Stitcher_create(True) #Try GPU #Stitcher::SCANS or Stitcher::PANORAMA

        final_result1 = stitcher.stitch(resultsToStitch1)
        print(final_result1[0])
        print(final_result1[1])

        stitcher = cv2.createStitcher(True) if imutils.is_cv3() else cv2.Stitcher_create(True) #Try GPU #Stitcher::SCANS or Stitcher::PANORAMA

        final_result2 = stitcher.stitch(resultsToStitch2)
        print(final_result2[0])
        print(final_result2[1])
        final_result_img1 = final_result1[1]
        final_result_img2 = final_result2[1]
#     {
#     OK = 0,
#     ERR_NEED_MORE_IMGS = 1,
#     ERR_HOMOGRAPHY_EST_FAIL = 2,
#     ERR_CAMERA_PARAMS_ADJUST_FAIL = 3
#     };

    cv2.imwrite(output_path+"/result_1.png", final_result_img1)
    cv2.imwrite(output_path+"/result_2.png", final_result_img2)

if __name__ == '__main__':
    run()