import argparse
import os, glob
from multiprocessing import Process, freeze_support
import imutils
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--image_path", required=False, help="image path to directory with all images inside of it. useful for using from command line. e.g. /home/nmorales/MicasenseTest/000")
ap.add_argument("-b", "--file_with_image_paths", required=False, help="file path to file that has all image file names in it, separated by a newline. useful for using from the web interface. e.g. /home/nmorales/myfilewithnames.txt")
ap.add_argument("-o", "--output_path", required=True, help="output path to directory in which all resulting files will be placed. useful for using from the command line")
ap.add_argument("-p", "--output_path_band1", required=True, help="output file path in which resulting band 1 will be placed. useful for using from the web interface")
ap.add_argument("-q", "--output_path_band2", required=True, help="output file path in which resulting band 2 will be placed. useful for using from the web interface")
ap.add_argument("-r", "--output_path_band3", required=True, help="output file path in which resulting band 3 will be placed. useful for using from the web interface")
ap.add_argument("-s", "--output_path_band4", required=True, help="output file path in which resulting band 4 will be placed. useful for using from the web interface")
ap.add_argument("-u", "--output_path_band5", required=True, help="output file path in which resulting band 5 will be placed. useful for using from the web interface")
ap.add_argument("-i", "--do_pairwise_stitch", required=False, help="do simple pairwise stitching, no GPS info. the number provided is the maximum number of captures that will be stitched at once.")
args = vars(ap.parse_args())

image_path = args["image_path"]
file_with_image_paths = args["file_with_image_paths"]
output_path = args["output_path"]
output_path_band1 = args["output_path_band1"]
output_path_band2 = args["output_path_band2"]
output_path_band3 = args["output_path_band3"]
output_path_band4 = args["output_path_band4"]
output_path_band5 = args["output_path_band5"]
do_pairwise_stitch = args["do_pairwise_stitch"]

panelNames = None

if image_path is not None:
    imageNamesAll = glob.glob(os.path.join(image_path,'*.tif'))
elif file_with_image_paths is not None:
    imageNamesAll = []
    with open(file_with_image_paths) as fp:
        for line in fp:
            imageNamesAll.append(line.strip())
else:
    print("No input images given. use image_path OR file_with_image_paths args")
    os._exit

imageNamesDict = {}
for i in imageNamesAll:
    s = i.split("_")
    k = s[-1].split(".")
    if s[-2] not in imageNamesDict:
        imageNamesDict[s[-2]] = {}
    imageNamesDict[s[-2]][k[0]] = i

imageNameCaptures = []
for i in sorted (imageNamesDict.keys()):
    im = []
    for j in sorted (imageNamesDict[i].keys()):
        im.append(imageNamesDict[i][j])
    if len(im) > 0:
        imageNameCaptures.append(im)

#print(imageNameCaptures)

def run():
    import micasense.capture as capture
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import micasense.imageutils as imageutils
    import micasense.plotutils as plotutils
    
    freeze_support()

    def enhance_image(rgb):
        gaussian_rgb = cv2.GaussianBlur(rgb, (9,9), 10.0)
        gaussian_rgb[gaussian_rgb<0] = 0
        gaussian_rgb[gaussian_rgb>1] = 1
        unsharp_rgb = cv2.addWeighted(rgb, 1.5, gaussian_rgb, -0.5, 0)
        unsharp_rgb[unsharp_rgb<0] = 0
        unsharp_rgb[unsharp_rgb>1] = 1

        # Apply a gamma correction to make the render appear closer to what our eyes would see
        gamma = 1.4
        gamma_corr_rgb = unsharp_rgb**(1.0/gamma)
        return(gamma_corr_rgb)

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
    doFinalMerge = True
    if do_pairwise_stitch is not None:
        do_pairwise_stitch_int = int(do_pairwise_stitch)
        if len(captures) <= do_pairwise_stitch_int:
            doFinalMerge = False
        for i in range(0, len(captures), do_pairwise_stitch_int):
            im = captures[i:i + do_pairwise_stitch_int]
            if len(im) > 0:
                imageCaptureSets.append(im)
    else:
        for i in sorted (GPSsorter.keys()):
            im = []
            for j in sorted (GPSsorter[i].keys()):
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

    match_index = 0 # Index of the band 
    max_alignment_iterations = 1000
    warp_mode = cv2.MOTION_HOMOGRAPHY # MOTION_HOMOGRAPHY or MOTION_AFFINE. For Altum images only use HOMOGRAPHY
    pyramid_levels = None # for images with RigRelatives, setting this to 0 or 1 may improve alignment

    print(img_type)
    print("Alinging images. Depending on settings this can take from a few seconds to many minutes")
    warp_matrices, alignment_pairs = imageutils.align_capture(captures[0],
                                                              ref_index = match_index,
                                                              max_iterations = max_alignment_iterations,
                                                              warp_mode = warp_mode,
                                                              pyramid_levels = pyramid_levels,
                                                              multithreaded = True)

    print("Finished Aligning, warp matrices={}".format(warp_matrices))


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
            i1 = enhance_image(i1)
            image1 = np.uint8(i1*255)
            images_to_stich1.append(image1)

            i2 = im_aligned[:,:,[2,3,4]]
            i2 = enhance_image(i2)
            image2 = np.uint8(i2*255)
            images_to_stich2.append(image2)

        stitcher = cv2.createStitcher(True) if imutils.is_cv3() else cv2.Stitcher_create(True) #Try GPU #Stitcher::SCANS or Stitcher::PANORAMA
        stitch_result1 = stitcher.stitch(images_to_stich1)
        print(stitch_result1[0])
        print(stitch_result1[1])
        resultsToStitch1.append(stitch_result1[1])
        cv2.imwrite(output_path+"/resultstostitch1_"+str(count)+".png", stitch_result1[1])

        stitcher = cv2.createStitcher(True) if imutils.is_cv3() else cv2.Stitcher_create(True) #Try GPU #Stitcher::SCANS or Stitcher::PANORAMA
        stitch_result2 = stitcher.stitch(images_to_stich2)
        print(stitch_result2[0])
        print(stitch_result2[1])
        resultsToStitch2.append(stitch_result2[1])
        cv2.imwrite(output_path+"/resultstostitch2_"+str(count)+".png", stitch_result2[1])

        count = count + 1

    if doFinalMerge == True:
        stitcher = cv2.createStitcher(True) if imutils.is_cv3() else cv2.Stitcher_create(True) #Try GPU #Stitcher::SCANS or Stitcher::PANORAMA
        final_result1 = stitcher.stitch(resultsToStitch1)
        print(final_result1[0])
        print(final_result1[1])
        final_result_img1 = final_result1[1]

        stitcher = cv2.createStitcher(True) if imutils.is_cv3() else cv2.Stitcher_create(True) #Try GPU #Stitcher::SCANS or Stitcher::PANORAMA
        final_result2 = stitcher.stitch(resultsToStitch2)
        print(final_result2[0])
        print(final_result2[1])
        final_result_img2 = final_result2[1]
    else :
        final_result_img1 = resultsToStitch1[0]
        final_result_img2 = resultsToStitch2[0]


    final_result_img1 = enhance_image(final_result_img1/255) * 255
    final_result_img2 = enhance_image(final_result_img2/255) * 255

    cv2.imwrite(output_path_band1, final_result_img1[:,:,0])
    cv2.imwrite(output_path_band2, final_result_img1[:,:,1])
    cv2.imwrite(output_path_band3, final_result_img1[:,:,2])
    cv2.imwrite(output_path_band4, final_result_img2[:,:,1])
    cv2.imwrite(output_path_band5, final_result_img2[:,:,2])

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