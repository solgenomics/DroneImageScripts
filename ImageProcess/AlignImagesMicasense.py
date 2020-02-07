# Works with Micasense 5 band images. Outputs orthophotomosaic images of each bandself.
# Required cpp/stitching.cpp to be compiled and executable as 'stitching_multi' . Use g++ stitching.cpp -u /usr/bin/stitching_multi `pkg-config opencv4 --cflags --libs`
# stitching_multi program will use CUDA GPU if opencv was installed with CUDA support

def run():
    import sys
    from micasense.capture import Capture
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import micasense.imageutils as imageutils
    import micasense.plotutils as plotutils
    import argparse
    import os, glob
    from multiprocessing import Process, freeze_support
    import imutils
    import statistics
    import matplotlib.pyplot as plt
    from micasense.image import Image
    from micasense.panel import Panel
    import micasense.utils as msutils
    import csv
    import pickle

    freeze_support()

    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--log_file_path", required=False, help="file path to write log to. useful for using from the web interface")
    ap.add_argument("-a", "--image_path", required=False, help="image path to directory with all images inside of it. useful for using from command line. e.g. /home/nmorales/MicasenseTest/000. NOTE: a temp folder will be created within this directory")
    ap.add_argument("-b", "--file_with_image_paths", required=False, help="file path to file that has all image file names and temporary file names for each image in it, comma separated and separated by a newline. useful for using from the web interface. e.g. /home/nmorales/myfilewithnames.txt")
    ap.add_argument("-c", "--panel_image_path", required=False, help="image path to directory with all 5 panel images inside of it. useful for using from command line. e.g. /home/nmorales/MicasenseTest/000")
    ap.add_argument("-d", "--file_with_panel_image_paths", required=False, help="file path to file that has all image file names in it, separated by a newline. useful for using from the web interface. e.g. /home/nmorales/myfilewithnames.txt")
    ap.add_argument("-o", "--output_path", required=True, help="output path to directory in which all resulting files will be placed. useful for using from the command line")
    ap.add_argument("-y", "--final_rgb_output_path", required=True, help="output file path for stitched RGB image")
    ap.add_argument("-z", "--final_rnre_output_path", required=True, help="output file path for stitched RNRe image")
    ap.add_argument("-p", "--output_path_band1", required=True, help="output file path in which resulting band 1 will be placed. useful for using from the web interface")
    ap.add_argument("-q", "--output_path_band2", required=True, help="output file path in which resulting band 2 will be placed. useful for using from the web interface")
    ap.add_argument("-r", "--output_path_band3", required=True, help="output file path in which resulting band 3 will be placed. useful for using from the web interface")
    ap.add_argument("-s", "--output_path_band4", required=True, help="output file path in which resulting band 4 will be placed. useful for using from the web interface")
    ap.add_argument("-u", "--output_path_band5", required=True, help="output file path in which resulting band 5 will be placed. useful for using from the web interface")
    ap.add_argument("-n", "--number_captures", required=False, help="When you want to test using only a subset of images.")
    ap.add_argument("-k", "--thin_images", required=False, help="When you have too many images, specify a number of images to skip. e.g. 1 will only use every other image, 2 will use every third image, 3 will use every fourth image.")
    ap.add_argument("-w", "--work_megapix", required=False, help="Resolution for image registration step. The default is 0.6 Mpx")
    ap.add_argument("-x", "--ba_refine_mask", required=False, default='xxxxx', help="Set refinement mask for bundle adjustment. It looks like 'x_xxx' where 'x' means refine respective parameter and '_' means don't refine one, and has the following format: <fx><skew><ppx><aspect><ppy>. The default mask is 'xxxxx'. If bundle adjustment doesn't support estimation of selected parameter then the respective flag is ignored.")
    args = vars(ap.parse_args())

    log_file_path = args["log_file_path"]
    image_path = args["image_path"]
    file_with_image_paths = args["file_with_image_paths"]
    panel_image_path = args["panel_image_path"]
    file_with_panel_image_paths = args["file_with_panel_image_paths"]
    output_path = args["output_path"]
    final_rgb_output_path = args["final_rgb_output_path"]
    final_rnre_output_path = args["final_rnre_output_path"]
    output_path_band1 = args["output_path_band1"]
    output_path_band2 = args["output_path_band2"]
    output_path_band3 = args["output_path_band3"]
    output_path_band4 = args["output_path_band4"]
    output_path_band5 = args["output_path_band5"]
    thin_images = args["thin_images"]
    if thin_images is not None:
        thin_images = int(thin_images)
    number_captures = args["number_captures"]
    if number_captures is not None:
        number_captures = int(number_captures)
    work_megapix = args["work_megapix"]
    ba_refine_mask = args["ba_refine_mask"]

    if sys.version_info[0] < 3:
        raise Exception("Must use Python3. Use python3 in your command line.")

    if log_file_path is not None and log_file_path != '':
        sys.stderr = open(log_file_path, 'a')

    def eprint(*args, **kwargs):
        print(*args, file=sys.stderr, **kwargs)

    #Must supply either image_path or file_with_image_paths as a source of images
    imageNamesAll = []
    imageTempNames = []
    tempImagePath = None
    if image_path is not None:

        tempImagePath = os.path.join(image_path,'temp')
        if not os.path.exists(tempImagePath):
            os.makedirs(tempImagePath)

        imageNamesAll = glob.glob(os.path.join(image_path,'*.tif'))
        for idx, val in enumerate(imageNamesAll):
            imageTempNames.append(os.path.join(tempImagePath,'temp'+str(idx)+'.tif'))

    elif file_with_image_paths is not None:
        with open(file_with_image_paths) as fp:
            for line in fp:
                imageName, tempImageName = line.strip().split(",")
                imageNamesAll.append(imageName)
                imageTempNames.append(tempImageName)
    else:
        if log_file_path is not None:
            eprint("No input images given. use image_path OR file_with_image_paths args")
        else:
            print("No input images given. use image_path OR file_with_image_paths args")
        os._exit

    panelNames = []
    if panel_image_path is not None:
        panelNames = glob.glob(os.path.join(panel_image_path,'*.tif'))
    elif file_with_panel_image_paths is not None:
        with open(file_with_panel_image_paths) as fp:
            for line in fp:
                imageName = line.strip()
                panelNames.append(imageName)
    else:
        if log_file_path is not None:
            eprint("No panel input images given. use panel_image_path OR file_with_panel_image_paths args")
        else:
            print("No panel input images given. use panel_image_path OR file_with_panel_image_paths args")
        os._exit

    panelCap = Capture.from_filelist(panelNames)
    if panelCap.panel_albedo() is not None:
        panel_reflectance_by_band = panelCap.panel_albedo()
    else:
        panel_reflectance_by_band = [0.58, 0.59, 0.59, 0.54, 0.58] #RedEdge band_index order
    panel_irradiance = panelCap.panel_irradiance(panel_reflectance_by_band)

    # for imageName in panelNames:
    #     img = Image(imageName)
    #     band_name = img.band_name
    #     if img.auto_calibration_image:
    #         if log_file_path is not None:
    #             eprint("Found automatic calibration image")
    #         else:
    #             print("Found automatic calibration image")
    #     panel = Panel(img)
    # 
    #     if not panel.panel_detected():
    #         raise IOError("Panel Not Detected!")
    # 
    #     mean, std, num, sat_count = panel.raw()
    #     micasense_panel_calibration = panel.reflectance_from_panel_serial()
    #     radianceToReflectance = micasense_panel_calibration / mean
    #     panelBandCorrection[band_name] = radianceToReflectance
    #     if log_file_path is not None:
    #         eprint("Detected panel serial: {}".format(panel.serial))
    #         eprint("Extracted Panel Statistics:")
    #         eprint("Mean: {}".format(mean))
    #         eprint("Standard Deviation: {}".format(std))
    #         eprint("Panel Pixel Count: {}".format(num))
    #         eprint("Saturated Pixel Count: {}".format(sat_count))
    #         eprint('Panel Calibration: {:1.3f}'.format(micasense_panel_calibration))
    #         eprint('Radiance to reflectance conversion factor: {:1.3f}'.format(radianceToReflectance))
    #     else:
    #         print("Detected panel serial: {}".format(panel.serial))
    #         print("Extracted Panel Statistics:")
    #         print("Mean: {}".format(mean))
    #         print("Standard Deviation: {}".format(std))
    #         print("Panel Pixel Count: {}".format(num))
    #         print("Saturated Pixel Count: {}".format(sat_count))
    #         print('Panel Calibration: {:1.3f}'.format(micasense_panel_calibration))
    #         print('Radiance to reflectance conversion factor: {:1.3f}'.format(radianceToReflectance))

    imageNamesDict = {}
    for i in imageNamesAll:
        s = i.split("_")
        k = s[-1].split(".")
        if s[-2] not in imageNamesDict:
            imageNamesDict[s[-2]] = {}
        imageNamesDict[s[-2]][k[0]] = i

    imageNameCaptures = []
    capture_count = 0
    skip_count = 0
    image_count = 0
    skip_proceed = 1
    num_captures_proceed = 1
    for i in sorted (imageNamesDict.keys()):
        im = []
        if thin_images is not None:
            if image_count > 0 and skip_count < thin_images:
                skip_count = skip_count + 1
                skip_proceed = 0
            else:
                skip_count = 0
                skip_proceed = 1
            image_count = image_count + 1

        if skip_proceed == 1:
            if number_captures is not None:
                if capture_count < number_captures:
                    num_captures_proceed = 1
                else:
                    num_captures_proceed = 0
            if num_captures_proceed == 1:
                for j in sorted (imageNamesDict[i].keys()):
                    imageName = imageNamesDict[i][j]
                    img = Image(imageName)
                    # meta = img.meta
                    # flightImageRaw=plt.imread(imageName)
                    # flightRadianceImage, _, _, _ = msutils.raw_image_to_radiance(meta, flightImageRaw)
                    # flightReflectanceImage = flightRadianceImage * panelBandCorrection[img.band_name]
                    # flightUndistortedReflectance = msutils.correct_lens_distortion(meta, flightReflectanceImage)
                    # calibratedImage = imageNameToCalibratedImageName[imageName]
                    # print(flightUndistortedReflectance.shape)
                    # plt.imsave(calibratedImage, flightUndistortedReflectance, cmap='gray')
                    # calIm = Image(calibratedImage, meta = meta)
                    im.append(img)
                if len(im) > 0:
                    imageNameCaptures.append(im)
                    capture_count = capture_count + 1

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

    captures = []
    for i in imageNameCaptures:
        im = Capture(i)
        captures.append(im)

    match_index = 0 # Index of the band 
    max_alignment_iterations = 1000
    warp_mode = cv2.MOTION_HOMOGRAPHY # MOTION_HOMOGRAPHY or MOTION_AFFINE. For Altum images only use HOMOGRAPHY
    pyramid_levels = None # for images with RigRelatives, setting this to 0 or 1 may improve alignment

    if log_file_path is not None:
        eprint("Aligning images. Depending on settings this can take from a few seconds to many minutes")
    else:
        print("Aligning images. Depending on settings this can take from a few seconds to many minutes")

    warp_matrices = None
    if tempImagePath is not None:
        if os.path.exists(os.path.join(tempImagePath,'capturealignment.pkl')):
            with open(os.path.join(tempImagePath,'capturealignment.pkl'), 'rb') as f:
                warp_matrices, alignment_pairs = pickle.load(f)

    if warp_matrices is None:
        warp_matrices, alignment_pairs = imageutils.align_capture(captures[0],
                                                              ref_index = match_index,
                                                              max_iterations = max_alignment_iterations,
                                                              warp_mode = warp_mode,
                                                              pyramid_levels = pyramid_levels,
                                                              multithreaded = True)

    if log_file_path is not None:
        eprint("Finished Aligning, warp matrices={}".format(warp_matrices))
    else:
        print("Finished Aligning, warp matrices={}".format(warp_matrices))

    if tempImagePath is not None:
        with open(os.path.join(tempImagePath,'capturealignment.pkl'), 'wb') as f:
            pickle.dump([warp_matrices, alignment_pairs], f)

    images_to_stitch1 = []
    images_to_stitch2 = []
    count = 0
    for x in captures:
        im_aligned = x.create_aligned_capture(
            irradiance_list = panel_irradiance,
            warp_matrices = warp_matrices,
            match_index = match_index,
            warp_mode = warp_mode
        )
        if log_file_path is not None:
            eprint(im_aligned.shape)
        else:
            print(im_aligned.shape)

        i1 = im_aligned[:,:,[0,1,2]]
        # i1 = enhance_image(i1)
        image1 = np.uint8(i1*255)
        cv2.imwrite(imageTempNames[count], image1)
        images_to_stitch1.append(imageTempNames[count])
        count = count + 1

        i2 = im_aligned[:,:,[2,3,4]]
        # i2 = enhance_image(i2)
        image2 = np.uint8(i2*255)
        cv2.imwrite(imageTempNames[count], image2)
        images_to_stitch2.append(imageTempNames[count])
        count = count + 1

        del im_aligned
        del i1
        del i2
        del image1
        del image2

    sep = " ";
    images_string1 = sep.join(images_to_stitch1)
    images_string2 = sep.join(images_to_stitch2)
    num_images = len(images_to_stitch1)

    del imageNamesAll
    del imageTempNames
    del imageNamesDict
    del panelNames
    del imageNameCaptures
    del images_to_stitch1
    del images_to_stitch2

    log_file_path_string = ''
    if log_file_path is not None and log_file_path != '':
        log_file_path_string = " --log_file '"+log_file_path+"'"
    stitchCmd = "stitching_multi "+images_string1+" "+images_string2+" --num_images "+str(num_images)+" --result1 '"+final_rgb_output_path+"' --result2 '"+final_rnre_output_path+"' "+log_file_path_string
    # stitchCmd = "stitching_multi "+images_string1+" "+images_string2+" --num_images "+str(num_images)+" --result1 '"+final_rgb_output_path+"' --result2 '"+final_rnre_output_path+"' --log_file "+log_file_path+" --work_megapix "+work_megapix+" --ba_refine_mask "+ba_refine_mask
    # stitchCmd = "stitching_multi "+images_string1+" "+images_string2+" --num_images "+str(len(images_to_stitch1))+" --result1 '"+final_rgb_output_path+"' --result2 '"+final_rnre_output_path+"' --try_cuda yes --log_file "+log_file_path+" --work_megapix "+work_megapix
    if log_file_path is not None:
        eprint(stitchCmd)
        eprint(len(stitchCmd))
    else:
        print(stitchCmd)
        print(len(stitchCmd))
    os.system(stitchCmd)

    final_result_img1 = cv2.imread(final_rgb_output_path, cv2.IMREAD_UNCHANGED)
    final_result_img2 = cv2.imread(final_rnre_output_path, cv2.IMREAD_UNCHANGED)
    final_result_img1 = final_result_img1/255
    final_result_img2 = final_result_img2/255
    # final_result_img1 = enhance_image(final_result_img1)
    # final_result_img2 = enhance_image(final_result_img2)

    plt.imsave(final_rgb_output_path, final_result_img1)
    plt.imsave(final_rnre_output_path, final_result_img2)

    plt.imsave(output_path_band1, final_result_img1[:,:,0], cmap='gray')
    plt.imsave(output_path_band2, final_result_img1[:,:,1], cmap='gray')
    plt.imsave(output_path_band3, final_result_img1[:,:,2], cmap='gray')
    plt.imsave(output_path_band4, final_result_img2[:,:,1], cmap='gray')
    plt.imsave(output_path_band5, final_result_img2[:,:,2], cmap='gray')

#     {
#     OK = 0,
#     ERR_NEED_MORE_IMGS = 1,
#     ERR_HOMOGRAPHY_EST_FAIL = 2,
#     ERR_CAMERA_PARAMS_ADJUST_FAIL = 3
#     };

if __name__ == '__main__':
    run()