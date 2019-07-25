
#python3 /workdir/cxgn/DroneImageScripts/ImageProcess/AlignImagesMicasense.py --log_file_path /workdir/exampleDroneImagesTest/log.txt --image_path /workdir/exampleDroneImagesTest/pruned_small --panel_image_path /workdir/exampleDroneImagesTest/panel --output_path /workdir/exampleDroneImagesTest/output/ --final_rgb_output_path /workdir/exampleDroneImagesTest/output/rgb.png --final_rnre_output_path /workdir/exampleDroneImagesTest/output/rnre.png --output_path_band1 /workdir/exampleDroneImagesTest/output/b1.png --output_path_band2 /workdir/exampleDroneImagesTest/output/b2.png --output_path_band3 /workdir/exampleDroneImagesTest/output/b3.png --output_path_band4 /workdir/exampleDroneImagesTest/output/b4.png --output_path_band5 /workdir/exampleDroneImagesTest/output/b5.png --work_megapix 0.6

#python3 /DroneImageScripts/ImageProcess/AlignImagesMicasense.py --log_file_path /DroneImageScripts/Y_07162019_pruned/log.txt --image_path /DroneImageScripts/Y_07162019_pruned/prunedv2 --panel_image_path /DroneImageScripts/Y_07162019_pruned/panel2 --output_path /DroneImageScripts/Y_07162019_pruned/output/ --final_rgb_output_path /DroneImageScripts/Y_07162019_pruned/output/rgb.png --final_rnre_output_path /DroneImageScripts/Y_07162019_pruned/output/rnre.png --output_path_band1 /DroneImageScripts/Y_07162019_pruned/output/b1.png --output_path_band2 /DroneImageScripts/Y_07162019_pruned/output/b2.png --output_path_band3 /DroneImageScripts/Y_07162019_pruned/output/b3.png --output_path_band4 /DroneImageScripts/Y_07162019_pruned/output/b4.png --output_path_band5 /DroneImageScripts/Y_07162019_pruned/output/b5.png 

#python3 /workdir/cxgn/DroneImageScripts/ImageProcess/AlignImagesMicasense.py --log_file_path /workdir/Y_07162019_pruned/log.txt --image_path /workdir/Y_07162019_pruned/prunedv2 --panel_image_path /workdir/Y_07162019_pruned/panel2 --output_path /workdir/Y_07162019_pruned/output/ --final_rgb_output_path /workdir/Y_07162019_pruned/output/rgb.png --final_rnre_output_path /workdir/Y_07162019_pruned/output/rnre.png --output_path_band1 /workdir/Y_07162019_pruned/output/b1.png --output_path_band2 /workdir/Y_07162019_pruned/output/b2.png --output_path_band3 /workdir/Y_07162019_pruned/output/b3.png --output_path_band4 /workdir/Y_07162019_pruned/output/b4.png --output_path_band5 /workdir/Y_07162019_pruned/output/b5.png --work_megapix 0.6

#python3 /workdir/cxgn/DroneImageScripts/ImageProcess/AlignImagesMicasense.py --log_file_path /workdir/Y_07162019_pruned_i2/log.txt --image_path /workdir/Y_07162019_pruned_i2/prunedv2 --panel_image_path /workdir/Y_07162019_pruned_i2/panel2 --output_path /workdir/Y_07162019_pruned_i2/output/ --final_rgb_output_path /workdir/Y_07162019_pruned_i2/output/rgb.png --final_rnre_output_path /workdir/Y_07162019_pruned_i2/output/rnre.png --output_path_band1 /workdir/Y_07162019_pruned_i2/output/b1.png --output_path_band2 /workdir/Y_07162019_pruned_i2/output/b2.png --output_path_band3 /workdir/Y_07162019_pruned_i2/output/b3.png --output_path_band4 /workdir/Y_07162019_pruned_i2/output/b4.png --output_path_band5 /workdir/Y_07162019_pruned_i2/output/b5.png --work_megapix 0.1 --ba_refine_mask '_____'

#python3 /DroneImageScripts/ImageProcess/AlignImagesMicasense.py --log_file_path /Y_07162019_pruned/log.txt --image_path /Y_07162019_pruned/prunedv2 --panel_image_path /Y_07162019_pruned/panel2 --output_path /Y_07162019_pruned/output/ --final_rgb_output_path /Y_07162019_pruned/output/rgb.png --final_rnre_output_path /Y_07162019_pruned/output/rnre.png --output_path_band1 /Y_07162019_pruned/output/b1.png --output_path_band2 /Y_07162019_pruned/output/b2.png --output_path_band3 /Y_07162019_pruned/output/b3.png --output_path_band4 /Y_07162019_pruned/output/b4.png --output_path_band5 /Y_07162019_pruned/output/b5.png --work_megapix 0.6

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
    ap.add_argument("-a", "--image_path", required=False, help="image path to directory with all images inside of it. useful for using from command line. e.g. /home/nmorales/MicasenseTest/000")
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
    ap.add_argument("-w", "--work_megapix", required=False, default=0.6, help="Resolution for image registration step. The default is 0.6 Mpx")
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
    work_megapix = args["work_megapix"]
    ba_refine_mask = args["ba_refine_mask"]

    if sys.version_info[0] < 3:
        raise Exception("Must use Python3. Use python3 in your command line.")

    if log_file_path is not None:
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

    panelBandCorrection = {}
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

    for imageName in panelNames:
        img = Image(imageName)
        band_name = img.band_name
        if img.auto_calibration_image:
            if log_file_path is not None:
                eprint("Found automatic calibration image")
            else:
                print("Found automatic calibration image")
        panel = Panel(img)

        if not panel.panel_detected():
            raise IOError("Panel Not Detected!")

        mean, std, num, sat_count = panel.raw()
        micasense_panel_calibration = panel.reflectance_from_panel_serial()
        radianceToReflectance = micasense_panel_calibration / mean
        panelBandCorrection[band_name] = radianceToReflectance
        if log_file_path is not None:
            eprint("Detected panel serial: {}".format(panel.serial))
            eprint("Extracted Panel Statistics:")
            eprint("Mean: {}".format(mean))
            eprint("Standard Deviation: {}".format(std))
            eprint("Panel Pixel Count: {}".format(num))
            eprint("Saturated Pixel Count: {}".format(sat_count))
            eprint('Panel Calibration: {:1.3f}'.format(micasense_panel_calibration))
            eprint('Radiance to reflectance conversion factor: {:1.3f}'.format(radianceToReflectance))
        else:
            print("Detected panel serial: {}".format(panel.serial))
            print("Extracted Panel Statistics:")
            print("Mean: {}".format(mean))
            print("Standard Deviation: {}".format(std))
            print("Panel Pixel Count: {}".format(num))
            print("Saturated Pixel Count: {}".format(sat_count))
            print('Panel Calibration: {:1.3f}'.format(micasense_panel_calibration))
            print('Radiance to reflectance conversion factor: {:1.3f}'.format(radianceToReflectance))

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
    captureGPSDict = {}
    counter = 0
    for i in imageNameCaptures:
        im = Capture(i)
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

    imageCaptureSets = captures

    img_type = "reflectance"
    match_index = 0 # Index of the band 
    max_alignment_iterations = 1000
    warp_mode = cv2.MOTION_HOMOGRAPHY # MOTION_HOMOGRAPHY or MOTION_AFFINE. For Altum images only use HOMOGRAPHY
    pyramid_levels = None # for images with RigRelatives, setting this to 0 or 1 may improve alignment

    if log_file_path is not None:
        eprint(img_type)
        eprint("Alinging images. Depending on settings this can take from a few seconds to many minutes")
    else:
        print(img_type)
        print("Alinging images. Depending on settings this can take from a few seconds to many minutes")

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
    for x in imageCaptureSets:
        cropped_dimensions, edges = imageutils.find_crop_bounds(x, warp_matrices, warp_mode=warp_mode)
        im_aligned = imageutils.aligned_capture(x, warp_matrices, warp_mode, cropped_dimensions, match_index, img_type=img_type)
        if log_file_path is not None:
            eprint(im_aligned.shape)
        else:
            print(im_aligned.shape)

        i1 = im_aligned[:,:,[0,1,2]]
        i1 = enhance_image(i1)
        image1 = np.uint8(i1*255)
        cv2.imwrite(imageTempNames[count], image1)
        images_to_stitch1.append(imageTempNames[count])
        count = count + 1

        i2 = im_aligned[:,:,[2,3,4]]
        i2 = enhance_image(i2)
        image2 = np.uint8(i2*255)
        cv2.imwrite(imageTempNames[count], image2)
        images_to_stitch2.append(imageTempNames[count])
        count = count + 1

    sep = " ";
    images_string1 = sep.join(images_to_stitch1)
    images_string2 = sep.join(images_to_stitch2)
    num_images = len(images_to_stitch1)

    del imageNamesAll
    del imageTempNames
    del imageNamesDict
    del panelNames
    del imageNameCaptures
    del imageCaptureSets
    del images_to_stitch1
    del images_to_stitch2

    stitchCmd = "stitching_multi "+images_string1+" "+images_string2+" --num_images "+str(num_images)+" --result1 '"+final_rgb_output_path+"' --result2 '"+final_rnre_output_path+"' --log_file "+log_file_path
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
    final_result_img1 = enhance_image(final_result_img1/255)
    final_result_img2 = enhance_image(final_result_img2/255)

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