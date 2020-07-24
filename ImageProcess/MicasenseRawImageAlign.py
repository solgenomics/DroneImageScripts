# Works with Micasense 5 band images. Outputs aligned stacks of images

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
    ap.add_argument("-b", "--file_with_image_paths", required=False, help="file path to file that has all image file names and temporary file names for each image in it, comma separated and separated by a newline. useful for using from the web interface. e.g. /home/nmorales/myfilewithnames.txt")
    ap.add_argument("-d", "--file_with_panel_image_paths", required=False, help="file path to file that has all image file names in it, separated by a newline. useful for using from the web interface. e.g. /home/nmorales/myfilewithnames.txt")
    ap.add_argument("-o", "--output_path", required=True, help="output path to directory in which all resulting files will be placed. useful for using from the command line")
    ap.add_argument("-y", "--temporary_development_path", required=False, help="output file path for stitched RGB image")
    ap.add_argument("-a", "--outfile_alignment_file", required=False, help="output file path for alignment matrices")
    ap.add_argument("-i", "--infile_alignment_file", required=False, help="input file path for alignment matrices")
    args = vars(ap.parse_args())

    log_file_path = args["log_file_path"]
    file_with_image_paths = args["file_with_image_paths"]
    file_with_panel_image_paths = args["file_with_panel_image_paths"]
    output_path = args["output_path"]
    temporary_development_path = args["temporary_development_path"]
    infile_alignment_file = args["infile_alignment_file"]
    outfile_alignment_file = args["outfile_alignment_file"]

    if sys.version_info[0] < 3:
        raise Exception("Must use Python3. Use python3 in your command line.")

    if log_file_path is not None and log_file_path != '':
        sys.stderr = open(log_file_path, 'a')

    def eprint(*args, **kwargs):
        print(*args, file=sys.stderr, **kwargs)

    basePath = ''
    imageNamesAll = []
    imageTempNamesBlue = []
    imageTempNamesGreen = []
    imageTempNamesRed = []
    imageTempNamesNIR = []
    imageTempNamesRedEdge = []
    with open(file_with_image_paths) as fp:
        for line in fp:
            imageName, basePath, tempImageNameBlue, tempImageNameGreen, tempImageNameRed, tempImageNameNIR, tempImageNameRedEdge = line.strip().split(",")
            imageNamesAll.append(imageName)
            imageTempNamesBlue.append(tempImageNameBlue)
            imageTempNamesGreen.append(tempImageNameGreen)
            imageTempNamesRed.append(tempImageNameRed)
            imageTempNamesNIR.append(tempImageNameNIR)
            imageTempNamesRedEdge.append(tempImageNameRedEdge)

    panelNames = []
    with open(file_with_panel_image_paths) as fp:
        for line in fp:
            imageName = line.strip()
            panelNames.append(imageName)

    panelCap = Capture.from_filelist(panelNames)
    if panelCap.panel_albedo() is not None:
        panel_reflectance_by_band = panelCap.panel_albedo()
    else:
        panel_reflectance_by_band = [0.58, 0.59, 0.59, 0.54, 0.58] #RedEdge band_index order
    panel_irradiance = panelCap.panel_irradiance(panel_reflectance_by_band)

    imageNamesDict = {}
    for i in imageNamesAll:
        s = i.split("_")
        k = s[-1].split(".")
        if s[-2] not in imageNamesDict:
            imageNamesDict[s[-2]] = {}
        imageNamesDict[s[-2]][k[0]] = i

    match_index = 3 # Index of the band. NIR band
    imageNameCaptures = []
    imageNameMatchIndexImages = []
    for i in sorted (imageNamesDict.keys()):
        im = []
        for j in sorted (imageNamesDict[i].keys()):
            imageName = imageNamesDict[i][j]
            img = Image(imageName)
            im.append(img)
        if len(im) > 0:
            imageNameMatchIndexImages.append(im[match_index])
            imageNameCaptures.append(im)

    captures = []
    for i in imageNameCaptures:
        im = Capture(i)
        captures.append(im)

    max_alignment_iterations = 1000
    warp_mode = cv2.MOTION_HOMOGRAPHY # MOTION_HOMOGRAPHY or MOTION_AFFINE. For Altum images only use HOMOGRAPHY
    pyramid_levels = None # for images with RigRelatives, setting this to 0 or 1 may improve alignment

    if log_file_path is not None:
        eprint("Aligning images. Depending on settings this can take from a few seconds to many minutes")
    else:
        print("Aligning images. Depending on settings this can take from a few seconds to many minutes")


    warp_matrices = None
    if temporary_development_path is not None:
        if os.path.exists(os.path.join(temporary_development_path,'capturealignment.pkl')):
            with open(os.path.join(temporary_development_path,'capturealignment.pkl'), 'rb') as f:
                warp_matrices, alignment_pairs = pickle.load(f)
    if infile_alignment_file is not None:
        with open(infile_alignment_file, 'rb') as f:
            warp_matrices, alignment_pairs = pickle.load(f)

    if warp_matrices is None:
        warp_matrices, alignment_pairs = imageutils.align_capture(
            captures[0],
            ref_index = match_index,
            max_iterations = max_alignment_iterations,
            warp_mode = warp_mode,
            pyramid_levels = pyramid_levels,
            multithreaded = True
        )

    if temporary_development_path is not None:
        with open(os.path.join(temporary_development_path,'capturealignment.pkl'), 'wb') as f:
            pickle.dump([warp_matrices, alignment_pairs], f)
    if outfile_alignment_file is not None:
        with open(outfile_alignment_file, 'wb') as f:
            pickle.dump([warp_matrices, alignment_pairs], f)

    if log_file_path is not None:
        eprint("Finished Aligning, warp matrices={}".format(warp_matrices))
    else:
        print("Finished Aligning, warp matrices={}".format(warp_matrices))

    rotated_imgs = []
    output_lines = []
    counter = 0
    warp_matrices2 = []

    for x in captures:
        print(counter)
        if len(warp_matrices) == len(x.images):
            im_aligned = x.create_aligned_capture(
                irradiance_list = panel_irradiance,
                warp_matrices = warp_matrices,
                match_index = match_index,
                warp_mode = warp_mode
            )
        elif len(warp_matrices2) == len(x.images):
            im_aligned = x.create_aligned_capture(
                irradiance_list = panel_irradiance,
                warp_matrices = warp_matrices2,
                match_index = match_index,
                warp_mode = warp_mode
            )
        else:
            warp_matrices2, alignment_pairs2 = imageutils.align_capture(
                x,
                ref_index = match_index,
                max_iterations = max_alignment_iterations,
                warp_mode = warp_mode,
                pyramid_levels = pyramid_levels,
                multithreaded = True
            )
            im_aligned = x.create_aligned_capture(
                irradiance_list = panel_irradiance,
                warp_matrices = warp_matrices2,
                match_index = match_index,
                warp_mode = warp_mode
            )
        im_aligned_bands = im_aligned.shape[2];

        blue_img_file = imageTempNamesBlue[counter]
        green_img_file = imageTempNamesGreen[counter]
        red_img_file = imageTempNamesRed[counter]
        nir_img_file = imageTempNamesNIR[counter]

        plt.imsave(basePath+blue_img_file, im_aligned[:,:,0], cmap='gray')
        plt.imsave(basePath+green_img_file, im_aligned[:,:,1], cmap='gray')
        plt.imsave(basePath+red_img_file, im_aligned[:,:,2], cmap='gray')
        plt.imsave(basePath+nir_img_file, im_aligned[:,:,3], cmap='gray')

        gps_location_blue = x.images[0].location;
        gps_location_green = x.images[1].location;
        gps_location_red = x.images[2].location;
        gps_location_nir = x.images[3].location;

        output_lines.append([basePath+blue_img_file, gps_location_blue[0], gps_location_blue[1], gps_location_blue[2]])
        output_lines.append([basePath+green_img_file, gps_location_green[0], gps_location_green[1], gps_location_green[2]])
        output_lines.append([basePath+red_img_file, gps_location_red[0], gps_location_red[1], gps_location_red[2]])
        output_lines.append([basePath+nir_img_file, gps_location_nir[0], gps_location_nir[1], gps_location_nir[2]])

        if im_aligned_bands == 5 and len(x.images) == 5:
            red_edge_img_file = imageTempNamesRedEdge[counter]
            plt.imsave(basePath+red_edge_img_file, im_aligned[:,:,4], cmap='gray')
            gps_location_red_edge = x.images[4].location;
            output_lines.append([basePath+red_edge_img_file, gps_location_red_edge[0], gps_location_red_edge[1], gps_location_red_edge[2]])
        else:
            output_lines.append(['NA', 'NA', 'NA', 'NA'])

        counter += 1

    with open(output_path, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(output_lines)
    writeFile.close()

if __name__ == '__main__':
    run()