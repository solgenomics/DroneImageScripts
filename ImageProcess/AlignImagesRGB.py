
# Works with Micasense 5 band images. Outputs orthophotomosaic images of each bandself.
# Required cpp/stitching.cpp to be compiled and executable as 'stitching_multi' . Use g++ stitching.cpp -u /usr/bin/stitching_multi `pkg-config opencv4 --cflags --libs`
# stitching_multi program will use CUDA GPU if opencv was installed with CUDA support

def run():
    import sys
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import argparse
    import os, glob
    from multiprocessing import Process, freeze_support
    import imutils
    import statistics
    import matplotlib.pyplot as plt
    import csv

    freeze_support()

    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--log_file_path", required=False, help="file path to write log to. useful for using from the web interface")
    ap.add_argument("-a", "--image_path", required=False, help="image path to directory with all images inside of it. useful for using from command line. e.g. /home/nmorales/MicasenseTest/000")
    ap.add_argument("-b", "--file_with_image_paths", required=False, help="file path to file that has all image file names and temporary file names for each image in it, comma separated and separated by a newline. useful for using from the web interface. e.g. /home/nmorales/myfilewithnames.txt")
    ap.add_argument("-o", "--output_path", required=True, help="output path to directory in which all resulting files will be placed. useful for using from the command line")
    ap.add_argument("-y", "--final_rgb_output_path", required=True, help="output file path for stitched RGB image")
    ap.add_argument("-w", "--work_megapix", required=False, default=0.6, help="Resolution for image registration step. The default is 0.6 Mpx")
    args = vars(ap.parse_args())

    log_file_path = args["log_file_path"]
    image_path = args["image_path"]
    file_with_image_paths = args["file_with_image_paths"]
    output_path = args["output_path"]
    final_rgb_output_path = args["final_rgb_output_path"]
    work_megapix = args["work_megapix"]

    if log_file_path is not None and log_file_path != '':
        sys.stderr = open(log_file_path, 'a')

    def eprint(*args, **kwargs):
        print(*args, file=sys.stderr, **kwargs)

    #Must supply either image_path or file_with_image_paths as a source of images
    imageNamesAll = []
    imageTempNames = []
    if image_path is not None:
        imageNamesAll = glob.glob(os.path.join(image_path,'*.tif'))
        for i in imageNamesAll:
            imageTempNames.append(os.path.join(output_path,i+'temp.tif'))
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

    img_type = "reflectance"
    match_index = 0 # Index of the band 
    max_alignment_iterations = 1000
    warp_mode = cv2.MOTION_HOMOGRAPHY # MOTION_HOMOGRAPHY or MOTION_AFFINE. For Altum images only use HOMOGRAPHY
    pyramid_levels = None # for images with RigRelatives, setting this to 0 or 1 may improve alignment

    sep = " ";
    images_string1 = sep.join(imageNamesAll)

    log_file_path_string = ''
    if log_file_path is not None and log_file_path != '':
        log_file_path_string = " --log_file '"+log_file_path+"'"
    stitchCmd = "stitching_single "+images_string1+" --result1 '"+final_rgb_output_path+"' "+log_file_path_string
    # stitchCmd = "stitching_single "+images_string1+" --result1 '"+final_rgb_output_path+"' --try_cuda yes --log_file "+log_file_path+" --work_megapix "+work_megapix
    if log_file_path is not None:
        eprint(stitchCmd)
        eprint(len(stitchCmd))
    else:
        print(stitchCmd)
        print(len(stitchCmd))
    os.system(stitchCmd)

#     {
#     OK = 0,
#     ERR_NEED_MORE_IMGS = 1,
#     ERR_HOMOGRAPHY_EST_FAIL = 2,
#     ERR_CAMERA_PARAMS_ADJUST_FAIL = 3
#     };

if __name__ == '__main__':
    run()