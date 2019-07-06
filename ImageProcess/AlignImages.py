import argparse
import os, glob
from multiprocessing import Process, freeze_support

# panelNames = []
# imageNames = []
outpathNames = []

ap = argparse.ArgumentParser()
# ap.add_argument("-a", "--image_path_band_1", required=True, help="image path band 1")
# ap.add_argument("-b", "--image_path_band_2", required=True, help="image path band 2")
# ap.add_argument("-c", "--image_path_band_3", required=False, help="image path band 3")
# ap.add_argument("-d", "--image_path_band_4", required=False, help="image path band 4")
# ap.add_argument("-e", "--image_path_band_5", required=False, help="image path band 5")
ap.add_argument("-f", "--outpath_aligned_image_path_band_1", required=False, help="outpath for aligned image path band 1")
ap.add_argument("-g", "--outpath_aligned_image_path_band_2", required=False, help="outpath for aligned image path band 2")
ap.add_argument("-p", "--outpath_aligned_image_path_band_3", required=False, help="outpath for aligned image path band 3")
ap.add_argument("-i", "--outpath_aligned_image_path_band_4", required=False, help="outpath for aligned image path band 4")
ap.add_argument("-j", "--outpath_aligned_image_path_band_5", required=False, help="outpath for aligned image path band 5")
# ap.add_argument("-k", "--panel_image_path_band_1", required=False, help="panel image path band 1")
# ap.add_argument("-l", "--panel_image_path_band_2", required=False, help="panel image path band 2")
# ap.add_argument("-m", "--panel_image_path_band_3", required=False, help="panel image path band 3")
# ap.add_argument("-n", "--panel_image_path_band_4", required=False, help="panel image path band 4")
# ap.add_argument("-o", "--panel_image_path_band_5", required=False, help="panel image path band 5")
args = vars(ap.parse_args())

# input_image_bands = [args["image_path_band_1"], args["image_path_band_2"], args["image_path_band_3"], args["image_path_band_4"], args["image_path_band_5"]]
outpath_aligned_image_bands = [args["outpath_aligned_image_path_band_1"], args["outpath_aligned_image_path_band_2"], args["outpath_aligned_image_path_band_3"], args["outpath_aligned_image_path_band_4"], args["outpath_aligned_image_path_band_5"]]
# panel_image_bands = [args["panel_image_path_band_1"], args["panel_image_path_band_2"], args["panel_image_path_band_3"], args["panel_image_path_band_4"], args["panel_image_path_band_5"]]

panelNames = None

# imagePath = "Downloads/MicasenseExample5AlignmentImages"
# imageNames = glob.glob(os.path.join(imagePath,'IMG_0085_*.jpg'))

# imagePath = "Downloads/rededge-m"
# imageNames = glob.glob(os.path.join(imagePath,'img01_*.tif'))

imagePath = "Downloads/MicasenseTest/Panels"
imageNames = glob.glob(os.path.join(imagePath,'IMG_0432_*.tif'))

# imagePath = "Downloads/NickKExample5AlignmentImages"
# imageNames = glob.glob(os.path.join(imagePath,'IMG_0999_*.jpg'))

# for i in input_image_bands:
#     if i is not None:
#         imageNames.append(i)
# 
for i in outpath_aligned_image_bands:
    if i is not None:
        outpathNames.append(i)
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

    capture = capture.Capture.from_filelist(imageNames)

    if panelCap is not None:
        if panelCap.panel_albedo() is not None:
            panel_reflectance_by_band = panelCap.panel_albedo()
        else:
            panel_reflectance_by_band = [0.67, 0.69, 0.68, 0.61, 0.67] #RedEdge band_index order
        panel_irradiance = panelCap.panel_irradiance(panel_reflectance_by_band)    
        img_type = "reflectance"
    #    capture.plot_undistorted_reflectance(panel_irradiance)
    else:
        if capture.dls_present():
            img_type='reflectance'
    #        capture.plot_undistorted_reflectance(capture.dls_irradiance())
        else:
            img_type = "radiance"
    #        capture.plot_undistorted_radiance()

    ## Alignment settings
    match_index = 0 # Index of the band 
    max_alignment_iterations = 1000
    warp_mode = cv2.MOTION_HOMOGRAPHY # MOTION_HOMOGRAPHY or MOTION_AFFINE. For Altum images only use HOMOGRAPHY
    pyramid_levels = None # for images with RigRelatives, setting this to 0 or 1 may improve alignment

    print("Alinging images. Depending on settings this can take from a few seconds to many minutes")
    # Can potentially increase max_iterations for better results, but longer runtimes
    warp_matrices, alignment_pairs = imageutils.align_capture(capture,
                                                              ref_index = match_index,
                                                              max_iterations = max_alignment_iterations,
                                                              warp_mode = warp_mode,
                                                              pyramid_levels = pyramid_levels)

    print("Finished Aligning, warp matrices={}".format(warp_matrices))

    cropped_dimensions, edges = imageutils.find_crop_bounds(capture, warp_matrices, warp_mode=warp_mode)
    im_aligned = imageutils.aligned_capture(capture, warp_matrices, warp_mode, cropped_dimensions, match_index, img_type=img_type)
    print(im_aligned.shape)

    plt.imsave(outpathNames[0], im_aligned[:, :, 0], cmap='gray')
    plt.imsave(outpathNames[1], im_aligned[:, :, 1], cmap='gray')
    plt.imsave(outpathNames[2], im_aligned[:, :, 2], cmap='gray')
    plt.imsave(outpathNames[3], im_aligned[:, :, 3], cmap='gray')
    plt.imsave(outpathNames[4], im_aligned[:, :, 4], cmap='gray')

if __name__ == '__main__':
    run()