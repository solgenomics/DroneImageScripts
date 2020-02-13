# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/MicasenseRawImagePlotBoundaries.py --file_with_image_paths /folder/myimages.csv --file_with_panel_image_paths  /folder/mypanels.csv --output_path /export/myresults.csv --field_layout_path layout.csv --field_layout_params params.csv

# import the necessary packages
def run():
    import sys
    import os, glob
    import argparse
    import csv
    import imutils
    import cv2
    import numpy as np
    import math
    import json
    import random
    import matplotlib.pyplot as plt
    from multiprocessing import Process, freeze_support
    from PIL import Image
    import micasense.imageutils as imageutils
    import micasense.plotutils as plotutils
    from micasense.image import Image
    from micasense.panel import Panel
    import micasense.utils as msutils
    from micasense.capture import Capture
    import pickle

    freeze_support()

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--log_file_path", required=False, help="file path to write log to. useful for using from the web interface")
    ap.add_argument("-i", "--file_with_image_paths", required=True, help="file with file paths to the Micasense images in order")
    ap.add_argument("-p", "--file_with_panel_image_paths", required=True, help="file with file paths to the Micasense panel images in order")
    ap.add_argument("-a", "--field_layout_path", required=True, help="file with field layout")
    ap.add_argument("-r", "--field_layout_params", required=True, help="file with layout params")
    ap.add_argument("-o", "--output_path", required=True, help="file path where the output will be saved")
    ap.add_argument("-u", "--temporary_development_path", required=False, help="file path for saving warp matrices. only useful for development")
    args = vars(ap.parse_args())

    log_file_path = args["log_file_path"]
    file_with_image_paths = args["file_with_image_paths"]
    file_with_panel_image_paths = args["file_with_panel_image_paths"]
    field_layout_path = args["field_layout_path"]
    field_layout_params = args["field_layout_params"]
    output_path = args["output_path"]
    temporary_development_path = args["temporary_development_path"]

    if sys.version_info[0] < 3:
        raise Exception("Must use Python3. Use python3 in your command line.")

    if log_file_path is not None:
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

    field_layout = []
    with open(field_layout_path) as fp:
        for line in fp:
            plot_id, plot_name, plot_number = line.strip().split(",")
            field_layout.append([plot_id, plot_name, plot_number])

    field_params = []
    with open(field_layout_params) as fp:
        for line in fp:
            param = line.strip()
            field_params.append(param)

    first_plot_corner = field_params[0] #north_west, north_east, south_west, south_east
    second_plot_direction = field_params[1] #north_to_south, south_to_north, east_to_west, west_to_east
    plot_orientation = field_params[2] #serpentine, zigzag
    corners_obj = json.loads(field_params[3])
    corner_gps_obj = json.loads(field_params[4])
    rotate_angle = float(field_params[5])
    num_rows = int(field_params[6])
    num_columns = int(field_params[7])
    flight_direction = field_params[8] #rows, columns, #DEPRECATED
    plot_width_m = float(field_params[9])
    plot_length_m = float(field_params[10])
    plot_corners_pixels = json.loads(field_params[11])
    gps_precision_to_mm = float(field_params[12])
    start_direction = field_params[13] #north_to_south, south_to_north, east_to_west, west_to_east
    turn_direction = field_params[14] #north_to_south, south_to_north, east_to_west, west_to_east
    geographic_position = field_params[15] #Q1, Q2, Q3, Q4
    image_top_direction = field_params[16] #north, south, east, west
    # row_alley_width_m = float(field_params[17])
    # column_alley_width_m = float(field_params[18])

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

    if log_file_path is not None:
        eprint("Finished Aligning, warp matrices={}".format(warp_matrices))
    else:
        print("Finished Aligning, warp matrices={}".format(warp_matrices))

    rotated_imgs = []
    img_gps_locations = []
    counter = 0
    for x in captures:
        im_aligned = x.create_aligned_capture(
            irradiance_list = panel_irradiance,
            warp_matrices = warp_matrices,
            match_index = match_index,
            warp_mode = warp_mode
        )

        img = imageNameMatchIndexImages[counter]
        latitude = img.latitude
        longitude = img.longitude
        altitude = img.altitude
        # GSD resolution for Micasenes camera in m/p
        img_gps_locations.append([latitude, longitude, altitude, 0.06857*100*altitude/10])

        rows,cols,d = im_aligned.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),rotate_angle,1)
        rotated_img = cv2.warpAffine(im_aligned,M,(cols,rows))

        if log_file_path is not None:
            eprint(rotated_img.shape)
        else:
            print(rotated_img.shape)

        rotated_imgs.append(rotated_img)
        
        counter += 1

    img_rows_pixels, img_columns_pixels, d = rotated_imgs[0].shape
    img_rows_pixels_half = img_columns_pixels/2
    img_columns_pixels_half = img_rows_pixels/2
    print(img_rows_pixels_half)
    print(img_columns_pixels_half)

    plot_width_1_pixel_nw = int(plot_corners_pixels['north_west'][1]['x']) - int(plot_corners_pixels['north_west'][0]['x'])
    plot_width_1_pixel_ne = int(plot_corners_pixels['north_east'][1]['x']) - int(plot_corners_pixels['north_east'][0]['x'])
    plot_width_1_pixel_sw = int(plot_corners_pixels['south_west'][1]['x']) - int(plot_corners_pixels['south_west'][0]['x'])
    plot_width_1_pixel_se = int(plot_corners_pixels['south_east'][1]['x']) - int(plot_corners_pixels['south_east'][0]['x'])
    plot_width_2_pixel_nw = int(plot_corners_pixels['north_west'][2]['x']) - int(plot_corners_pixels['north_west'][3]['x'])
    plot_width_2_pixel_ne = int(plot_corners_pixels['north_east'][2]['x']) - int(plot_corners_pixels['north_east'][3]['x'])
    plot_width_2_pixel_sw = int(plot_corners_pixels['south_west'][2]['x']) - int(plot_corners_pixels['south_west'][3]['x'])
    plot_width_2_pixel_se = int(plot_corners_pixels['south_east'][2]['x']) - int(plot_corners_pixels['south_east'][3]['x'])
    plot_length_1_pixel_nw = int(plot_corners_pixels['north_west'][2]['y']) - int(plot_corners_pixels['north_west'][1]['y'])
    plot_length_1_pixel_ne = int(plot_corners_pixels['north_east'][2]['y']) - int(plot_corners_pixels['north_east'][1]['y'])
    plot_length_1_pixel_sw = int(plot_corners_pixels['south_west'][2]['y']) - int(plot_corners_pixels['south_west'][1]['y'])
    plot_length_1_pixel_se = int(plot_corners_pixels['south_east'][2]['y']) - int(plot_corners_pixels['south_east'][1]['y'])
    plot_length_2_pixel_nw = int(plot_corners_pixels['north_west'][3]['y']) - int(plot_corners_pixels['north_west'][0]['y'])
    plot_length_2_pixel_ne = int(plot_corners_pixels['north_east'][3]['y']) - int(plot_corners_pixels['north_east'][0]['y'])
    plot_length_2_pixel_sw = int(plot_corners_pixels['south_west'][3]['y']) - int(plot_corners_pixels['south_west'][0]['y'])
    plot_length_2_pixel_se = int(plot_corners_pixels['south_east'][3]['y']) - int(plot_corners_pixels['south_east'][0]['y'])

    plot_width_pixel_avg = int((plot_width_1_pixel_nw + plot_width_1_pixel_ne + plot_width_1_pixel_sw + plot_width_1_pixel_se + plot_width_2_pixel_nw + plot_width_2_pixel_ne + plot_width_2_pixel_sw + plot_width_2_pixel_se)/8)
    plot_length_pixel_avg = int((plot_length_1_pixel_nw + plot_length_1_pixel_ne + plot_length_1_pixel_sw + plot_length_1_pixel_se + plot_length_2_pixel_nw + plot_length_2_pixel_ne + plot_length_2_pixel_sw + plot_length_2_pixel_se)/8)
    print(plot_width_pixel_avg)
    print(plot_length_pixel_avg)

    plot_width_pixels_per_m = plot_width_pixel_avg/plot_width_m
    plot_length_pixels_per_m = plot_length_pixel_avg/plot_length_m
    print(plot_width_pixels_per_m)
    print(plot_length_pixels_per_m)

    gps_precision_to_mm = gps_precision_to_mm * 10

    plot_pixels_per_gps_width = int(plot_width_pixels_per_m * gps_precision_to_mm * 1000)
    plot_pixels_per_gps_length = int(plot_length_pixels_per_m * gps_precision_to_mm * 1000)
    print(plot_pixels_per_gps_width)
    print(plot_pixels_per_gps_length)

    column_width_gps = 0
    column_height_gps = 0
    row_width_gps = 0
    row_height_gps = 0
    column_width_pixels = 0
    column_height_pixels = 0
    row_width_pixels = 0
    row_height_pixels = 0
    latitude_to_pixel_sign = 1
    longitude_to_pixel_sign = 1

    nw_pixel_x_diff = int(corners_obj['north_west']['x']) - img_rows_pixels_half
    nw_pixel_y_diff = int(corners_obj['north_west']['y']) - img_columns_pixels_half
    ne_pixel_x_diff = int(corners_obj['north_east']['x']) - img_rows_pixels_half
    ne_pixel_y_diff = int(corners_obj['north_east']['y']) - img_columns_pixels_half
    sw_pixel_x_diff = int(corners_obj['south_west']['x']) - img_rows_pixels_half
    sw_pixel_y_diff = int(corners_obj['south_west']['y']) - img_columns_pixels_half
    se_pixel_x_diff = int(corners_obj['south_east']['x']) - img_rows_pixels_half
    se_pixel_y_diff = int(corners_obj['south_east']['y']) - img_columns_pixels_half

    def distance(lat1, lon1, lat2, lon2):
        p = 0.017453292519943295
        a = 0.5 - math.cos((lat2-lat1)*p)/2 + math.cos(lat1*p)*math.cos(lat2*p) * (1-math.cos((lon2-lon1)*p)) / 2
        return 12742 * math.asin(math.sqrt(a))

    def distances(data, v):
        distances = []
        for d in data:
            distances.append(distance(d[0], d[1], v['lat'], v['lon']))
        return distances

    def min_distance(data, v):
        d = distances(data, v)
        val, idx = min((val, idx) for (idx, val) in enumerate(d))
        return (val, idx)

    def crop_polygon(input_image, polygon):
        pts_array = []
        for point in polygon:
            x = point['x']
            y = point['y']

            x = int(round(x))
            y = int(round(y))
            pts_array.append([x,y])

        pts = np.array(pts_array)
        rect = cv2.boundingRect(pts)
        x,y,w,h = rect
        finalImage = input_image[y:y+h, x:x+w, :]
        return finalImage

    plot_polygons_gps = []
    plot_polygons_pixels = []

    output_lines = []
    print(corner_gps_obj)

    # Q1 is north of 0 and west of 0 e.g. North America
    if geographic_position == 'Q1':

        if image_top_direction == 'north':

            latitude_to_pixel_sign = -1
            latitude_to_pixel_sign = -1

            if first_plot_corner == 'north_west' and second_plot_direction == 'west_to_east':
                # field_nw_longitude_gps = float(corner_gps_obj['north_west'][1]) + (nw_pixel_x_diff*longitude_to_pixel_sign*float(corner_gps_obj['north_west'][3])/gps_precision_to_mm)
                # field_nw_latitude_gps = float(corner_gps_obj['north_west'][0]) + (nw_pixel_y_diff*latitude_to_pixel_sign*float(corner_gps_obj['north_west'][3])/gps_precision_to_mm)
                # field_ne_longitude_gps = float(corner_gps_obj['north_east'][1]) + (ne_pixel_x_diff*longitude_to_pixel_sign*float(corner_gps_obj['north_east'][3])/gps_precision_to_mm)
                # field_ne_latitude_gps = float(corner_gps_obj['north_east'][0]) + (ne_pixel_y_diff*latitude_to_pixel_sign*float(corner_gps_obj['north_east'][3])/gps_precision_to_mm)
                # field_sw_longitude_gps = float(corner_gps_obj['south_west'][1]) + (sw_pixel_x_diff*longitude_to_pixel_sign*float(corner_gps_obj['south_west'][3])/gps_precision_to_mm)
                # field_sw_latitude_gps = float(corner_gps_obj['south_west'][0]) + (sw_pixel_y_diff*latitude_to_pixel_sign*float(corner_gps_obj['south_west'][3])/gps_precision_to_mm)
                # field_se_longitude_gps = float(corner_gps_obj['south_east'][1]) + (se_pixel_x_diff*longitude_to_pixel_sign*float(corner_gps_obj['south_east'][3])/gps_precision_to_mm)
                # field_se_latitude_gps = float(corner_gps_obj['south_east'][0]) + (se_pixel_y_diff*latitude_to_pixel_sign*float(corner_gps_obj['south_east'][3])/gps_precision_to_mm)

                field_nw_longitude_gps = float(corner_gps_obj['north_west'][1]) + (nw_pixel_x_diff*longitude_to_pixel_sign/plot_pixels_per_gps_width)
                field_nw_latitude_gps = float(corner_gps_obj['north_west'][0]) + (nw_pixel_y_diff*latitude_to_pixel_sign/plot_pixels_per_gps_length)
                field_ne_longitude_gps = float(corner_gps_obj['north_east'][1]) + (ne_pixel_x_diff*longitude_to_pixel_sign/plot_pixels_per_gps_width)
                field_ne_latitude_gps = float(corner_gps_obj['north_east'][0]) + (ne_pixel_y_diff*latitude_to_pixel_sign/plot_pixels_per_gps_length)
                field_sw_longitude_gps = float(corner_gps_obj['south_west'][1]) + (sw_pixel_x_diff*longitude_to_pixel_sign/plot_pixels_per_gps_width)
                field_sw_latitude_gps = float(corner_gps_obj['south_west'][0]) + (sw_pixel_y_diff*latitude_to_pixel_sign/plot_pixels_per_gps_length)
                field_se_longitude_gps = float(corner_gps_obj['south_east'][1]) + (se_pixel_x_diff*longitude_to_pixel_sign/plot_pixels_per_gps_width)
                field_se_latitude_gps = float(corner_gps_obj['south_east'][0]) + (se_pixel_y_diff*latitude_to_pixel_sign/plot_pixels_per_gps_length)
    
                plot_width_top_gps = (field_ne_longitude_gps - field_nw_longitude_gps)/num_columns
                plot_width_bottom_gps = (field_se_longitude_gps - field_sw_longitude_gps)/num_columns
                plot_width_gps_avg = (plot_width_top_gps + plot_width_bottom_gps)/2

                plot_length_left_gps = (field_nw_latitude_gps - field_sw_latitude_gps)/num_rows
                plot_length_right_gps = (field_ne_latitude_gps - field_se_latitude_gps)/num_rows
                plot_length_gps_avg = (plot_length_left_gps + plot_length_right_gps)/2
                
        if image_top_direction == 'west':

            # field_nw_longitude_gps = float(corner_gps_obj['north_west'][1]) + (nw_pixel_y_diff*longitude_to_pixel_sign*float(corner_gps_obj['north_west'][3])/gps_precision_to_mm)
            # field_nw_latitude_gps = float(corner_gps_obj['north_west'][0]) + (nw_pixel_x_diff*latitude_to_pixel_sign*float(corner_gps_obj['north_west'][3])/gps_precision_to_mm)
            # field_ne_longitude_gps = float(corner_gps_obj['north_east'][1]) + (ne_pixel_y_diff*longitude_to_pixel_sign*float(corner_gps_obj['north_east'][3])/gps_precision_to_mm)
            # field_ne_latitude_gps = float(corner_gps_obj['north_east'][0]) + (ne_pixel_x_diff*latitude_to_pixel_sign*float(corner_gps_obj['north_east'][3])/gps_precision_to_mm)
            # field_sw_longitude_gps = float(corner_gps_obj['south_west'][1]) + (sw_pixel_y_diff*longitude_to_pixel_sign*float(corner_gps_obj['south_west'][3])/gps_precision_to_mm)
            # field_sw_latitude_gps = float(corner_gps_obj['south_west'][0]) + (sw_pixel_x_diff*latitude_to_pixel_sign*float(corner_gps_obj['south_west'][3])/gps_precision_to_mm)
            # field_se_longitude_gps = float(corner_gps_obj['south_east'][1]) + (se_pixel_y_diff*longitude_to_pixel_sign*float(corner_gps_obj['south_east'][3])/gps_precision_to_mm)
            # field_se_latitude_gps = float(corner_gps_obj['south_east'][0]) + (se_pixel_x_diff*latitude_to_pixel_sign*float(corner_gps_obj['south_east'][3])/gps_precision_to_mm)

            field_nw_longitude_gps = float(corner_gps_obj['north_west'][1]) + (nw_pixel_y_diff*longitude_to_pixel_sign/plot_pixels_per_gps_length)
            field_nw_latitude_gps = float(corner_gps_obj['north_west'][0]) + (nw_pixel_x_diff*latitude_to_pixel_sign/plot_pixels_per_gps_width)
            field_ne_longitude_gps = float(corner_gps_obj['north_east'][1]) + (ne_pixel_y_diff*longitude_to_pixel_sign/plot_pixels_per_gps_length)
            field_ne_latitude_gps = float(corner_gps_obj['north_east'][0]) + (ne_pixel_x_diff*latitude_to_pixel_sign/plot_pixels_per_gps_width)
            field_sw_longitude_gps = float(corner_gps_obj['south_west'][1]) + (sw_pixel_y_diff*longitude_to_pixel_sign/plot_pixels_per_gps_length)
            field_sw_latitude_gps = float(corner_gps_obj['south_west'][0]) + (sw_pixel_x_diff*latitude_to_pixel_sign/plot_pixels_per_gps_width)
            field_se_longitude_gps = float(corner_gps_obj['south_east'][1]) + (se_pixel_y_diff*longitude_to_pixel_sign/plot_pixels_per_gps_length)
            field_se_latitude_gps = float(corner_gps_obj['south_east'][0]) + (se_pixel_x_diff*latitude_to_pixel_sign/plot_pixels_per_gps_width)

            plot_width_top_gps = (field_nw_latitude_gps - field_sw_latitude_gps)/num_columns
            plot_width_bottom_gps = (field_ne_latitude_gps - field_se_latitude_gps)/num_columns
            plot_width_gps_avg = (plot_width_top_gps + plot_width_bottom_gps)/2
            print(plot_width_gps_avg)

            plot_length_left_gps = (field_ne_longitude_gps - field_nw_longitude_gps)/num_rows
            plot_length_right_gps = (field_se_longitude_gps - field_sw_longitude_gps)/num_rows
            plot_length_gps_avg = (plot_length_left_gps + plot_length_right_gps)/2
            print(plot_length_gps_avg)

            plot_total_vertical_shift_gps = ((field_nw_longitude_gps - field_sw_longitude_gps) + (field_ne_longitude_gps - field_se_longitude_gps))/2
            plot_vertical_shift_avg_gps = plot_total_vertical_shift_gps/num_columns
            print(plot_vertical_shift_avg_gps)

            plot_horizontal_shift_left_gps = (field_sw_latitude_gps - field_se_latitude_gps)/num_rows
            plot_horizontal_shift_right_gps = (field_nw_latitude_gps - field_ne_latitude_gps)/num_rows
            plot_horizontal_shift_avg_gps = (plot_horizontal_shift_left_gps + plot_horizontal_shift_right_gps)/2
            print(plot_horizontal_shift_avg_gps)

            if first_plot_corner == 'north_west' and second_plot_direction == 'north_to_south' and plot_orientation == 'zigzag':
                x_pos = field_nw_latitude_gps - plot_width_gps_avg
                y_pos = field_nw_longitude_gps

                plot_width_fix = 10
                plot_length_fix = 10

                plot_counter = 1
                row_num = 1
                #Visualize the GPS on http://www.copypastemap.com/index.html
                dumper_str = ''
                colors = ['red', 'blue', 'green', 'yellow', 'white']
                for i in range(0, num_rows, 1):
                    for j in range(0, num_columns, 1):
                        x_pos_val = x_pos
                        y_pos_val = y_pos
                        plot_polygons_gps.append([
                            {'lat':x_pos_val, 'lon':y_pos_val},
                            {'lat':x_pos_val + plot_width_gps_avg, 'lon':y_pos_val},
                            {'lat':x_pos_val + plot_width_gps_avg, 'lon':y_pos_val + plot_length_gps_avg},
                            {'lat':x_pos_val, 'lon':y_pos_val + plot_length_gps_avg}
                        ])

                        color = random.choice(colors)
                        dumper_str = dumper_str + str(x_pos_val)+'\t'+str(y_pos_val)+'\tnumbered\t'+color+'\t'+str(plot_counter)+'\n'
                        dumper_str = dumper_str + str(x_pos_val + plot_width_gps_avg)+'\t'+str(y_pos_val)+'\tnumbered\t'+color+'\t'+str(plot_counter)+'\n'
                        dumper_str = dumper_str + str(x_pos_val + plot_width_gps_avg)+'\t'+str(y_pos_val + plot_length_gps_avg)+'\tnumbered\t'+color+'\t'+str(plot_counter)+'\n'
                        dumper_str = dumper_str + str(x_pos_val)+'\t'+str(y_pos_val + plot_length_gps_avg)+'\tnumbered\t'+color+'\t'+str(plot_counter)+'\n'

                        x_pos = x_pos - plot_width_gps_avg
                        y_pos = y_pos + plot_vertical_shift_avg_gps
                        plot_counter += 1
                    x_pos = field_nw_latitude_gps - plot_width_gps_avg + (row_num * plot_horizontal_shift_avg_gps)
                    # y_pos = y_pos + plot_length_gps_avg + plot_total_vertical_shift_gps
                    y_pos = y_pos + plot_length_gps_avg - plot_total_vertical_shift_gps
                    row_num = row_num + 1
                print(dumper_str)

                x_offset_pixels = 0
                y_offset_pixels = 0

                counter = 0
                for p in plot_polygons_gps:
                    #Find image closest to plot GPS
                    img_distance, img_index = min_distance(img_gps_locations, p[0])
                    img = rotated_imgs[img_index]
                    img_gps = img_gps_locations[img_index]

                    # polygon = [{
                    #     'x':img_rows_pixels_half + (p[0]['lat'] - img_gps[0])*gps_precision_to_mm/img_gps[3],
                    #     'y':img_columns_pixels_half - abs(p[0]['lon'] - img_gps[1])*gps_precision_to_mm/img_gps[3]
                    # },
                    # {
                    #     'x':img_rows_pixels_half + abs(p[1]['lat'] - img_gps[0])*gps_precision_to_mm/img_gps[3],
                    #     'y':img_columns_pixels_half - abs(p[1]['lon'] - img_gps[1])*gps_precision_to_mm/img_gps[3]
                    # },
                    # {
                    #     'x':img_rows_pixels_half + abs(p[2]['lat'] - img_gps[0])*gps_precision_to_mm/img_gps[3],
                    #     'y':img_columns_pixels_half - abs(p[2]['lon'] - img_gps[1])*gps_precision_to_mm/img_gps[3]
                    # },
                    # {
                    #     'x':img_rows_pixels_half + abs(p[3]['lat'] - img_gps[0])*gps_precision_to_mm/img_gps[3],
                    #     'y':img_columns_pixels_half - abs(p[3]['lon'] - img_gps[1])*gps_precision_to_mm/img_gps[3]
                    # }]

                    polygon = [{
                        'x':img_rows_pixels_half + x_offset_pixels + abs(p[0]['lat'] - img_gps[0])*plot_pixels_per_gps_width,
                        'y':img_columns_pixels_half + y_offset_pixels - abs(p[0]['lon'] - img_gps[1])*plot_pixels_per_gps_length
                    },
                    {
                        'x':img_rows_pixels_half + x_offset_pixels + abs(p[1]['lat'] - img_gps[0])*plot_pixels_per_gps_width,
                        'y':img_columns_pixels_half + y_offset_pixels - abs(p[1]['lon'] - img_gps[1])*plot_pixels_per_gps_length
                    },
                    {
                        'x':img_rows_pixels_half + x_offset_pixels + abs(p[2]['lat'] - img_gps[0])*plot_pixels_per_gps_width,
                        'y':img_columns_pixels_half + y_offset_pixels - abs(p[2]['lon'] - img_gps[1])*plot_pixels_per_gps_length
                    },
                    {
                        'x':img_rows_pixels_half + x_offset_pixels + abs(p[3]['lat'] - img_gps[0])*plot_pixels_per_gps_width,
                        'y':img_columns_pixels_half + y_offset_pixels - abs(p[3]['lon'] - img_gps[1])*plot_pixels_per_gps_length
                    }]

                    plot_polygons_pixels.append({
                        'img_index':img_index,
                        'p':polygon
                    })

                    plot_stack = crop_polygon(img, polygon)

                    blue_img = cv2.rectangle(img[:,:,0]*255, (int(polygon[0]['x']), int(polygon[0]['y'])), (int(polygon[2]['x']), int(polygon[2]['y'])), (0,0,0), 1)
                    green_img = cv2.rectangle(img[:,:,1]*255, (int(polygon[0]['x']), int(polygon[0]['y'])), (int(polygon[2]['x']), int(polygon[2]['y'])), (0,0,0), 1)
                    red_img = cv2.rectangle(img[:,:,2]*255, (int(polygon[0]['x']), int(polygon[0]['y'])), (int(polygon[2]['x']), int(polygon[2]['y'])), (0,0,0), 1)
                    nir_img = cv2.rectangle(img[:,:,3]*255, (int(polygon[0]['x']), int(polygon[0]['y'])), (int(polygon[2]['x']), int(polygon[2]['y'])), (0,0,0), 1)
                    red_edge_img = cv2.rectangle(img[:,:,4]*255, (int(polygon[0]['x']), int(polygon[0]['y'])), (int(polygon[2]['x']), int(polygon[2]['y'])), (0,0,0), 1)

                    blue_img_file = imageTempNamesBlue[counter]
                    green_img_file = imageTempNamesGreen[counter]
                    red_img_file = imageTempNamesRed[counter]
                    nir_img_file = imageTempNamesNIR[counter]
                    red_edge_img_file = imageTempNamesRedEdge[counter]

                    cv2.imwrite(basePath+blue_img_file, blue_img)
                    cv2.imwrite(basePath+green_img_file, green_img)
                    cv2.imwrite(basePath+red_img_file, red_img)
                    cv2.imwrite(basePath+nir_img_file, nir_img)
                    cv2.imwrite(basePath+red_edge_img_file, red_edge_img)

                    # plt.imsave(basePath+blue_img_file, plot_stack[:,:,0], cmap='gray')
                    # plt.imsave(basePath+green_img_file, plot_stack[:,:,1], cmap='gray')
                    # plt.imsave(basePath+red_img_file, plot_stack[:,:,2], cmap='gray')
                    # plt.imsave(basePath+nir_img_file, plot_stack[:,:,3], cmap='gray')
                    # plt.imsave(basePath+red_edge_img_file, plot_stack[:,:,4], cmap='gray')

                    if counter in range(-len(field_layout), len(field_layout)):
                        plot_database = field_layout[counter]
                        output_lines.append([plot_database[0], plot_database[1], plot_database[2], blue_img_file, green_img_file, red_img_file, nir_img_file, red_edge_img_file, json.dumps(polygon)])

                    counter += 1

    print(plot_width_gps_avg)
    print(plot_length_gps_avg)
    print(plot_polygons_gps)
    print(len(plot_polygons_gps))
    print(plot_polygons_pixels)
    print(len(plot_polygons_pixels))
    print(output_lines)

    with open(output_path, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(output_lines)

    writeFile.close()

if __name__ == '__main__':
    run()

