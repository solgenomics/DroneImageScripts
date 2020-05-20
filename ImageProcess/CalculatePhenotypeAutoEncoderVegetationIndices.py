# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/CalculatePhenotypeAutoEncoderVegetationIndices.py --image_paths /folder/mypic1.png,/folder/mypic2.png --results_outfile_path /folder/myresults.csv --image_band_index 0 --plot_polygon_type observation_unit_polygon_vari_imagery --margin_percent 5

# import the necessary packages
import sys
import argparse
import imutils
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import statistics
from collections import defaultdict
import csv
import pandas as pd
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import backend as K

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_image_file", required=True, help="file path with stock_id and image paths")
ap.add_argument("-r", "--outfile_path", required=True, help="file path where results will be saved")
args = vars(ap.parse_args())

input_images_file = args["input_image_file"]
results_outfile = args["outfile_path"]

result_file_lines = [
    ['stock_id', 'nonzero_pixel_count', 'total_pixel_sum', 'mean_pixel_value', 'harmonic_mean_value', 'median_pixel_value', 'variance_pixel_value', 'stdev_pixel_value', 'pstdev_pixel_value', 'min_pixel_value', 'max_pixel_value', 'minority_pixel_value', 'minority_pixel_count', 'majority_pixel_value', 'majority_pixel_count', 'pixel_variety_count']
]

def build_autoencoder(self, width, height, depth, filters=(32, 64), latentDim=16):
    inputShape = (height, width, depth)
    chanDim = -1

    # define the input to the encoder
    inputs = Input(shape=inputShape)
    x = inputs

    # loop over the number of filters
    for f in filters:
        # apply a CONV => RELU => BN operation
        x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(axis=chanDim)(x)

    # flatten the network and then construct our latent vector
    volumeSize = K.int_shape(x)
    x = Flatten()(x)
    latent = Dense(latentDim)(x)

    # build the encoder model
    encoder = Model(inputs, latent, name="encoder")

    # start building the decoder model which will accept the
    # output of the encoder as its inputs
    latentInputs = Input(shape=(latentDim,))
    x = Dense(np.prod(volumeSize[1:]))(latentInputs)
    x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

    # loop over our number of filters again, but this time in
    # reverse order
    for f in filters[::-1]:
        # apply a CONV_TRANSPOSE => RELU => BN operation
        x = Conv2DTranspose(f, (3, 3), strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(axis=chanDim)(x)

    # apply a single CONV_TRANSPOSE layer used to recover the
    # original depth of the image
    x = Conv2DTranspose(depth, (3, 3), padding="same")(x)
    outputs = Activation("sigmoid")(x)

    # build the decoder model
    decoder = Model(latentInputs, outputs, name="decoder")

    # our autoencoder is the encoder + decoder
    autoencoder = Model(inputs, decoder(encoder(inputs)), name="autoencoder")

    # return a 3-tuple of the encoder, decoder, and autoencoder
    return (encoder, decoder, autoencoder)

input_image_file_data = pd.read_csv(input_images_file, sep="\t", header=None)

nir_images = []
red_images = []
rededge_images = []
for index, row in input_image_file_data.iterrows():
    stock_id = row[0]
    red_images = row[1]
    rededge_images = row[1]
    nir_images = row[1]
    red_images_array = red_images.split(',')
    rededge_images_array = rededge_images.split(',')
    nir_images_array = nir_images.split(',')

    nir_images.append(nir_images_array)
    red_images.append(red_images_array)
    rededge_images.append(rededge_images_array)

    non_zero_list = []
    total_pixel_sum_list = []
    mean_pixel_value_list = []
    harmonic_mean_pixel_value_list = []
    pixel_median_value_list = []
    pixel_variance_list = []
    pixel_standard_dev_list = []
    pixel_pstandard_dev_list = []
    min_pixel_list = []
    max_pixel_list = []
    minority_pixel_value_list = []
    minority_pixel_count_list = []
    majority_pixel_value_list = []
    majority_pixel_count_list = []
    pixel_group_count_list = []

    for image in images_array:
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        img_shape = img.shape

        if len(img_shape) == 3:
            if img_shape[2] == 3:
                b,g,r = cv2.split(img)
                if image_band_index == 0:
                    img = b
                if image_band_index == 1:
                    img = g
                if image_band_index == 2:
                    img = r

        non_zero = cv2.countNonZero(img)
        #print("Nonzero: %s" % non_zero)

        original_height, original_width = img.shape
        width_margin = margin_percent * original_width
        height_margin = margin_percent * original_height

        img = crop(img, [[{'x':width_margin, 'y':height_margin}, {'x':original_width-width_margin, 'y':height_margin}, {'x':original_width-width_margin, 'y':original_height-height_margin}, {'x':width_margin, 'y':original_height-height_margin}]])

        height, width = img.shape

        total_pixel_sum = 0
        pixel_array = []
        pixel_dict = defaultdict(int)

        for i in range(0, height):
            for j in range(0, width):
                px = int(img[i,j])
                total_pixel_sum += px
                pixel_array.append(px)
                pixel_dict[px] += 1

        #print("Total: %s" % total_pixel_sum)

        mean_pixel_value = statistics.mean(pixel_array)
        #print("Mean: %s" % mean_pixel_value)

        if sys.version_info >= (3,6,0):
            harmonic_mean_pixel_value = statistics.harmonic_mean(pixel_array) #required python >= 3.6
        else:
            harmonic_mean_pixel_value = 0
        #print("Harmonic Mean: %s" % harmonic_mean_pixel_value)

        pixel_array_np = np.array(pixel_array)
        pixel_array_sort = np.sort(pixel_array_np)
        pixel_median_value = statistics.median(pixel_array_sort)
        #print("Median: %s" % pixel_median_value)

        pixel_variance = statistics.variance(pixel_array)
        #print("Variance: %s" % pixel_variance)

        pixel_standard_dev = statistics.stdev(pixel_array)
        #print("Stdev: %s" % pixel_standard_dev)

        pixel_pstandard_dev = statistics.pstdev(pixel_array)
        #print("Pstdev %s" % pixel_pstandard_dev)

        min_pixel = pixel_array_sort[0]
        max_pixel = pixel_array_sort[-1]
        #print("Min: %s" % min_pixel)
        #print("Max: %s" % max_pixel)

        pixel_sorted_by_value = sorted(pixel_dict.items(), key=lambda kv: kv[1])
        minority_pixel = pixel_sorted_by_value[0]
        majority_pixel = pixel_sorted_by_value[-1]
        minority_pixel_value = minority_pixel[0]
        minority_pixel_count = minority_pixel[1]
        majority_pixel_value = majority_pixel[0]
        majority_pixel_count = majority_pixel[1]
        #print("Minority: %s" % minority_pixel_value)
        #print("Minority Count: %s" % minority_pixel_count)
        #print("Majority: %s" % majority_pixel_value)
        #print("Majority Count: %s" % majority_pixel_count)

        pixel_group_count = len(pixel_dict)
        #print("Variety: %s" % pixel_group_count)

        #cv2.imshow('image'+str(count),kpsimage)
        #cv2.imwrite(outfiles[count], kpsimage)

        total_pixel_sum = total_pixel_sum / 255
        mean_pixel_value = mean_pixel_value / 255
        harmonic_mean_pixel_value = harmonic_mean_pixel_value / 255
        pixel_median_value = pixel_median_value / 255
        pixel_variance = pixel_variance / 255
        pixel_standard_dev = pixel_standard_dev / 255
        pixel_pstandard_dev = pixel_pstandard_dev / 255
        min_pixel = min_pixel / 255
        max_pixel = max_pixel / 255
        minority_pixel_value = minority_pixel_value / 255
        majority_pixel_value = majority_pixel_value / 255

        non_zero_list.append(non_zero)
        total_pixel_sum_list.append(total_pixel_sum)
        mean_pixel_value_list.append(mean_pixel_value)
        harmonic_mean_pixel_value_list.append(harmonic_mean_pixel_value)
        pixel_median_value_list.append(pixel_median_value)
        pixel_variance_list.append(pixel_variance)
        pixel_standard_dev_list.append(pixel_standard_dev)
        pixel_pstandard_dev_list.append(pixel_pstandard_dev)
        min_pixel_list.append(min_pixel)
        max_pixel_list.append(max_pixel)
        minority_pixel_value_list.append(minority_pixel_value)
        minority_pixel_count_list.append(minority_pixel_count)
        majority_pixel_value_list.append(majority_pixel_value)
        majority_pixel_count_list.append(majority_pixel_count)
        pixel_group_count_list.append(pixel_group_count)

    result_file_lines.append([
        stock_id,
        statistics.mean(non_zero_list),
        statistics.mean(total_pixel_sum_list),
        statistics.mean(mean_pixel_value_list),
        statistics.mean(harmonic_mean_pixel_value_list),
        statistics.mean(pixel_median_value_list),
        statistics.mean(pixel_variance_list),
        statistics.mean(pixel_standard_dev_list),
        statistics.mean(pixel_pstandard_dev_list),
        statistics.mean(min_pixel_list),
        statistics.mean(max_pixel_list),
        statistics.mean(minority_pixel_value_list),
        statistics.mean(minority_pixel_count_list),
        statistics.mean(majority_pixel_value_list),
        statistics.mean(majority_pixel_count_list),
        statistics.mean(pixel_group_count_list)
    ])

with open(results_outfile, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(result_file_lines)

writeFile.close()
