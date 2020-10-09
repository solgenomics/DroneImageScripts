# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/CalculatePhenotypeAutoEncoderVegetationIndices.py

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
from sklearn.model_selection import train_test_split
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--log_file_path", required=False, help="file path to write log to. useful for using from the web interface")
ap.add_argument("-k", "--input_training_image_file", required=True, help="file path with stock_id and image paths for training autoencoder")
ap.add_argument("-i", "--input_image_file", required=True, help="file path with stock_id and image paths")
ap.add_argument("-r", "--outfile_path", required=True, help="file path where results will be saved")
ap.add_argument("-j", "--autoencoder_model_type", required=True, help="type of autoencoder model")
ap.add_argument("-o", "--output_encoded_images_file", required=True, help="file path with file paths for encoded stock images")
args = vars(ap.parse_args())

log_file_path = args["log_file_path"]
input_training_images_file = args["input_training_image_file"]
input_images_file = args["input_image_file"]
autoencoder_model_type = args["autoencoder_model_type"]
results_outfile = args["outfile_path"]
output_encoded_images_file = args["output_encoded_images_file"]

if sys.version_info[0] < 3:
    raise Exception("Must use Python3. Use python3 in your command line.")

if log_file_path is not None:
    sys.stderr = open(log_file_path, 'a')

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

result_file_lines = [
    ['stock_id', 'ndvi', 'ndre']
]

image_size = 96

def get_imagedatagenerator():
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        #rotation_range=20,
        #width_shift_range=0.05,
        #height_shift_range=0.05,
        #horizontal_flip=True,
        # vertical_flip=True,
        #brightness_range=[0.8,1.2]
    )
    return datagen

def build_autoencoder(width, height, depth, filters=(32, 64), latentDim=16):
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

input_training_image_file_data = pd.read_csv(input_training_images_file, sep="\t", header=0)
input_image_file_data = pd.read_csv(input_images_file, sep="\t", header=0)
output_encoded_images_file_data = pd.read_csv(output_encoded_images_file, sep="\t", header=0)

output_ndvi_images = []
output_ndre_images = []
for index, row in output_encoded_images_file_data.iterrows():
    stock_id = row[0]
    output_ndvi_image = row[1]
    output_ndre_image = row[2]
    output_ndvi_images.append(output_ndvi_image)
    output_ndre_images.append(output_ndre_image)

stock_ids = []
nir_images_data = []
stock_nir_images_data = []
red_images_data = []
stock_red_images_data = []
rededge_images_data = []
stock_rededge_images_data = []
for index, row in input_image_file_data.iterrows():
    stock_id = row[0]
    red_images = row[1]
    rededge_images = row[2]
    nir_images = row[3]

    stock_ids.append(stock_id)
    red_images_array = red_images.split(',')
    rededge_images_array = rededge_images.split(',')
    nir_images_array = nir_images.split(',')

    stock_nir_images_array = []
    for nir_img in nir_images_array:
        image = cv2.imread(nir_img, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (image_size, image_size)) / 255.0

        if (len(image.shape) == 2):
            empty_mat = np.ones(image.shape, dtype=image.dtype) * 0
            image = cv2.merge((image, empty_mat, empty_mat))

        nir_images_data.append(image)
        stock_nir_images_array.append(image)
    stock_nir_images_data.append(np.array(stock_nir_images_array))

    stock_red_images_array = []
    for red_img in red_images_array:
        image = cv2.imread(red_img, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (image_size, image_size)) / 255.0

        if (len(image.shape) == 2):
            empty_mat = np.ones(image.shape, dtype=image.dtype) * 0
            image = cv2.merge((image, empty_mat, empty_mat))

        red_images_data.append(image)
        stock_red_images_array.append(image)
    stock_red_images_data.append(np.array(stock_red_images_array))

    stock_rededge_images_array = []
    for rededge_img in rededge_images_array:
        image = cv2.imread(rededge_img, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (image_size, image_size)) / 255.0

        if (len(image.shape) == 2):
            empty_mat = np.ones(image.shape, dtype=image.dtype) * 0
            image = cv2.merge((image, empty_mat, empty_mat))

        rededge_images_data.append(image)
        stock_rededge_images_array.append(image)
    stock_rededge_images_data.append(np.array(stock_rededge_images_array))

nir_images_data = np.array(nir_images_data)
red_images_data = np.array(red_images_data)
rededge_images_data = np.array(rededge_images_data)
stock_nir_images_data = np.array(stock_nir_images_data)
stock_red_images_data = np.array(stock_red_images_data)
stock_rededge_images_data = np.array(stock_rededge_images_data)

training_stock_ids = []
training_nir_images_data = []
training_stock_nir_images_data = []
training_red_images_data = []
training_stock_red_images_data = []
training_rededge_images_data = []
training_stock_rededge_images_data = []
for index, row in input_training_image_file_data.iterrows():
    stock_id = row[0]
    red_images = row[1]
    rededge_images = row[2]
    nir_images = row[3]

    if pd.isnull(red_images):
        continue

    training_stock_ids.append(stock_id)
    training_red_images_array = red_images.split(',')
    training_rededge_images_array = rededge_images.split(',')
    training_nir_images_array = nir_images.split(',')

    training_stock_nir_images_array = []
    for nir_img in training_nir_images_array:
        image = cv2.imread(nir_img, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (image_size, image_size)) / 255.0

        if (len(image.shape) == 2):
            empty_mat = np.ones(image.shape, dtype=image.dtype) * 0
            image = cv2.merge((image, empty_mat, empty_mat))

        training_nir_images_data.append(image)
        training_stock_nir_images_array.append(image)
    training_stock_nir_images_data.append(np.array(training_stock_nir_images_array))

    training_stock_red_images_array = []
    for red_img in training_red_images_array:
        image = cv2.imread(red_img, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (image_size, image_size)) / 255.0

        if (len(image.shape) == 2):
            empty_mat = np.ones(image.shape, dtype=image.dtype) * 0
            image = cv2.merge((image, empty_mat, empty_mat))

        training_red_images_data.append(image)
        training_stock_red_images_array.append(image)
    training_stock_red_images_data.append(np.array(training_stock_red_images_array))

    training_stock_rededge_images_array = []
    for rededge_img in training_rededge_images_array:
        image = cv2.imread(rededge_img, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (image_size, image_size)) / 255.0

        if (len(image.shape) == 2):
            empty_mat = np.ones(image.shape, dtype=image.dtype) * 0
            image = cv2.merge((image, empty_mat, empty_mat))

        training_rededge_images_data.append(image)
        training_stock_rededge_images_array.append(image)
    training_stock_rededge_images_data.append(np.array(training_stock_rededge_images_array))

training_nir_images_data = np.array(training_nir_images_data)
training_red_images_data = np.array(training_red_images_data)
training_rededge_images_data = np.array(training_rededge_images_data)
training_stock_nir_images_data = np.array(training_stock_nir_images_data)
training_stock_red_images_data = np.array(training_stock_red_images_data)
training_stock_rededge_images_data = np.array(training_stock_rededge_images_data)

nir_datagen = get_imagedatagenerator()
nir_datagen.fit(training_nir_images_data)
training_nir_images_data = nir_datagen.standardize(training_nir_images_data)
nir_images_data = nir_datagen.standardize(nir_images_data)

red_datagen = get_imagedatagenerator()
red_datagen.fit(training_red_images_data)
training_red_images_data = red_datagen.standardize(training_red_images_data)
red_images_data = red_datagen.standardize(red_images_data)

rededge_datagen = get_imagedatagenerator()
rededge_datagen.fit(training_rededge_images_data)
training_rededge_images_data = rededge_datagen.standardize(training_rededge_images_data)
rededge_images_data = rededge_datagen.standardize(rededge_images_data)

(nir_encoder, nir_decoder, nir_autoencoder) = build_autoencoder(image_size, image_size, 3)
(red_encoder, red_decoder, red_autoencoder) = build_autoencoder(image_size, image_size, 3)
(rededge_encoder, rededge_decoder, rededge_autoencoder) = build_autoencoder(image_size, image_size, 3)

opt = Adam(lr=1e-3)
nir_autoencoder.compile(loss="mse", optimizer=opt)
red_autoencoder.compile(loss="mse", optimizer=opt)
rededge_autoencoder.compile(loss="mse", optimizer=opt)

(training_nir_train_images, training_nir_test_images, training_nir_train_images, training_nir_test_images) = train_test_split(training_nir_images_data, training_nir_images_data, test_size=0.2)
(nir_train_images, nir_test_images, nir_train_images, nir_test_images) = train_test_split(nir_images_data, nir_images_data, test_size=0.2)
(training_red_train_images, training_red_test_images, training_red_train_images, training_red_test_images) = train_test_split(training_red_images_data, training_red_images_data, test_size=0.2)
(red_train_images, red_test_images, red_train_images, red_test_images) = train_test_split(red_images_data, red_images_data, test_size=0.2)
(training_rededge_train_images, training_rededge_test_images, training_rededge_train_images, training_rededge_test_images) = train_test_split(training_rededge_images_data, training_rededge_images_data, test_size=0.2)
(rededge_train_images, rededge_test_images, rededge_train_images, rededge_test_images) = train_test_split(rededge_images_data, rededge_images_data, test_size=0.2)

#checkpoint = ModelCheckpoint(filepath=output_autoencoder_model_file_path, monitor='loss', verbose=1, save_best_only=True, mode='min', save_frequency=1, save_weights_only=False)
#callbacks_list = [checkpoint]

H_nir = nir_autoencoder.fit(
    training_nir_train_images, training_nir_train_images,
    validation_data=(training_nir_test_images, training_nir_test_images),
    epochs=25,
    batch_size=32,
    #callbacks=callbacks_list
)
#nir_decoded = nir_autoencoder.predict(nir_images_data)

H_red = red_autoencoder.fit(
    training_red_train_images, training_red_train_images,
    validation_data=(training_red_test_images, training_red_test_images),
    epochs=25,
    batch_size=32,
    #callbacks=callbacks_list
)
#red_decoded = red_autoencoder.predict(red_images_data)

H_rededge = rededge_autoencoder.fit(
    training_rededge_train_images, training_rededge_train_images,
    validation_data=(training_rededge_test_images, training_rededge_test_images),
    epochs=25,
    batch_size=32,
    #callbacks=callbacks_list
)
#rededge_decoded = rededge_autoencoder.predict(rededge_images_data)

stock_counter = 0
for stock_id in stock_ids:
    nir_images = stock_nir_images_data[stock_counter]
    nir_decoded = nir_autoencoder.predict(nir_images)

    red_images = stock_red_images_data[stock_counter]
    red_decoded = red_autoencoder.predict(red_images)

    rededge_images = stock_rededge_images_data[stock_counter]
    rededge_decoded = rededge_autoencoder.predict(rededge_images)

    ndvi = np.divide(nir_decoded[0][:,:,0] - red_decoded[0][:,:,0], nir_decoded[0][:,:,0] + red_decoded[0][:,:,0])
    ndvi[np.isnan(ndvi)] = 0
    ndvi = ndvi * 255
    ndvi = ndvi.astype(np.uint8)

    ndre = np.divide(nir_decoded[0][:,:,0] - rededge_decoded[0][:,:,0], nir_decoded[0][:,:,0] + rededge_decoded[0][:,:,0])
    ndre[np.isnan(ndre)] = 0
    ndre = ndre * 255
    ndre = ndre.astype(np.uint8)

    cv2.imwrite(output_ndvi_images[stock_counter], ndvi)
    cv2.imwrite(output_ndre_images[stock_counter], ndre)

    stock_counter += 1

    ndvi_list = []
    ndre_list = []

    img_counter = 0
    for image in nir_images:
        nir_img = nir_decoded[img_counter][:,:,0]
        red_img = red_decoded[img_counter][:,:,0]
        rededge_img = rededge_decoded[img_counter][:,:,0]

        img_counter += 1

        nir_mean_pixel = nir_img.mean(axis=0).mean()
        red_mean_pixel = red_img.mean(axis=0).mean()
        rededge_mean_pixel = rededge_img.mean(axis=0).mean()

        nir_mean_pixel = nir_mean_pixel / 255
        red_mean_pixel = red_mean_pixel / 255
        rededge_mean_pixel = rededge_mean_pixel / 255

        ndvi_list.append((nir_mean_pixel - red_mean_pixel)/(nir_mean_pixel + red_mean_pixel))
        ndre_list.append((nir_mean_pixel - rededge_mean_pixel)/(nir_mean_pixel + rededge_mean_pixel))

    result_file_lines.append([
        stock_id,
        statistics.mean(ndvi_list),
        statistics.mean(ndre_list)
    ])

with open(results_outfile, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(result_file_lines)

writeFile.close()
