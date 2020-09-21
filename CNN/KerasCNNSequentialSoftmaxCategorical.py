# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/CNN/KerasCNNSequentialSoftmaxCategorical.py --input_image_label_file  /folder/myimagesandlabels.csv --output_model_file_path /folder/mymodel.h5 --outfile_path /export/myresults.csv

# import the necessary packages
import sys
import argparse
import csv
import imutils
import cv2
import numpy as np
import math
import pandas as pd
import CNNProcessData
import tensorflow
from PIL import Image
from sklearn import preprocessing
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras import backend
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import inception_resnet_v2
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import densenet
from tensorflow.keras.applications import VGG16
from kerastuner.tuners import RandomSearch

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--log_file_path", required=False, help="file path to write log to. useful for using from the web interface")
ap.add_argument("-i", "--input_image_label_file", required=True, help="file path for file holding image names and labels to be trained. It is assumed that the input_image_label_file is ordered by field trial, then by the drone runs in chronological ascending order, then by the plots, then by the image types. For LSTM models, it is assumed that the input_image_label_file is ordered by the field trial, then by plots, then image types, then drone runs in chronological ascending order. The number of time points (chronological order) is only actively useful when using time-series (LSTM) CNNs. Contains the following header: stock_id,image_path,image_type,day,drone_run_project_id,value")
ap.add_argument("-a", "--input_aux_data_file", required=True, help="file path for aux data containing the following header: stock_id,value,trait_name,field_trial_id,accession_id,female_id,male_id")
ap.add_argument("-m", "--output_model_file_path", required=True, help="file path for saving keras model, so that it can be loaded again in the future. it saves an hdf5 file as the model")
ap.add_argument("-e", "--output_autoencoder_model_file_path", required=True, help="file path for saving keras autoencoder model for filtering images, so that it can be loaded again in the future. it saves an hdf5 file as the model")
ap.add_argument("-o", "--outfile_path", required=True, help="file path where the output will be saved")
ap.add_argument("-f", "--output_loss_history", required=True, help="file path where the output for loss history during training will be saved")
ap.add_argument("-k", "--keras_model_type", required=True, help="type of keras model to train: densenet121_lstm_imagenet, simple_1, inceptionresnetv2, inceptionresnetv2application, densenet121application, simple_1_tuner, simple_tuner")
ap.add_argument("-w", "--keras_model_weights", required=False, help="the name of the pre-trained Keras CNN model weights to use e.g. imagenet for the InceptionResNetV2 model. Leave empty to instantiate the model with random weights")
ap.add_argument("-n", "--keras_model_layers", required=False, help="the first X layers to use from a pre-trained Keras CNN model e.g. 10 for the first 10 layers from the InceptionResNetV2 model")
ap.add_argument("-p", "--output_random_search_result_project", required=False, help="project dir name where the keras tuner random search results output will be saved. only required for tuner models")
ap.add_argument("-g", "--grm_input_file", required=False, help="file path input for the genetic relationship matrix to use")
args = vars(ap.parse_args())

log_file_path = args["log_file_path"]
input_file = args["input_image_label_file"]
input_aux_data_file = args["input_aux_data_file"]
output_model_file_path = args["output_model_file_path"]
output_autoencoder_model_file_path = args["output_autoencoder_model_file_path"]
outfile_path = args["outfile_path"]
output_loss_history = args["output_loss_history"]
output_random_search_result_project = args["output_random_search_result_project"]
keras_model_type = args["keras_model_type"]
keras_model_weights = args["keras_model_weights"]
keras_model_layers = args["keras_model_layers"]
grm_input_file = args["grm_input_file"]

image_size = 96
montage_image_size = image_size*2

if sys.version_info[0] < 3:
    raise Exception("Must use Python3. Use python3 in your command line.")

if log_file_path is not None:
    sys.stderr = open(log_file_path, 'a')

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def conv2d(x,numfilt,filtsz,strides=1,pad='same',act=True,name=None):
    x = Conv2D(numfilt,filtsz,strides=strides,padding=pad,data_format='channels_last',use_bias=False,name=name+'conv2d')(x)
    x = BatchNormalization(axis=3,scale=False,name=name+'conv2d'+'bn')(x)
    if act:
        x = Activation('relu',name=name+'conv2d'+'act')(x)
    return x

def incresA(x,scale,name=None):
    pad = 'same'
    branch0 = conv2d(x,32,1,1,pad,True,name=name+'b0')
    branch1 = conv2d(x,32,1,1,pad,True,name=name+'b1_1')
    branch1 = conv2d(branch1,32,3,1,pad,True,name=name+'b1_2')
    branch2 = conv2d(x,32,1,1,pad,True,name=name+'b2_1')
    branch2 = conv2d(branch2,48,3,1,pad,True,name=name+'b2_2')
    branch2 = conv2d(branch2,64,3,1,pad,True,name=name+'b2_3')
    branches = [branch0,branch1,branch2]
    mixed = Concatenate(axis=3, name=name + '_concat')(branches)
    filt_exp_1x1 = conv2d(mixed,384,1,1,pad,False,name=name+'filt_exp_1x1')
    final_lay = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=backend.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=name+'act_scaling')([x, filt_exp_1x1])
    return final_lay

def incresB(x,scale,name=None):
    pad = 'same'
    branch0 = conv2d(x,192,1,1,pad,True,name=name+'b0')
    branch1 = conv2d(x,128,1,1,pad,True,name=name+'b1_1')
    branch1 = conv2d(branch1,160,[1,7],1,pad,True,name=name+'b1_2')
    branch1 = conv2d(branch1,192,[7,1],1,pad,True,name=name+'b1_3')
    branches = [branch0,branch1]
    mixed = Concatenate(axis=3, name=name + '_mixed')(branches)
    filt_exp_1x1 = conv2d(mixed,1152,1,1,pad,False,name=name+'filt_exp_1x1')
    final_lay = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=backend.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=name+'act_scaling')([x, filt_exp_1x1])
    return final_lay

def incresC(x,scale,name=None):
    pad = 'same'
    branch0 = conv2d(x,192,1,1,pad,True,name=name+'b0')
    branch1 = conv2d(x,192,1,1,pad,True,name=name+'b1_1')
    branch1 = conv2d(branch1,224,[1,3],1,pad,True,name=name+'b1_2')
    branch1 = conv2d(branch1,256,[3,1],1,pad,True,name=name+'b1_3')
    branches = [branch0,branch1]
    mixed = Concatenate(axis=3, name=name + '_mixed')(branches)
    filt_exp_1x1 = conv2d(mixed,2048,1,1,pad,False,name=name+'fin1x1')
    final_lay = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=backend.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=name+'act_saling')([x, filt_exp_1x1])
    return final_lay

def build_simple_model(hp):
    img_input = Input(shape=(montage_image_size,montage_image_size,3))

    x = Conv2D(
        filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
        kernel_size=hp.Choice('conv_1_kernel', values = [3,5]),
        activation='relu'
    )(img_input)
    x = Conv2D(
        filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),
        kernel_size=hp.Choice('conv_2_kernel', values = [3,5]),
        activation='relu'
    )(x)
    x = Flatten()(x)
    x = Dense(
        units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),
        activation='relu'
    )(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(4)(x)
    x = Activation("relu")(x)

    x = Dense(1, activation="linear")(x)

    model = Model(img_input, x)
    model.compile(loss="mean_absolute_percentage_error",optimizer=Adam(hp.Choice('learning_rate', values=[1e-3])))
    return model

def build_simple_1_model(hp):
    init = "he_normal"
    reg = regularizers.l2(0.01)
    chanDim = -1

    img_input = Input(shape=(montage_image_size,montage_image_size,3))

    x = Conv2D(
        filters=hp.Int('conv_1_filter', min_value=16, max_value=32, step=16),
        kernel_size=hp.Choice('conv_1_kernel', values = [3,5,7]),
        activation='relu',
        strides=(2, 2),
        padding="valid",
        kernel_initializer=init,
        kernel_regularizer=reg
    )(img_input)
    x = Conv2D(
        filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),
        kernel_size=hp.Choice('conv_2_kernel', values = [3,5]),
        activation='relu',
        padding="same",
        kernel_initializer=init,
        kernel_regularizer=reg
    )(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Conv2D(
        filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),
        kernel_size=hp.Choice('conv_2_kernel', values = [3,5]),
        strides=(2, 2),
        activation='relu',
        padding="same",
        kernel_initializer=init,
        kernel_regularizer=reg
    )(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.25)(x)
    x = Conv2D(
        filters=hp.Int('conv_2_filter', min_value=64, max_value=128, step=32),
        kernel_size=hp.Choice('conv_2_kernel', values = [3,5]),
        activation='relu',
        padding="same",
        kernel_initializer=init,
        kernel_regularizer=reg
    )(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Conv2D(
        filters=hp.Int('conv_2_filter', min_value=64, max_value=128, step=32),
        kernel_size=hp.Choice('conv_2_kernel', values = [3,5]),
        strides=(2, 2),
        activation='relu',
        padding="same",
        kernel_initializer=init,
        kernel_regularizer=reg
    )(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.25)(x)
    x = Conv2D(
        filters=hp.Int('conv_2_filter', min_value=128, max_value=256, step=64),
        kernel_size=hp.Choice('conv_2_kernel', values = [3,5]),
        activation='relu',
        padding="same",
        kernel_initializer=init,
        kernel_regularizer=reg
    )(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Conv2D(
        filters=hp.Int('conv_2_filter', min_value=128, max_value=256, step=64),
        kernel_size=hp.Choice('conv_2_kernel', values = [3,5]),
        strides=(2, 2),
        activation='relu',
        padding="same",
        kernel_initializer=init,
        kernel_regularizer=reg
    )(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(
        units=hp.Int('dense_1_units', min_value=256, max_value=512, step=64),
        activation='relu'
    )(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)
    
    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)

    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(4)(x)
    x = Activation("relu")(x)

    x = Dense(1, activation="linear")(x)

    model = Model(img_input, x)
    model.compile(loss="mean_absolute_percentage_error", optimizer=Adam(hp.Choice('learning_rate', values=[1e-3])))
    return model

def create_mlp(dim, regress = False):
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation="relu"))
    model.add(Dense(4, activation="relu"))
    if regress:
        model.add(Dense(1, activation="linear"))
    return model

def create_cnn_example(width, height, depth, filters=(16, 32, 64), regress=False):
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    chanDim = -1

    inputs = Input(shape=inputShape)

    for (i, f) in enumerate(filters):
        if i == 0:
            x = inputs

        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)

    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(4)(x)
    x = Activation("relu")(x)

    if regress:
        x = Dense(1, activation="linear")(x)

    model = Model(inputs, x)
    return model

class LossHistory(tensorflow.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

unique_stock_ids = {}
unique_time_days = {}
unique_labels = {}
unique_image_types = {}
unique_germplasm = {}
unique_female_parents = {}
unique_male_parents = {}
unique_drone_run_project_ids = {}
labels = []
data = []
labels_time_series = []
data_time_series = []

# def data_augment_rotate(angle, image):
#     (h, w) = image.shape[:2]
#     center = (w / 2, h / 2)
#     scale = 1.0
#     M = cv2.getRotationMatrix2D(center, angle, scale)
#     rotated = cv2.warpAffine(image, M, (h, w))
#     return rotated

print("[INFO] reading labels and image data...")

csv_data = pd.read_csv(input_file, sep=",", header=0, index_col=False, usecols=['stock_id','image_path','image_type','day','drone_run_project_id','value'])
for index, row in csv_data.iterrows():
    image_path = row['image_path']
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    value = float(row['value'])

    image = cv2.resize(image, (image_size,image_size)) / 255.0

    if (len(image.shape) == 2):
        empty_mat = np.ones(image.shape, dtype=image.dtype) * 0
        image = cv2.merge((image, empty_mat, empty_mat))

    data.append(image)

unique_stock_ids = csv_data.stock_id.unique()
unique_time_days = csv_data.day.unique()
unique_drone_run_project_ids = csv_data.drone_run_project_id.unique()
unique_image_types = csv_data.image_type.unique()

aux_data_cols = ["stock_id","value","trait_name","field_trial_id","accession_id","female_id","male_id","output_image_file","genotype_file"]
aux_data_trait_cols = [col for col in csv_data.columns if 'aux_trait_' in col]
aux_data_cols = aux_data_cols.extend(aux_data_trait_cols)
aux_data = pd.read_csv(input_aux_data_file, sep=",", header=0, index_col=False, usecols=aux_data_cols)

if log_file_path is not None:
    eprint(csv_data)
    eprint(aux_data)
else:
    print(csv_data)
    print(aux_data)

unique_labels = aux_data.value.unique()
unique_germplasm = aux_data.accession_id.unique()
unique_female_parents = aux_data.female_id.unique()
unique_male_parents = aux_data.male_id.unique()

labels = aux_data["value"].tolist()

num_unique_stock_ids = len(unique_stock_ids)
num_unique_image_types = len(unique_image_types)
num_unique_time_days = len(unique_time_days)
num_unique_drone_run_project_ids = len(unique_drone_run_project_ids)
if len(data) % num_unique_stock_ids or len(labels) % num_unique_stock_ids:
    raise Exception('Number of images or labels does not divide evenly among stock_ids. This means the input data is uneven.')
if keras_model_type == 'densenet121_lstm_imagenet' and ( num_unique_stock_ids * num_unique_time_days * num_unique_image_types != len(data) or num_unique_stock_ids * num_unique_time_days * num_unique_image_types != len(labels) ):
    print(num_unique_stock_ids)
    print(num_unique_time_days)
    print(num_unique_image_types)
    print(len(data))
    print(len(labels))
    raise Exception('Number of rows in input file (images and labels) is not equal to the number of unique stocks times the number of unique time points times the number of unique image types. This means the input data in uneven for a LSTM model')

lines = []
class_map_lines = []
history_loss_lines = []

# germplasmBinarizer = LabelBinarizer().fit(unique_germplasm)
# trainCategorical = zipBinarizer.transform(train["zipcode"])

if len(unique_labels) < 2:
    print("Number of unique labels less than 2!")
    lines = ["Number of labels is less than 2, so nothing to predict!"]
else:
    separator = ","
    labels_string = separator.join([str(x) for x in labels])
    unique_labels_string = separator.join([str(x) for x in unique_labels])
    if log_file_path is not None:
        eprint("Labels " + str(len(labels)) + ": " + labels_string)
        eprint("Unique Labels " + str(len(unique_labels)) + ": " + unique_labels_string)
    else:
        print("Labels " + str(len(labels)) + ": " + labels_string)
        print("Unique Labels " + str(len(unique_labels)) + ": " + unique_labels_string)

    if log_file_path is not None:
        eprint("[INFO] number of labels: %d" % (len(labels)))
        eprint("[INFO] number of images: %d" % (len(data)))
        eprint("[INFO] augmenting data and splitting training set...")
    else:
        print("[INFO] number of labels: %d" % (len(labels)))
        print("[INFO] number of images: %d" % (len(data)))
        print("[INFO] augmenting data and splitting training set...")

    data = np.array(data)
    labels = np.array(labels)

    data_augmentation = 1
    data_augmentation_test = 1
    montage_image_number = 4 # Implemented to combine 4 different image types of the same plot into a single montage image
    process_data = CNNProcessData.CNNProcessData()
    print(aux_data.shape)
    print(data.shape)
    (testImages, testX, testY, testGenotypes, trainImages, trainX, trainY, trainGenotypes) = process_data.process_cnn_data(data, aux_data, num_unique_stock_ids, num_unique_image_types, num_unique_time_days, image_size, keras_model_type, data_augmentation, data_augmentation_test, montage_image_number, montage_image_size, output_autoencoder_model_file_path, log_file_path)
    print(testImages.shape)
    print(testX.shape)
    print(testY.shape)
    print(trainImages.shape)
    print(trainX.shape)
    print(trainY.shape)

    if log_file_path is not None:
        eprint("[INFO] number of augmented training images: %d" % (len(trainImages)))
        eprint("[INFO] number of augmented testing images: %d" % (len(testImages)))
        eprint("[INFO] number of augmented aux data: %d" % (len(trainX)))
        eprint("[INFO] number of augmented aux data: %d" % (len(testX)))
        eprint("[INFO] number of augmented training labels: %d" % (len(trainY)))
        eprint("[INFO] number of augmented testing labels: %d" % (len(testY)))
    else:
        print("[INFO] number of augmented training images: %d" % (len(trainImages)))
        print("[INFO] number of augmented testing images: %d" % (len(testImages)))
        print("[INFO] number of augmented aux data: %d" % (len(trainX)))
        print("[INFO] number of augmented aux data: %d" % (len(testX)))
        print("[INFO] number of augmented training labels: %d" % (len(trainY)))
        print("[INFO] number of augmented testing labels: %d" % (len(testY)))

    model = None
    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    initial_epoch = 0
    epochs = 100

    print(keras_model_type)
    if keras_model_type == 'inceptionresnetv2':
        img_input = Input(shape=(montage_image_size,montage_image_size,3))

        #STEM
        x = conv2d(img_input,32,3,2,'valid',True,name='conv1')
        x = conv2d(x,32,3,1,'valid',True,name='conv2')
        x = conv2d(x,64,3,1,'valid',True,name='conv3')

        x_11 = MaxPooling2D(3,strides=1,padding='valid',name='stem_br_11'+'_maxpool_1', data_format="channels_last")(x)
        x_12 = conv2d(x,64,3,1,'valid',True,name='stem_br_12')

        x = Concatenate(axis=3, name = 'stem_concat_1')([x_11,x_12])

        x_21 = conv2d(x,64,1,1,'same',True,name='stem_br_211')
        x_21 = conv2d(x_21,64,[1,7],1,'same',True,name='stem_br_212')
        x_21 = conv2d(x_21,64,[7,1],1,'same',True,name='stem_br_213')
        x_21 = conv2d(x_21,96,3,1,'valid',True,name='stem_br_214')

        x_22 = conv2d(x,64,1,1,'same',True,name='stem_br_221')
        x_22 = conv2d(x_22,96,3,1,'valid',True,name='stem_br_222')

        x = Concatenate(axis=3, name = 'stem_concat_2')([x_21,x_22])

        x_31 = conv2d(x,192,3,1,'valid',True,name='stem_br_31')
        x_32 = MaxPooling2D(3,strides=1,padding='valid',name='stem_br_32'+'_maxpool_2', data_format="channels_last")(x)
        x = Concatenate(axis=3, name = 'stem_concat_3')([x_31,x_32])

        #Inception-ResNet-A modules
        x = incresA(x,0.15,name='incresA_1')
        x = incresA(x,0.15,name='incresA_2')
        x = incresA(x,0.15,name='incresA_3')
        x = incresA(x,0.15,name='incresA_4')

        #35 × 35 to 17 × 17 reduction module.
        x_red_11 = MaxPooling2D(3,strides=2,padding='valid',name='red_maxpool_1', data_format="channels_last")(x)

        x_red_12 = conv2d(x,384,3,2,'valid',True,name='x_red1_c1')

        x_red_13 = conv2d(x,256,1,1,'same',True,name='x_red1_c2_1')
        x_red_13 = conv2d(x_red_13,256,3,1,'same',True,name='x_red1_c2_2')
        x_red_13 = conv2d(x_red_13,384,3,2,'valid',True,name='x_red1_c2_3')

        x = Concatenate(axis=3, name='red_concat_1')([x_red_11,x_red_12,x_red_13])

        #Inception-ResNet-B modules
        x = incresB(x,0.1,name='incresB_1')
        x = incresB(x,0.1,name='incresB_2')
        x = incresB(x,0.1,name='incresB_3')
        x = incresB(x,0.1,name='incresB_4')
        x = incresB(x,0.1,name='incresB_5')
        x = incresB(x,0.1,name='incresB_6')
        x = incresB(x,0.1,name='incresB_7')

        #17 × 17 to 8 × 8 reduction module.
        x_red_21 = MaxPooling2D(3,strides=2,padding='valid',name='red_maxpool_2', data_format="channels_last")(x)

        x_red_22 = conv2d(x,256,1,1,'same',True,name='x_red2_c11')
        x_red_22 = conv2d(x_red_22,384,3,2,'valid',True,name='x_red2_c12')

        x_red_23 = conv2d(x,256,1,1,'same',True,name='x_red2_c21')
        x_red_23 = conv2d(x_red_23,256,3,2,'valid',True,name='x_red2_c22')

        x_red_24 = conv2d(x,256,1,1,'same',True,name='x_red2_c31')
        x_red_24 = conv2d(x_red_24,256,3,1,'same',True,name='x_red2_c32')
        x_red_24 = conv2d(x_red_24,256,3,2,'valid',True,name='x_red2_c33')

        x = Concatenate(axis=3, name='red_concat_2')([x_red_21,x_red_22,x_red_23,x_red_24])

        #Inception-ResNet-C modules
        x = incresC(x,0.2,name='incresC_1')
        x = incresC(x,0.2,name='incresC_2')
        x = incresC(x,0.2,name='incresC_3')

        # flatten the volume, then FC => RELU => BN => DROPOUT
        x = Flatten()(x)
        x = Dense(16)(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Dropout(0.5)(x)

        # apply another FC layer, this one to match the number of nodes
        # coming out of the MLP
        x = Dense(4)(x)
        x = Activation("relu")(x)

        x = Dense(1, activation="linear")(x)
        model = Model(img_input, x)
        model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

    if keras_model_type == 'simple_1':

        init = "he_normal"
        reg = regularizers.l2(0.01)
        chanDim = -1

        img_input = Input(shape=(montage_image_size,montage_image_size,3))

        x = Conv2D(16, (7, 7), strides=(2, 2), padding="valid", kernel_initializer=init, kernel_regularizer=reg)(img_input)
        x = Conv2D(32, (3, 3), padding="same", kernel_initializer=init, kernel_regularizer=reg)(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Conv2D(32, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init, kernel_regularizer=reg)(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Dropout(0.25)(x)

        # stack two more CONV layers, keeping the size of each filter
        # as 3x3 but increasing to 64 total learned filters
        x = Conv2D(64, (3, 3), padding="same", kernel_initializer=init, kernel_regularizer=reg)(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Conv2D(64, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init, kernel_regularizer=reg)(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Dropout(0.25)(x)

        # increase the number of filters again, this time to 128
        x = Conv2D(128, (3, 3), padding="same", kernel_initializer=init, kernel_regularizer=reg)(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Conv2D(128, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init, kernel_regularizer=reg)(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Dropout(0.25)(x)

        # fully-connected layer
        x = Flatten()(x)
        x = Dense(512, kernel_initializer=init)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        # flatten the volume, then FC => RELU => BN => DROPOUT
        x = Flatten()(x)
        x = Dense(16)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        # apply another FC layer, this one to match the number of nodes
        # coming out of the MLP
        x = Dense(4)(x)
        x = Activation("relu")(x)

        x = Dense(1, activation="linear")(x)

        model = Model(img_input, x)
        model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

    if keras_model_type == 'densenet121application':
        trainX = densenet.preprocess_input(trainX)
        testX = densenet.preprocess_input(testX)

        input_tensor = Input(shape=(montage_image_size,montage_image_size,3))
        base_model = DenseNet121(
            include_top = False,
            weights = keras_model_weights,
            input_tensor = input_tensor,
            input_shape = (montage_image_size,montage_image_size,3)
        )

        # flatten the volume, then FC => RELU => BN => DROPOUT
        x = Flatten()(base_model.output)
        x = Dense(16)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        # apply another FC layer, this one to match the number of nodes
        # coming out of the MLP
        x = Dense(4)(x)
        x = Activation("relu")(x)

        x = Dense(1, activation="linear")(x)
        model = Model(input_tensor, x)
        model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

    if keras_model_type == 'inceptionresnetv2application':
        trainX = inception_resnet_v2.preprocess_input(trainX)
        testX = inception_resnet_v2.preprocess_input(testX)

        input_tensor = Input(shape=(montage_image_size,montage_image_size,3))
        base_model = InceptionResNetV2(
            include_top = False,
            weights = keras_model_weights,
            input_tensor = input_tensor,
            input_shape = (montage_image_size,montage_image_size,3),
            pooling = 'avg'
        )

        # flatten the volume, then FC => RELU => BN => DROPOUT
        x = Flatten()(base_model.output)
        x = Dense(16)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        # apply another FC layer, this one to match the number of nodes
        # coming out of the MLP
        x = Dense(4)(x)
        x = Activation("relu")(x)

        x = Dense(1, activation="linear")(x)
        model = Model(input_tensor, x)
        model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

    if keras_model_type == 'densenet121_lstm_imagenet':
        trainX = densenet.preprocess_input(trainX)
        testX = densenet.preprocess_input(testX)

        n = DenseNet121(
            include_top=False,
            weights='imagenet',
            input_shape=(montage_image_size, montage_image_size, 3),
            pooling = 'avg'
        )

        # do not train first layers, I want to only train
        # the 4 last layers (my own choice, up to you)
        for layer in n.layers[:-2]:
            layer.trainable = False

        model = Sequential()
        model.add(
            TimeDistributed(n, input_shape=(num_unique_time_days, montage_image_size, montage_image_size, 3))
        )
        model.add(
            TimeDistributed(
                Flatten()
            )
        )
        model.add(LSTM(256, activation='relu', return_sequences=False))

        # flatten the volume, then FC => RELU => BN => DROPOUT
        model.add(Flatten())
        model.add(Dense(16))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # apply another FC layer, this one to match the number of nodes
        # coming out of the MLP
        model.add(Dense(4))
        model.add(Activation("relu"))

        model.add(Dense(1, activation="linear"))

        model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

    if keras_model_type == 'simple_tuner':
        tuner = RandomSearch(
            build_simple_model,
            objective='loss',
            max_trials=100,
            directory=output_random_search_result_project,
            project_name=output_random_search_result_project
        )
        initial_epoch = 5
        tuner.search(trainX, trainY, epochs=initial_epoch, validation_split=0.1)
        model = tuner.get_best_models(num_models=1)[0]

    if keras_model_type == 'simple_1_tuner':
        tuner = RandomSearch(
            build_simple_1_model,
            objective='loss',
            max_trials=100,
            directory=output_random_search_result_project,
            project_name=output_random_search_result_project
        )
        initial_epoch = 5
        tuner.search(trainX, trainY, epochs=initial_epoch, validation_split=0.1)
        model = tuner.get_best_models(num_models=1)[0]

    # if keras_model_type == 'inceptionresnetv2application_tuner':
        # train_images = inception_resnet_v2.preprocess_input(train_images)
        # test_images = inception_resnet_v2.preprocess_input(test_images)
        # 
        # # hypermodel = HyperResNet(input_shape=(image_size, image_size, 3), classes=NUM_LABELS, weights=keras_model_weights)
        # hypermodel = HyperResNet(input_shape=(image_size, image_size, 3), classes=NUM_LABELS)
        # tuner = Hyperband(
        #     hypermodel,
        #     objective='val_accuracy',
        #     directory=output_random_search_result_project,
        #     project_name=output_random_search_result_project,
        #     max_epochs=50
        # )
        # tuner.search(trainX, trainY, epochs=5, validation_split=0.1)
        # model = tuner.get_best_models(num_models=1)[0]

    if keras_model_type == 'mlp_cnn_example':
        mlp = create_mlp(trainGenotypes.shape[1], regress=False)
        mlp2 = create_mlp(trainX.shape[1], regress=False)
        cnn = create_cnn_example(montage_image_size, montage_image_size, 3, regress=False)
        combinedInput = concatenate([mlp.output, mlp2.output, cnn.output])
        x = Dense(4, activation="relu")(combinedInput)
        x = Dense(1, activation="linear")(x)
        model = Model(inputs=[mlp.input, mlp2.input, cnn.input], outputs=x)
        opt = Adam(lr=1e-3, decay=1e-3 / 200)
        model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

        trainImages = [trainGenotypes, trainX, trainImages]
        testImages = [testGenotypes, testX, testImages]
        epochs = 25

    for layer in model.layers:
        if log_file_path is not None:
            eprint(layer.output_shape)
        else:
            print(layer.output_shape)

    print("[INFO] training network...")

    checkpoint = ModelCheckpoint(filepath=output_model_file_path, monitor='loss', verbose=1, save_best_only=True, mode='min', save_frequency=1, save_weights_only=False)
    es = EarlyStopping(monitor='loss', mode='min', min_delta=0.01, patience=35, verbose=1)
    history = LossHistory()
    callbacks_list = [history, es, checkpoint]

    H = model.fit(trainImages, trainY, validation_data=(testImages, testY), epochs=epochs, batch_size=16, initial_epoch=initial_epoch, callbacks=callbacks_list)

    for h in history.losses:
        history_loss_lines.append([h])

    # H = model.fit_generator(
    #     generator = datagen.flow(trainX, trainY, batch_size=batch_size),
    #     validation_data = (testX, testY),
    #     steps_per_epoch = len(trainX) // (batch_size / 8),
    #     epochs = 150,
    #     callbacks = callbacks_list
    # )

    # print("[INFO] evaluating network...")
    # predictions = model.predict(testX, batch_size=32)
    # report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_)
    # print(report)
    # 
    # report_lines = report.split('\n')
    # separator = ""
    # for l in report_lines:
    #     lines.append(separator.join(l))

#print(lines)
with open(outfile_path, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(lines)
writeFile.close()

with open(output_loss_history, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(history_loss_lines)
writeFile.close()
