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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import VGG16

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--log_file_path", required=False, help="file path to write log to. useful for using from the web interface")
ap.add_argument("-i", "--input_image_label_file", required=True, help="file path for file holding image names and labels to be trained. It is assumed that the input_image_label_file is ordered by the plots and the time points in ascending order. The number of time points is only useful when using time-series (LSTM) CNNs.")
ap.add_argument("-m", "--output_model_file_path", required=True, help="file path for saving keras model, so that it can be loaded again in the future. it saves an hdf5 file as the model")
ap.add_argument("-o", "--outfile_path", required=True, help="file path where the output will be saved")
ap.add_argument("-c", "--output_class_map", required=True, help="file path where the output for class map will be saved")
ap.add_argument("-k", "--keras_model_type", required=True, help="type of keras model to train: densenet121_lstm, simple_1, inceptionresnetv2, inceptionresnetv2application, densenet121")
ap.add_argument("-w", "--keras_model_weights", required=False, help="the name of the pre-trained Keras CNN model weights to use e.g. imagenet for the InceptionResNetV2 model. Leave empty to instantiate the model with random weights")
ap.add_argument("-n", "--keras_model_layers", required=False, help="the first X layers to use from a pre-trained Keras CNN model e.g. 10 for the first 10 layers from the InceptionResNetV2 model")
args = vars(ap.parse_args())

log_file_path = args["log_file_path"]
input_file = args["input_image_label_file"]
output_model_file_path = args["output_model_file_path"]
outfile_path = args["outfile_path"]
output_class_map = args["output_class_map"]
keras_model_type = args["keras_model_type"]
keras_model_weights = args["keras_model_weights"]
keras_model_layers = args["keras_model_layers"]

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

unique_stock_ids = {}
unique_time_days = {}
unique_labels = {}
unique_image_types = {}
unique_drone_run_band_names = {}
labels = []
data = []
labels_time_series = []
data_time_series = []

image_size = 75
if keras_model_type == 'simple_1':
    image_size = 32
if keras_model_type == 'densenet121_lstm':
    image_size = 75
if keras_model_type == 'inceptionresnetv2':
    image_size = 75
if keras_model_type == 'inceptionresnetv2application':
    image_size = 75
if keras_model_type == 'densenet121application':
    image_size = 75

print("[INFO] reading labels and image data...")
with open(input_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        stock_id = row[0]
        trait_name = row[3]
        image_type = row[4]
        time_days = row[6]
        
        image = Image.open(row[1])
        image = np.array(image.resize((image_size,image_size))) / 255.0

        if (len(image.shape) == 2):
            empty_mat = np.ones(image.shape, dtype=image.dtype) * 0
            image = cv2.merge((image, empty_mat, empty_mat))

        #print(image.shape)
        data.append(image)

        value = float(row[2])
        labels.append(value)

        if value in unique_labels.keys():
            unique_labels[str(value)] += 1
        else:
            unique_labels[str(value)] = 1

        if image_type in unique_image_types.keys():
            unique_image_types[image_type] += 1
        else:
            unique_image_types[image_type] = 1

        if drone_run_band_name in unique_drone_run_band_names.keys():
            unique_drone_run_band_names[drone_run_band_name] += 1
        else:
            unique_drone_run_band_names[drone_run_band_name] = 1

        if stock_id in unique_stock_ids.keys():
            unique_stock_ids[stock_id] += 1
        else:
            unique_stock_ids[stock_id] = 1

        if time_days in unique_time_days.keys():
            unique_time_days[time_days] += 1
        else:
            unique_time_days[time_days] = 1

num_unique_stock_ids = len(unique_stock_ids.keys())
num_unique_image_types = len(unique_image_types.keys())
num_unique_drone_run_band_names = len(unique_drone_run_band_names.keys())
num_unique_time_days = len(unique_time_days.keys())
if num_unique_stock_ids * num_unique_time_days * num_unique_image_types != len(data) or num_unique_stock_ids * num_unique_time_days * num_unique_image_types != len(labels):
    print(num_unique_stock_ids)
    print(num_unique_time_days)
    print(num_unique_image_types)
    print(num_unique_drone_run_band_names)
    print(len(data))
    print(len(labels))
    raise Exception('Number of rows in input file (images and labels) is not equal to the number of unique stocks times the number of unique time points times the number of unique image types. This means the input data in uneven')

lines = []
class_map_lines = []
if len(unique_labels.keys()) < 2:
    lines = ["Number of labels is less than 2, so nothing to predict!"]
else:
    separator = ","
    labels_string = separator.join([str(x) for x in labels])
    unique_labels_string = separator.join([str(x) for x in unique_labels.keys()])
    if log_file_path is not None:
        eprint("Labels " + str(len(labels)) + ": " + labels_string)
        eprint("Unique Labels " + str(len(unique_labels.keys())) + ": " + unique_labels_string)
    else:
        print("Labels " + str(len(labels)) + ": " + labels_string)
        print("Unique Labels " + str(len(unique_labels.keys())) + ": " + unique_labels_string)

    categorical_object = pd.cut(labels, 25)
    labels_predict_codes = categorical_object.codes
    categories = categorical_object.categories

    labels_predict_map = {}
    labels_predict_unique = {}
    for index in range(len(labels)):
        label = labels[index]
        label_code = labels_predict_codes[index]
        cat_mid = categories[label_code].mid
        labels_predict_map[str(label_code)] = cat_mid
        if str(label_code) in labels_predict_unique.keys():
            labels_predict_unique[str(label_code)] += 1
        else:
            labels_predict_unique[str(label_code)] = 1

    #labels_predict = preprocessing.normalize([labels_predict], norm='l2')
    #labels_predict = labels_predict[0]
    labels_predict = labels_predict_codes.astype(str)
    lb = LabelBinarizer()
    labels_lb = lb.fit_transform(labels_predict)
    number_labels = len(lb.classes_)

    separator = ","
    lb_classes_string = separator.join([str(x) for x in lb.classes_])
    if log_file_path is not None:
        eprint("Classes " + str(len(lb.classes_)) + ": " + lb_classes_string)
    else:
        print("Classes " + str(len(lb.classes_)) + ": " + lb_classes_string)

    separator = ", "
    lines.append("Predicted Labels: " + separator.join(lb.classes_))

    print("[INFO] number of labels: %d" % (len(labels_lb)))
    print("[INFO] number of images: %d" % (len(data)))

    print("[INFO] splitting training set...")

    trainX = []
    testX = []
    trainY = []
    testY = []
    # For LSTM CNN model the images across time points for a single entity are held together, but that stack of images is trained against a single label
    if keras_model_type == 'densenet121_lstm':
        data = np.array(data)
        data = data.reshape(num_unique_stock_ids * num_unique_image_types, num_unique_time_days, image_size, image_size, 3)
        labels_lb = labels_lb.reshape(num_unique_stock_ids * num_unique_image_types, num_unique_time_days, number_labels)
        labels = []
        for l in labels_lb:
            labels.append(l[0])
        (trainX, testX, trainY, testY) = train_test_split(data, np.array(labels), test_size=0.2)
    else:
        (trainX, testX, trainY, testY) = train_test_split(np.array(data), np.array(labels_lb), test_size=0.2)

    model = None
    if keras_model_type == 'inceptionresnetv2':
        img_input = Input(shape=(image_size,image_size,3))

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

        #TOP
        x = GlobalAveragePooling2D(data_format='channels_last')(x)
        x = Dropout(0.6)(x)
        x = Dense(number_labels, activation='softmax')(x)

        model = Model(img_input, x, name='inception_resnet_v2')

    if keras_model_type == 'simple_1':

        init = "he_normal"
        reg = regularizers.l2(0.01)
        chanDim = -1

        model = Sequential()
        model.add(Conv2D(16, (7, 7), strides=(2, 2), padding="valid", kernel_initializer=init, kernel_regularizer=reg, input_shape=(image_size, image_size, 3)))
        model.add(Conv2D(32, (3, 3), padding="same", kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        # stack two more CONV layers, keeping the size of each filter
        # as 3x3 but increasing to 64 total learned filters
        model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        # increase the number of filters again, this time to 128
        model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        # fully-connected layer
        model.add(Flatten())
        model.add(Dense(512, kernel_initializer=init))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(number_labels))
        model.add(Activation("softmax"))

    if keras_model_type == 'densenet121application':
        input_tensor = Input(shape=(image_size,image_size,3))
        base_model = DenseNet121(
            include_top = False,
            weights = keras_model_weights,
            input_tensor = input_tensor,
            input_shape = (image_size,image_size,3)
        )

        new_model = Sequential()
        layer_num = 0
        for layer in base_model.layers:
            layer.trainable = True

            if keras_model_layers is not None and layer_num < keras_model_layers:
                new_model.add(layer)
            
            layer_num += 1

        if keras_model_layers is not None:
            new_model.add(Flatten())
            base_model = new_model

        op = Dense(256, activation='relu')(base_model.output)
        op = Dropout(0.25)(op)
        output_tensor = Dense(number_labels, activation='softmax')(op)

        model = Model(inputs=input_tensor, outputs=output_tensor)

    if keras_model_type == 'inceptionresnetv2application':
        input_tensor = Input(shape=(image_size,image_size,3))
        base_model = InceptionResNetV2(
            include_top = False,
            weights = keras_model_weights,
            input_tensor = input_tensor,
            input_shape = (image_size,image_size,3),
            pooling = 'avg'
        )

        new_model = Sequential()
        layer_num = 0
        for layer in base_model.layers:
            layer.trainable = True

            if keras_model_layers is not None and layer_num < keras_model_layers:
                new_model.add(layer)
            
            layer_num += 1

        if keras_model_layers is not None:
            new_model.add(Flatten())
            base_model = new_model

        op = Dense(256, activation='relu')(base_model.output)
        op = Dropout(0.25)(op)
        output_tensor = Dense(number_labels, activation='softmax')(op)

        model = Model(inputs=input_tensor, outputs=output_tensor)

    if keras_model_type == 'densenet121_lstm':
        n = DenseNet121(
            include_top=False,
            weights=keras_model_weights,
            input_shape=(image_size, image_size, 3),
            pooling = 'avg'
        )

        # do not train first layers, I want to only train
        # the 4 last layers (my own choice, up to you)
        for layer in n.layers[:-4]:
            layer.trainable = False

        model = Sequential()
        model.add(
            TimeDistributed(n, input_shape=(num_unique_time_days, image_size, image_size, 3))
        )
        model.add(
            TimeDistributed(
                Flatten()
            )
        )
        model.add(LSTM(256, activation='relu', return_sequences=False))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(number_labels, activation='softmax'))

    for layer in model.layers:
        print(layer.output_shape)

    print("[INFO] training network...")
    opt = Adam(lr=1e-3, decay=1e-3 / 50)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    checkpoint = ModelCheckpoint(filepath=output_model_file_path, monitor='accuracy', verbose=1, save_best_only=True, mode='max', save_frequency=1, save_weights_only=False)
    es = EarlyStopping(monitor='loss', mode='min', min_delta=0.01, patience=35, verbose=1)
    callbacks_list = [es, checkpoint]

    H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=15, batch_size=8, callbacks=callbacks_list)

    # print("[INFO] evaluating network...")
    # predictions = model.predict(testX, batch_size=32)
    # report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_)
    # print(report)
    # 
    # report_lines = report.split('\n')
    # separator = ""
    # for l in report_lines:
    #     lines.append(separator.join(l))

    iterator = 0
    for c in lb.classes_:
        class_map_lines.append([iterator, labels_predict_map[str(c)], labels_predict_unique[str(c)]])
        iterator += 1

#print(lines)
with open(outfile_path, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(lines)
writeFile.close()

#print(class_map_lines)
with open(output_class_map, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(class_map_lines)
writeFile.close()
