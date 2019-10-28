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
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.optimizers import Adam
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from PIL import Image
from keras.models import load_model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import pandas as pd

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--log_file_path", required=False, help="file path to write log to. useful for using from the web interface")
ap.add_argument("-i", "--input_image_label_file", required=True, help="file path for file holding image names and labels to be trained")
ap.add_argument("-m", "--output_model_file_path", required=True, help="file path for saving keras model, so that it can be loaded again in the future. it saves an hdf5 file as the model")
ap.add_argument("-o", "--outfile_path", required=True, help="file path where the output will be saved")
ap.add_argument("-c", "--output_class_map", required=True, help="file path where the output for class map will be saved")
args = vars(ap.parse_args())

log_file_path = args["log_file_path"]
input_file = args["input_image_label_file"]
output_model_file_path = args["output_model_file_path"]
outfile_path = args["outfile_path"]
output_class_map = args["output_class_map"]

if sys.version_info[0] < 3:
    raise Exception("Must use Python3. Use python3 in your command line.")

if log_file_path is not None:
    sys.stderr = open(log_file_path, 'a')

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

unique_labels = {}
unique_image_types = {}
unique_drone_run_band_names = {}
labels = []
data = []

print("[INFO] reading labels and image data...")
with open(input_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        stock_id = row[0]
        trait_name = row[3]
        image_type = row[4]
        drone_run_band_name = row[5]
        image = Image.open(row[1])
        image = np.array(image.resize((32,32))) / 255.0

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

    categorical_object = pd.cut(labels, 15)
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
    (trainX, testX, trainY, testY) = train_test_split(np.array(data), np.array(labels_lb), test_size=0.2)

    init = "he_normal"
    reg = regularizers.l2(0.01)
    chanDim = -1

    model = Sequential()
    model.add(Conv2D(16, (7, 7), strides=(2, 2), padding="valid", kernel_initializer=init, kernel_regularizer=reg, input_shape=(32, 32, 3)))
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
    model.add(Dense(len(lb.classes_)))
    model.add(Activation("softmax"))

    # model.add(Conv2D(8, (3, 3), padding="same", input_shape=(32, 32, 3)))
    # model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Conv2D(16, (3, 3), padding="same"))
    # model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Conv2D(32, (3, 3), padding="same"))
    # model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Flatten())
    # model.add(Dense(3))
    # model.add(Activation("softmax"))

    for layer in model.layers:
        print(layer.output_shape)

    print("[INFO] training network...")
    opt = Adam(lr=1e-3, decay=1e-3 / 50)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    checkpoint = ModelCheckpoint(output_model_file_path, monitor='acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=50, batch_size=8, callbacks=callbacks_list)

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
