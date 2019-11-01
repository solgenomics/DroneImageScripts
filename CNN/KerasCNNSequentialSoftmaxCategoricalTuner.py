# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/CNN/KerasCNNSequentialSoftmaxCategoricalTuner.py --input_image_label_file  /folder/myimagesandlabels.csv --output_model_file_path /folder/mymodel.h5 --outfile_path /export/myresults.csv

# import the necessary packages
import sys
import argparse
import csv
import imutils
import cv2
import numpy as np
import math
from keras.models import Sequential
from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers import Concatenate
from keras.layers.core import Lambda
from keras.layers import GlobalAveragePooling2D
from keras.engine.input_layer import Input
from keras.optimizers import Adam
from keras import backend
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from PIL import Image
from keras.models import load_model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import pandas as pd
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from kerastuner.tuners import RandomSearch

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

def build_model(hp):
    model = Sequential()
    model.add(Conv2D(
        filters=hp.Range('conv_1_filter', min_value=64, max_value=128, step=16),
        kernel_size=hp.Choice('conv_1_kernel', values = [3,5]),
        activation='relu',
        input_shape=(28,28,1)
    ))
    model.add(Conv2D(
        filters=hp.Range('conv_2_filter', min_value=32, max_value=64, step=16),
        kernel_size=hp.Choice('conv_2_kernel', values = [3,5]),
        activation='relu'
    ))
    model.add(Flatten())
    model.add(Dense(
        units=hp.Range('dense_1_units', min_value=32, max_value=128, step=16),
        activation='relu'
    ))
    model.add(Dense(len(lb.classes_), activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

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
        image = np.array(image.resize((75,75))) / 255.0

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


    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=5,
        directory='output',
        project_name='FashionMNIST'
    )
    tuner.search(trainX, trainY, epochs=2, validation_split=0.1)

    H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=50, batch_size=32, callbacks=callbacks_list)

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
