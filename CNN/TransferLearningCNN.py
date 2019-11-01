# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/CNN/TransferLearningCNN.py --input_image_label_file  /folder/myimagesandlabels.csv --output_model_file_path /folder/mymodel.h5 --outfile_path /export/myresults.csv

# import the necessary packages
import sys
import argparse
import csv
import imutils
import cv2
import numpy as np
import math
from PIL import Image
import pandas as pd
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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--log_file_path", required=False, help="file path to write log to. useful for using from the web interface")
ap.add_argument("-i", "--input_image_label_file", required=True, help="file path for file holding image names and labels to be trained")
ap.add_argument("-m", "--output_model_file_path", required=True, help="file path for saving keras model, so that it can be loaded again in the future. it saves an hdf5 file as the model")
ap.add_argument("-o", "--outfile_path", required=True, help="file path where the output will be saved")
ap.add_argument("-c", "--output_class_map", required=True, help="file path where the output for class map will be saved")
ap.add_argument("-a", "--keras_model_type_name", required=True, help="the name of the pre-trained Keras CNN model to use e.g. InceptionResNetV2")
ap.add_argument("-w", "--keras_model_weights", required=False, help="the name of the pre-trained Keras CNN model weights to use e.g. imagenet for the InceptionResNetV2 model. Leave empty to instantiate the model with random weights")
ap.add_argument("-k", "--keras_model_layers", required=False, help="the first X layers to use from a pre-trained Keras CNN model e.g. 10 for the first 10 layers from the InceptionResNetV2 model")
args = vars(ap.parse_args())

log_file_path = args["log_file_path"]
input_file = args["input_image_label_file"]
output_model_file_path = args["output_model_file_path"]
outfile_path = args["outfile_path"]
output_class_map = args["output_class_map"]
keras_model_name = args["keras_model_type_name"]
keras_model_weights = args["keras_model_weights"]
keras_model_layers = args["keras_model_layers"]
if keras_model_layers is not None:
    keras_model_layers = int(keras_model_layers)

if sys.version_info[0] < 3:
    raise Exception("Must use Python3. Use python3 in your command line.")

if log_file_path is not None:
    sys.stderr = open(log_file_path, 'a')

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

labels = [];
unique_labels = {}
unique_image_types = {}
unique_drone_run_band_names = {}
data = []
image_size = 75
#image_size = 299

def build_model(model_name, number_labels, weights, use_layers_num):
    model = Model()
    if model_name == 'InceptionResNetV2':
        input_tensor = Input(shape=(image_size,image_size,3))
        base_model = InceptionResNetV2(
            include_top = False,
            weights = weights,
            input_tensor = input_tensor,
            input_shape = (image_size,image_size,3),
            pooling = 'avg'
        )

        new_model = Sequential()
        layer_num = 0
        for layer in base_model.layers:
            layer.trainable = True

            if use_layers_num is not None and layer_num < use_layers_num:
                new_model.add(layer)
            
            layer_num += 1

        if use_layers_num is not None:
            new_model.add(Flatten())
            base_model = new_model
    
        op = Dense(256, activation='relu')(base_model.output)
        op = Dropout(0.25)(op)
        output_tensor = Dense(number_labels, activation='softmax')(op)
        
        model = Model(inputs=input_tensor, outputs=output_tensor)
    return model

print("[INFO] reading labels and image data...")
with open(input_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        stock_id = row[0]
        trait_name = row[3]
        image_type = row[4]
        drone_run_band_name = row[5]
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

    categorical_object = pd.cut(labels, 10)
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

    print("[INFO] building model...")
    model = build_model(keras_model_name, len(lb.classes_), keras_model_weights, keras_model_layers)

    for layer in model.layers:
        print(layer.output_shape)

    print("[INFO] training network...")
    opt = Adam(lr=1e-3, decay=1e-3 / 50)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    checkpoint = ModelCheckpoint(output_model_file_path, monitor='acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    history_callback = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=50, batch_size=8, callbacks=callbacks_list)
    # loss_history = history_callback.history["loss"]
    # numpy_loss_history = numpy.array(loss_history)
    # print(numpy_loss_history)

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
