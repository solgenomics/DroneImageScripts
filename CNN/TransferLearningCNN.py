# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/CNN/TransferLearningCNN.py --input_image_label_file  /folder/myimagesandlabels.csv --output_model_file_path /folder/mymodel.h5 --outfile_path /export/myresults.csv

# import the necessary packages
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
from keras.layers import Input
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from PIL import Image
from keras.models import load_model
from keras.models import Model
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import ModelCheckpoint

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_image_label_file", required=True, help="file path for file holding image names and labels to be trained")
ap.add_argument("-m", "--output_model_file_path", required=True, help="file path for saving keras model, so that it can be loaded again in the future. it saves an hdf5 file as the model")
ap.add_argument("-o", "--outfile_path", required=True, help="file path where the output will be saved")
ap.add_argument("-c", "--output_class_map", required=True, help="file path where the output for class map will be saved")
ap.add_argument("-a", "--keras_model_type_name", required=True, help="the name of the per-trained Keras CNN model to use e.g. InceptionResNetV2")
args = vars(ap.parse_args())

input_file = args["input_image_label_file"]
output_model_file_path = args["output_model_file_path"]
outfile_path = args["outfile_path"]
output_class_map = args["output_class_map"]
keras_model_name = args["keras_model_type_name"]

labels = [];
unique_labels = {}
data = []
image_size = 75

def build_model(model_name, number_labels):
    model = Model()
    if model_name == 'InceptionResNetV2':
        input_tensor = Input(shape=(image_size,image_size,3))
        base_model = InceptionResNetV2(
            include_top = False,
            weights = 'imagenet',
            #input_tensor = None,
            input_tensor = input_tensor,
            #input_shape = None,
            input_shape = (image_size,image_size,3),
            pooling = 'avg'
            #classes = 1000
        )
        for layer in base_model.layers:
            layer.trainable = True
        
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
        image = Image.open(row[1])
        image = np.array(image.resize((image_size,image_size))) / 255.0

        if (len(image.shape) == 2):
            empty_mat = np.ones(image.shape, dtype=image.dtype) * 0
            image = cv2.merge((image, empty_mat, empty_mat))

        #print(image.shape)
        data.append(image)

        value = float(row[2])
        #value = str(int(float(row[1])*100))
        #value = str(math.ceil(float(row[1]) / 2.)*2)
        #value = str(math.ceil(float(row[1]) / 3.)*3)
        labels.append(value)
        if value in unique_labels.keys():
            unique_labels[value] += 1
        else:
            unique_labels[value] = 1

lines = []
class_map_lines = []
if len(unique_labels.keys()) < 2:
    lines = ["Number of labels is less than 2, so nothing to predict!"]
else:
    print(labels)

    labels_predict = []
    unique_labels_predict = {}
    if len(unique_labels.keys()) == len(data):
        print("Number of unique labels is equal to number of data points, so dividing number of labels by roughly 3")
        all_labels_decimal = 1
        for l in labels:
            if l > 1 or l < 0:
                all_labels_decimal = 0
        if all_labels_decimal == 1:
            for l in labels:
                labels_predict.append(str(math.ceil(float(l*100) / 3.)*3/100))
        else:
            for l in labels:
                labels_predict.append(str(math.ceil(float(l) / 3.)*3))
    elif len(unique_labels.keys())/len(data) > 0.6:
        print("Number of unique labels is greater than 60% the number of data points, so dividing number of labels by roughly 2")
        all_labels_decimal = 1
        for l in labels:
            if l > 1 or l < 0:
                all_labels_decimal = 0
        if all_labels_decimal == 1:
            for l in labels:
                labels_predict.append(str(math.ceil(float(l*100) / 2.)*2/100))
        else:
            for l in labels:
                labels_predict.append(str(math.ceil(float(l) / 2.)*2))
    else:
        for l in labels:
            labels_predict.append(str(l))


    for value in labels_predict:
        if value in unique_labels_predict.keys():
            unique_labels_predict[value] += 1
        else:
            unique_labels_predict[value] = 1

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels_predict)
    print(len(lb.classes_))
    print(lb.classes_)
    #print(labels)

    separator = ", "
    lines.append("Predicted Labels: " + separator.join(unique_labels_predict.keys()))

    print("[INFO] number of labels: %d" % (len(labels)))
    print("[INFO] number of images: %d" % (len(data)))

    print("[INFO] splitting training set...")
    (trainX, testX, trainY, testY) = train_test_split(np.array(data), np.array(labels), test_size=0.25)

    init = "he_normal"
    reg = regularizers.l2(0.01)
    chanDim = -1

    print("[INFO] building model...")
    model = build_model(keras_model_name, len(lb.classes_))

    for layer in model.layers:
        print(layer.output_shape)

    print("[INFO] training network...")
    opt = Adam(lr=1e-3, decay=1e-3 / 50)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    checkpoint = ModelCheckpoint(output_model_file_path, monitor='acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=50, batch_size=32, callbacks=callbacks_list)

    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=32)
    report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_)
    print(report)

    report_lines = report.split('\n')
    separator = ""
    for l in report_lines:
        lines.append(separator.join(l))

    iterator = 0
    for c in lb.classes_:
        class_map_lines.append([iterator, c])
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
