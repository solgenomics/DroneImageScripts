# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/CNN/BasicCNN.py --input_image_path_string /folder/mypic1.png,/folder/mypic2.png --input_label_file /folder/mylabels.csv --outfile_path /export/myresults.csv

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
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_image_path_string", required=True, help="comma separated image paths string")
ap.add_argument("-l", "--input_label_file", required=True, help="file path for file holding labels to be trained")
ap.add_argument("-o", "--outfile_path", required=True, help="file path where the output will be saved")
args = vars(ap.parse_args())

input_image_string = args["input_image_path_string"]
input_label_file = args["input_label_file"]
outfile_path = args["outfile_path"]
images = input_image_string.split(",")

labels = [];
with open(input_label_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        labels.append(row[0])

data = []
for image_path in images:
    image = Image.open(image_path)
    image = np.array(image.resize((32,32))) / 255.0
    data.append(image)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

print("[INFO] splitting training set...")
(trainX, testX, trainY, testY) = train_test_split(np.array(data), np.array(labels), test_size=0.25)

model = Sequential()
model.add(Conv2D(8, (3,3), padding="same", input_shape=(32,32,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(16, (3,3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(32, (3,3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Flatten())
model.add(Dense(3))
model.add(Activation("softmax"))

print("[INFO] training network...")
opt = Adam(lr=1e-3, decay=1e-3 / 50)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=50, batch_size=32)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

#cv2.imwrite(outfile_path, vari)
