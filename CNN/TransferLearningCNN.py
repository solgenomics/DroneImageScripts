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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_image_label_file", required=True, help="file path for file holding image names and labels to be trained")
ap.add_argument("-m", "--output_model_file_path", required=True, help="file path for saving keras model, so that it can be loaded again in the future. it saves an hdf5 file as the model")
ap.add_argument("-o", "--outfile_path", required=True, help="file path where the output will be saved")
args = vars(ap.parse_args())

input_file = args["input_image_label_file"]
output_model_file_path = args["output_model_file_path"]
outfile_path = args["outfile_path"]

labels = [];
data = []
image_size = 75

def build_model(number_labels):
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
        image = Image.open(row[0])
        image = 2*(np.array(image.resize((image_size,image_size))) / 255.0) - 1.0

        if (len(image.shape) == 2):
            empty_mat = np.ones(image.shape, dtype=image.dtype) * 0
            image = cv2.merge((image, empty_mat, empty_mat))

        #print(image.shape)
        data.append(image)

        #value = str(int(float(row[1])))
        #value = str(math.ceil(float(row[1]) / 2.)*2)
        value = str(math.ceil(float(row[1]) / 3.)*3)
        labels.append(value)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
print(len(lb.classes_))

print("[INFO] number of labels: %d" % (len(labels)))
print("[INFO] number of images: %d" % (len(data)))

print("[INFO] splitting training set...")
(trainX, testX, trainY, testY) = train_test_split(np.array(data), np.array(labels), test_size=0.25)

init = "he_normal"
reg = regularizers.l2(0.01)
chanDim = -1

print("[INFO] building model...")
model = build_model(len(lb.classes_))

for layer in model.layers:
    print(layer.output_shape)

print("[INFO] training network...")
opt = Adam(lr=1e-3, decay=1e-3 / 50)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=50, batch_size=32)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_)
print(report)


lines = report.split('\n')
with open(outfile_path, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(lines)

writeFile.close()
