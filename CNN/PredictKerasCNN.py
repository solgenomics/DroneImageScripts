# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/CNN/PredictKerasCNN.py --input_image_label_file  /folder/myimagesandlabels.csv --output_model_file_path /folder/mymodel.h5 --outfile_path /export/myresults.csv

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
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from PIL import Image
from keras.models import load_model
from keras.utils import to_categorical
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_image_label_file", required=True, help="file path for file holding image names to predict phenotypes from model")
ap.add_argument("-m", "--input_model_file_path", required=True, help="file path for saved keras model to use in prediction")
ap.add_argument("-o", "--outfile_path", required=True, help="file path where the output will be saved")
ap.add_argument("-a", "--keras_model_type_name", required=True, help="the name of the per-trained Keras CNN model to use e.g. InceptionResNetV2")

args = vars(ap.parse_args())

input_file = args["input_image_label_file"]
input_model_file_path = args["input_model_file_path"]
outfile_path = args["outfile_path"]
keras_model_name = args["keras_model_type_name"]

data = []

image_size = 32
if keras_model_name == 'KerasCNNSequentialSoftmaxCategorical':
    image_size = 32
elif keras_model_name == 'KerasCNNInceptionResNetV2':
    image_size = 75

print("[INFO] reading labels and image data...")
with open(input_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        stock_id = row[0]
        image = Image.open(row[1])
        image = np.array(image.resize((image_size,image_size))) / 255.0

        if (len(image.shape) == 2):
            empty_mat = np.ones(image.shape, dtype=image.dtype) * 0
            image = cv2.merge((image, empty_mat, empty_mat))

        #print(image.shape)
        data.append(image)

#print(unique_labels)
lines = []
if len(data) < 1:
    lines = ["No images, so nothing to predict!"]
else:
    print("[INFO] number of images: %d" % (len(data)))
    model = load_model(input_model_file_path)
    opt = Adam(lr=1e-3, decay=1e-3 / 50)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    vstack = []
    for img in data:
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        vstack.append(x)

    images = np.vstack(vstack)
    prob_predictions = model.predict(images, batch_size=10)
    predictions = np.argmax(prob_predictions, axis=1)
    iterator = 0
    for p in predictions:
        line = [p]
        for i in prob_predictions[iterator]:
            line.append(i)
        lines.append(line)
        iterator += 1

#print(lines)
with open(outfile_path, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(lines)

writeFile.close()
