# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/CNN/BasicCNNText.py --input_label_file  /folder/filenamesandlabels.csv --input_file_dir  /folder/ --output_model_file_path /folder/mymodel.h5 --outfile_path /export/myresults.csv

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
from numpy import genfromtxt
from sklearn.preprocessing import StandardScaler

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_label_file", required=True, help="file path for file holding image names and labels to be trained")
ap.add_argument("-f", "--input_file_dir", required=True, help="file dir for input files")
ap.add_argument("-m", "--output_model_file_path", required=True, help="file path for saving keras model, so that it can be loaded again in the future. it saves an hdf5 file as the model")
ap.add_argument("-o", "--outfile_path", required=True, help="file path where the output will be saved")
args = vars(ap.parse_args())

input_file = args["input_label_file"]
input_file_dir = args["input_file_dir"]
output_model_file_path = args["output_model_file_path"]
outfile_path = args["outfile_path"]

print("[INFO] reading labels and image data...")

matrices = []
labels = []
with open(input_file) as input_file:
    csv_reader = csv.reader(input_file, delimiter=',')
    for row in csv_reader:
        file_name = row[1]
        try:
            input_matrix = genfromtxt(input_file_dir+file_name+'.dat', delimiter='\t')
            input_matrix_clean = []
            for line in input_matrix:
                new_line = []
                nans = 0
                for i in line:
                    if math.isnan(i):
                        nans += 1
                    else:
                        input_matrix_clean.append(i)
            input_matrix_clean = np.array(input_matrix_clean)
            
            matrices.append(input_matrix_clean)
            value = str(math.ceil(float(row[2]) / 400.)*400)
            labels.append(value)
        except IOError:
            print('Not found:'+file_name)

sc = StandardScaler()
input_matrix_clean = sc.fit_transform(matrices)
print(matrices)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)



print("[INFO] number of labels: %d" % (len(labels)))
print("[INFO] number of matrices: %d" % (len(input_matrix_clean)))

print("[INFO] splitting training set...")
(trainX, testX, trainY, testY) = train_test_split(np.array(input_matrix_clean), np.array(labels), test_size=0.25)

model = Sequential()
model.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=3126))
model.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
model.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
model.add(Dense(len(lb.classes_)))
model.add(Activation("softmax"))

print("[INFO] training network...")
opt = Adam(lr=1e-3, decay=1e-3 / 50)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=50, batch_size=12)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_)
print(report)


lines = report.split('\n')
with open(outfile_path, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(lines)

writeFile.close()
