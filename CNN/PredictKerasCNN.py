# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/CNN/PredictKerasCNN.py --input_image_label_file  /folder/myimagesandlabels.csv --output_model_file_path /folder/mymodel.h5 --outfile_path /export/myresults.csv

# import the necessary packages
import argparse
import csv
import imutils
import cv2
import numpy as np
import math
from keras import models
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
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_image_label_file", required=True, help="file path for file holding image names to predict phenotypes from model")
ap.add_argument("-m", "--input_model_file_path", required=True, help="file path for saved keras model to use in prediction")
ap.add_argument("-o", "--outfile_path", required=True, help="file path where the output will be saved")
ap.add_argument("-e", "--outfile_activation_path", required=True, help="file path where the activation graph output will be saved")
ap.add_argument("-u", "--outfile_evaluation_path", required=True, help="file path where the model evaluation output will be saved (in the case there were previous phenotypes for the images)")
ap.add_argument("-a", "--keras_model_type_name", required=True, help="the name of the per-trained Keras CNN model to use e.g. InceptionResNetV2")
ap.add_argument("-r", "--retrain_model", help="whether to retrain the model on the images. this will only work if the images already have phenotypes saved in the database.")

args = vars(ap.parse_args())

input_file = args["input_image_label_file"]
input_model_file_path = args["input_model_file_path"]
outfile_path = args["outfile_path"]
outfile_activation_path = args["outfile_activation_path"]
outfile_evaluation_path = args["outfile_evaluation_path"]
keras_model_name = args["keras_model_type_name"]
retrain_model = args["retrain_model"]

data = []
previous_labeled_data = []
previous_labels = []
unique_labels = {}

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

        previous_value = row[2]
        if previous_value is not None:
            previous_value = float(previous_value)
            previous_labels.append(previous_value)
            previous_labeled_data.append(image)

            if previous_value in unique_labels.keys():
                unique_labels[previous_value] += 1
            else:
                unique_labels[previous_value] = 1

#print(unique_labels)
lines = []
evaluation_lines = []
if len(data) < 1:
    lines = ["No images, so nothing to predict!"]
else:
    print("[INFO] number of images: %d" % (len(data)))
    model = load_model(input_model_file_path)

    vstack = []
    for img in data:
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        vstack.append(x)

    images = np.vstack(vstack)
    prob_predictions = model.predict(images, batch_size=10)
    predictions = np.argmax(prob_predictions, axis=1)
    print(predictions)
    iterator = 0
    for p in predictions:
        line = [p]
        for i in prob_predictions[iterator]:
            line.append(i)
        lines.append(line)
        iterator += 1

        
    layer_outputs = [layer.output for layer in model.layers[:12]] # Extracts the outputs of the top 12 layers
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
    images_per_row = 16

    layer_names = []
    for layer in model.layers[:12]:
        layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot

    for img in vstack:
        activations = activation_model.predict(img/255) # Returns a list of five Numpy arrays: one array per layer activation

        layer_displays = {}
        for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
            n_features = layer_activation.shape[-1] # Number of features in the feature map
            size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
            n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
            for col in range(n_cols): # Tiles each filter into a big horizontal grid
                for row in range(images_per_row):
                    channel_image = layer_activation[0,
                                                     :, :,
                                                     col * images_per_row + row]
                    channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')

                    if layer_name in layer_displays.keys():
                        layer_displays[layer_name].append(channel_image)
                    else:
                        layer_displays[layer_name] = [channel_image]

    average_layer_display = {}
    activation_figures = []
    for layer_name in layer_displays.keys():
        activation_images = layer_displays[layer_name]
        avg_img = np.zeros_like(activation_images[0])
        for a in activation_images:
            avg_img += a
        avg_activation = avg_img/len(activation_images)
        average_layer_display[layer_name] = avg_activation

        plt.figure()
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(avg_activation, aspect='auto', cmap='viridis')
        fig = plt.gcf()
        activation_figures.append(fig)

    pdf = matplotlib.backends.backend_pdf.PdfPages(outfile_activation_path)
    for fig in activation_figures:
        pdf.savefig(fig)
    pdf.close()

    if retrain_model == True:
        if len(unique_labels.keys()) < 2:
            lines = ["Number of previous labels is less than 2, so will not evaluate model performance!"]
        else:
            labels_predict = []
            unique_labels_predict = {}
            if len(unique_labels.keys()) == len(previous_labeled_data):
                print("Number of unique labels is equal to number of data points, so dividing number of labels by roughly 3")
                all_labels_decimal = 1
                for l in previous_labels:
                    if l > 1 or l < 0:
                        all_labels_decimal = 0
                if all_labels_decimal == 1:
                    for l in previous_labels:
                        labels_predict.append(str(math.ceil(float(l*100) / 3.)*3/100))
                else:
                    for l in previous_labels:
                        labels_predict.append(str(math.ceil(float(l) / 3.)*3))
            elif len(unique_labels.keys())/len(previous_labeled_data) > 0.6:
                print("Number of unique labels is greater than 60% the number of data points, so dividing number of labels by roughly 2")
                all_labels_decimal = 1
                for l in previous_labels:
                    if l > 1 or l < 0:
                        all_labels_decimal = 0
                if all_labels_decimal == 1:
                    for l in previous_labels:
                        labels_predict.append(str(math.ceil(float(l*100) / 2.)*2/100))
                else:
                    for l in previous_labels:
                        labels_predict.append(str(math.ceil(float(l) / 2.)*2))
            else:
                for l in previous_labels:
                    labels_predict.append(str(l))

            lb = LabelBinarizer()
            labels = lb.fit_transform(labels_predict)
            print(len(lb.classes_))
            print(lb.classes_)

            (trainX, testX, trainY, testY) = train_test_split(np.array(previous_labeled_data), np.array(labels), test_size=0.25)

            checkpoint = ModelCheckpoint(input_model_file_path, monitor='acc', verbose=1, save_best_only=True, mode='max')
            callbacks_list = [checkpoint]

            H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=50, batch_size=32, callbacks=callbacks_list)

            print("[INFO] evaluating network...")
            predictions = model.predict(testX, batch_size=32)
            report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_)
            print(report)

            report_lines = report.split('\n')
            separator = ""
            for l in report_lines:
                evaluation_lines.append(separator.join(l))

#print(lines)
with open(outfile_path, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(lines)
writeFile.close()

with open(outfile_evaluation_path, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(evaluation_lines)
writeFile.close()
