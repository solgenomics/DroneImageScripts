# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/CNN/PredictKerasCNN.py --input_image_label_file  /folder/myimagesandlabels.csv --output_model_file_path /folder/mymodel.h5 --outfile_path /export/myresults.csv

# import the necessary packages
import sys
import argparse
import csv
import imutils
import cv2
import numpy as np
import math
import json
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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import inception_resnet_v2
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import densenet
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from kerastuner.tuners import RandomSearch
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf
from numpy.polynomial.polynomial import polyfit

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--log_file_path", required=False, help="file path to write log to. useful for using from the web interface")
ap.add_argument("-i", "--input_image_label_file", required=True, help="file path for file holding image names to predict phenotypes from model")
ap.add_argument("-m", "--input_model_file_path", required=True, help="file path for saved keras model to use in prediction")
ap.add_argument("-o", "--outfile_path", required=True, help="file path where the output will be saved")
ap.add_argument("-e", "--outfile_activation_path", required=True, help="file path where the activation graph output will be saved")
ap.add_argument("-u", "--outfile_evaluation_path", required=True, help="file path where the model evaluation output will be saved (in the case there were previous phenotypes for the images)")
ap.add_argument("-a", "--keras_model_type_name", required=True, help="the name of the per-trained Keras CNN model to use e.g. InceptionResNetV2")
ap.add_argument("-t", "--training_data_input_file", required=True, help="The input data file used to train the model previously. this file should have the image file paths and labels used during training")
ap.add_argument("-r", "--retrain_model", help="whether to retrain the model on the images. this will only work if the plots already have true phenotypes saved in the database.")
ap.add_argument("-c", "--class_map", help="whether to plot true vs prediction, provide a json encoded class map. this will only work if the plots already have true phenotypes saved in the database.")
ap.add_argument("-p", "--plot_prediction_comparison", help="whether to plot true vs prediction. this will only work if the plots already have true phenotypes saved in the database.")

args = vars(ap.parse_args())

log_file_path = args["log_file_path"]
input_file = args["input_image_label_file"]
input_model_file_path = args["input_model_file_path"]
outfile_path = args["outfile_path"]
outfile_activation_path = args["outfile_activation_path"]
outfile_evaluation_path = args["outfile_evaluation_path"]
keras_model_name = args["keras_model_type_name"]
training_data_input_file = args["training_data_input_file"]
retrain_model = args["retrain_model"]
plot_prediction_comparison = args["plot_prediction_comparison"]
class_map = args["class_map"]
if class_map is not None:
    class_map = json.loads(class_map)

if sys.version_info[0] < 3:
    raise Exception("Must use Python3. Use python3 in your command line.")

if log_file_path is not None:
    sys.stderr = open(log_file_path, 'a')

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

unique_stock_ids = {}
unique_time_days = {}
unique_image_types = {}
data = []
previous_labeled_data = []
previous_labels = []
unique_labels = {}

image_size = 75
if keras_model_name == 'KerasCNNSequentialSoftmaxCategorical':
    image_size = 32
if keras_model_name == 'SimpleKerasTunerCNNSequentialSoftmaxCategorical':
    image_size = 32
elif keras_model_name == 'KerasTunerCNNInceptionResNetV2':
    image_size = 75
elif keras_model_name == 'KerasTunerCNNSequentialSoftmaxCategorical':
    image_size = 32
elif keras_model_name == 'KerasCNNInceptionResNetV2':
    image_size = 75
elif keras_model_name == 'KerasCNNLSTMDenseNet121ImageNetWeights':
    image_size = 75
elif keras_model_name == 'KerasCNNInceptionResNetV2ImageNetWeights':
    image_size = 75

print("[INFO] reading labels and image data...")
with open(input_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        stock_id = row[0]
        image_type = row[3]
        time_days = row[4]
        image = Image.open(row[1])
        image = np.array(image.resize((image_size,image_size))) / 255.0

        if (len(image.shape) == 2):
            empty_mat = np.ones(image.shape, dtype=image.dtype) * 0
            image = cv2.merge((image, empty_mat, empty_mat))

        #print(image.shape)
        data.append(image)

        previous_value = row[2]
        previous_labels.append(previous_value)

        if image_type in unique_image_types.keys():
            unique_image_types[image_type] += 1
        else:
            unique_image_types[image_type] = 1

        if stock_id in unique_stock_ids.keys():
            unique_stock_ids[stock_id] += 1
        else:
            unique_stock_ids[stock_id] = 1

        if time_days in unique_time_days.keys():
            unique_time_days[time_days] += 1
        else:
            unique_time_days[time_days] = 1

trained_image_data = []
trained_labels = []

print("[INFO] reading labels and image data used to train model previously...")
with open(training_data_input_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        stock_id = row[0]
        trait_name = row[3]
        image_type = row[4]
        time_days = row[5]

        image = Image.open(row[1])
        image = np.array(image.resize((image_size,image_size))) / 255.0

        if (len(image.shape) == 2):
            empty_mat = np.ones(image.shape, dtype=image.dtype) * 0
            image = cv2.merge((image, empty_mat, empty_mat))

        #print(image.shape)
        trained_image_data.append(image)

        value = float(row[2])
        trained_labels.append(value)

num_unique_stock_ids = len(unique_stock_ids.keys())
num_unique_image_types = len(unique_image_types.keys())
num_unique_time_days = len(unique_time_days.keys())
num_unique_stock_ids = len(unique_stock_ids.keys())
num_unique_image_types = len(unique_image_types.keys())
num_previous_unique_time_days = len(previous_unique_time_days.keys())
num_previous_unique_stock_ids = len(previous_unique_stock_ids.keys())
num_previous_unique_image_types = len(previous_unique_image_types.keys())
if num_unique_stock_ids * num_unique_time_days * num_unique_image_types != len(data):
    print(num_unique_stock_ids)
    print(num_unique_time_days)
    print(num_unique_image_types)
    print(len(data))
    print(len(labels))
    raise Exception('Number of rows in input file (images) is not equal to the number of unique stocks times the number of unique time points times the number of unique image types. This means the input data in uneven')

print("[INFO] augmenting test images...")

#Data Generation uses the same settings during training!
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    #rotation_range=20,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # horizontal_flip=True,
    #vertical_flip=True,
    # brightness_range=[0.5,1.5],
    # zoom_range=[0.8,1.2],
    # shear_range=10
)

data = np.array(data)
trained_image_data = np.array(trained_image_data)
trained_labels = np.array(trained_labels)

datagen.fit(trained_image_data)

data_augmentation = 4
augmented_data = []
augmented = datagen.flow(data, None, batch_size=len(data))
for i in range(0, data_augmentation):
    for x_aug in augmented.next():
        augmented_data.append(x_aug)
augmented_data = np.array(augmented_data)

if data_augmentation * num_unique_stock_ids * num_unique_time_days * num_unique_image_types != len(augmented_data):
    print(num_unique_stock_ids)
    print(num_unique_time_days)
    print(num_unique_image_types)
    print(len(augmented_data))
    raise Exception('Number of augmented images is not equal to the number of unique stocks times the number of unique time points times the number of unique image types time the augmentation. This means the input data in uneven')

#print(unique_labels)
lines = []
evaluation_lines = []
if len(augmented_data) < 1:
    lines = ["No images, so nothing to predict!"]
else:
    print("[INFO] number of images: %d" % (len(data)))
    print("[INFO] number of augmented images: %d" % (len(augmented_data)))
    model = load_model(input_model_file_path)

    for layer in model.layers:
        print(layer.output_shape)

    if keras_model_name == 'KerasCNNLSTMDenseNet121ImageNetWeights':
        data = data.reshape(num_unique_stock_ids * num_unique_image_types, num_unique_time_days, image_size, image_size, 3)
        augmented_data = augmented_data.reshape(data_augmentation * num_unique_stock_ids * num_unique_image_types, num_unique_time_days, image_size, image_size, 3)
    else:
        data = data.reshape(len(data), image_size, image_size, 3)
        augmented_data = augmented_data.reshape(len(augmented_data), image_size, image_size, 3)

    prob_predictions = model.predict(augmented_data, batch_size=8)
    predictions = np.argmax(prob_predictions, axis=1)
    print(predictions)

    predictions = predictions.reshape(data_augmentation, int(len(augmented_data)/data_augmentation))
    print(predictions)
    averaged_predictions = np.mean(predictions, axis=0)
    print(averaged_predictions)
    averaged_predictions = [int(round(elem)) for elem in averaged_predictions]
    print(averaged_predictions)

    separator = ","
    prediction_string = separator.join([str(x) for x in averaged_predictions])
    if log_file_path is not None:
        eprint("Predictions: " + prediction_string)
    else:
        print("Predictions: " + prediction_string)

    mean_prediction = int(round(sum(averaged_predictions)/len(averaged_predictions)))
    mean_prediction_label = float(class_map[str(mean_prediction)]['label'])
    for p in averaged_predictions:
        line = [p]
        lines.append(line)

    print("[INFO] Getting model activations for each images and each layer")
    layer_names = []
    # layer_outputs = [layer.output for layer in model.layers[:20]][1:] # Extracts the outputs of the top 20 layers
    layer_outputs = [layer.output for layer in model.layers[:30]][1:] # Extracts the outputs of the top 20 layers
    for layer in model.layers[:30]:
        layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot

    activation_model = Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
    activation_figures = []

    if len(augmented_data[0].shape) == 3:
        layer_displays_above_median = {}
        layer_displays_below_median = {}
        average_img_above_median = np.zeros_like(augmented_data[0])
        average_img_below_median = np.zeros_like(augmented_data[0])
        num_img_above_median = 0
        num_img_below_median = 0
        itera = 0
        for image in data:
            activations = activation_model.predict(np.array([image]))
            pred = averaged_predictions[itera]
            pred_label = float(class_map[str(pred)]['label'])

            if pred_label > mean_prediction_label:
                average_img_above_median += image
                num_img_above_median += 1
            else:
                average_img_below_median += image
                num_img_below_median += 1

            for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
                if (len(layer_activation.shape) == 4):
                    n_features = layer_activation.shape[-1] # Number of features in the feature map
                    for n in range(n_features):
                        channel_image = layer_activation[0, :, :, n]
                        channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
                        channel_image /= channel_image.std()
                        channel_image *= 64
                        channel_image += 128
                        channel_image = np.clip(channel_image, 0, 255).astype('uint8')

                        if pred_label > mean_prediction_label:
                            if layer_name in layer_displays_above_median.keys():
                                layer_displays_above_median[layer_name].append(channel_image)
                            else:
                                layer_displays_above_median[layer_name] = [channel_image]
                        else:
                            if layer_name in layer_displays_below_median.keys():
                                layer_displays_below_median[layer_name].append(channel_image)
                            else:
                                layer_displays_below_median[layer_name] = [channel_image]
            itera += 1

        average_img_above_median = average_img_above_median/num_img_above_median
        average_img_below_median = average_img_below_median/num_img_below_median

        plt.figure()
        plt.title("Average Image Above Median")
        plt.grid(False)
        plt.imshow(average_img_above_median, aspect='auto', cmap='viridis')
        fig = plt.gcf()
        activation_figures.append(fig)

        plt.figure()
        plt.title("Average Image Below Median")
        plt.grid(False)
        plt.imshow(average_img_below_median, aspect='auto', cmap='viridis')
        fig = plt.gcf()
        activation_figures.append(fig)

        for layer_name in layer_displays_above_median.keys():
            activation_images = layer_displays_above_median[layer_name]
            avg_img = np.zeros_like(activation_images[0])
            for a in activation_images:
                avg_img += a
            avg_activation = avg_img/len(activation_images)

            plt.figure()
            plt.title("Above Median: "+layer_name)
            plt.grid(False)
            plt.imshow(avg_activation, aspect='auto', cmap='viridis')
            fig = plt.gcf()
            activation_figures.append(fig)

        for layer_name in layer_displays_below_median.keys():
            activation_images = layer_displays_below_median[layer_name]
            avg_img = np.zeros_like(activation_images[0])
            for a in activation_images:
                avg_img += a
            avg_activation = avg_img/len(activation_images)

            plt.figure()
            plt.title("Below Median: "+layer_name)
            plt.grid(False)
            plt.imshow(avg_activation, aspect='auto', cmap='viridis')
            fig = plt.gcf()
            activation_figures.append(fig)

    if retrain_model == True:
        if len(unique_labels.keys()) < 2:
            lines = ["Number of previous labels is less than 2, so will not evaluate model performance!"]
        else:
            categorical_object = pd.cut(previous_labels, 25)
            labels_predict_codes = categorical_object.codes
            categories = categorical_object.categories

            labels_predict_map = {}
            labels_predict_unique = {}
            for index in range(len(previous_labels)):
                label = previous_labels[index]
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

            (trainX, testX, trainY, testY) = train_test_split(np.array(previous_labeled_data), np.array(labels_lb), test_size=0.25)

            checkpoint = ModelCheckpoint(input_model_file_path, monitor='acc', verbose=1, save_best_only=True, mode='max')
            callbacks_list = [checkpoint]

            H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=50, batch_size=8, callbacks=callbacks_list)

            print("[INFO] evaluating network...")
            predictions = model.predict(testX, batch_size=32)
            report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_)
            print(report)

            report_lines = report.split('\n')
            separator = ""
            for l in report_lines:
                evaluation_lines.append(separator.join(l))

    if plot_prediction_comparison == "True":
        previous_labeled_data_no_missing = []
        predictions_no_missing = []
        for iterator in range(0,len(previous_labels)):
            previous_value = previous_labels[iterator]
            prediction = averaged_predictions[iterator]
            if previous_value is not None and previous_value != '' and previous_value != ' ':
                if isinstance(previous_value, int):
                    previous_value = int(previous_value)
                if isinstance(previous_value, float) or isinstance(previous_value, str):
                    previous_value = float(previous_value)
                previous_labeled_data_no_missing.append(previous_value)
                predictions_no_missing.append(prediction)
        
        prediction_converted = []
        for p in predictions_no_missing:
            prediction_converted.append(class_map[str(p)]['label'])

        prediction_converted = np.array(prediction_converted, dtype=np.float32)
        b, m = polyfit(previous_labeled_data_no_missing, prediction_converted, 1)

        regressed_predictions = []
        for p in previous_labeled_data_no_missing:
            r = b + (m * p)
            regressed_predictions.append(r)

        plt.figure()
        plt.title("Prediction vs True Values")
        plt.grid(True)
        plt.plot(previous_labeled_data_no_missing, prediction_converted, 'bo')
        plt.plot(previous_labeled_data_no_missing, regressed_predictions, '-')
        plt.xlabel("True")
        plt.ylabel("Predicted")
        fig = plt.gcf()
        activation_figures.append(fig)

    pdf = matplotlib.backends.backend_pdf.PdfPages(outfile_activation_path)
    for fig in activation_figures:
        pdf.savefig(fig)
    pdf.close()

#print(lines)
with open(outfile_path, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(lines)
writeFile.close()

with open(outfile_evaluation_path, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(evaluation_lines)
writeFile.close()
