# import the necessary packages
import sys
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class CNNProcessData:
    def __init__(self):
        pass

    def get_imagedatagenerator(self):
        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            #rotation_range=20,
            #width_shift_range=0.05,
            #height_shift_range=0.05,
            #horizontal_flip=True,
            # vertical_flip=True,
            #brightness_range=[0.8,1.2]
        )
        return datagen

    def generate_croppings(self, testX, testY, image_size, number):
        if number != 11:
            raise Exception("Only implemented for number = 11 right now")

        augmented_testX_1 = []
        augmented_testX_2 = []
        augmented_testX_3 = []
        augmented_testX_4 = []
        augmented_testX_5 = []
        augmented_testX_6 = []
        augmented_testX_7 = []
        augmented_testX_8 = []
        augmented_testX_9 = []
        augmented_testX_10 = []
        augmented_testX_11 = []
        mid_image_size = int(round(image_size/2))
        for img in testX:
            height = img.shape[0]
            small_height = int(round(height*0.1))
            mid_height = int(round(height/2))
            width = img.shape[1]
            mid_width = int(round(width/2))
            crop_img1 = img[height-image_size:height, 0:image_size]
            crop_img2 = img[height-image_size:height, width-image_size:width]
            crop_img3 = img[0:image_size, width-image_size:width]
            crop_img4 = img[0:image_size, 0:image_size]
            crop_img5 = img[mid_height-mid_image_size:mid_height+mid_image_size, mid_width-mid_image_size:mid_width+mid_image_size]
            crop_img6 = img[mid_height-mid_image_size:mid_height+mid_image_size, 0:image_size]
            crop_img7 = img[mid_height-mid_image_size:mid_height+mid_image_size, width-image_size:width]
            crop_img8 = img[mid_height+small_height-mid_image_size:mid_height+small_height+mid_image_size, 0:image_size]
            crop_img9 = img[mid_height+small_height-mid_image_size:mid_height+small_height+mid_image_size, width-image_size:width]
            crop_img10 = img[mid_height-small_height-mid_image_size:mid_height-small_height+mid_image_size, 0:image_size]
            crop_img11 = img[mid_height-small_height-mid_image_size:mid_height-small_height+mid_image_size, width-image_size:width]
            augmented_testX_1.append(crop_img1)
            augmented_testX_2.append(crop_img2)
            augmented_testX_3.append(crop_img3)
            augmented_testX_4.append(crop_img4)
            augmented_testX_5.append(crop_img5)
            augmented_testX_6.append(crop_img6)
            augmented_testX_7.append(crop_img7)
            augmented_testX_8.append(crop_img8)
            augmented_testX_9.append(crop_img9)
            augmented_testX_10.append(crop_img10)
            augmented_testX_11.append(crop_img11)

        augmented_testX_1 = np.array(augmented_testX_1)
        augmented_testX_2 = np.array(augmented_testX_2)
        augmented_testX_3 = np.array(augmented_testX_3)
        augmented_testX_4 = np.array(augmented_testX_4)
        augmented_testX_5 = np.array(augmented_testX_5)
        augmented_testX_6 = np.array(augmented_testX_6)
        augmented_testX_7 = np.array(augmented_testX_7)
        augmented_testX_8 = np.array(augmented_testX_8)
        augmented_testX_9 = np.array(augmented_testX_9)
        augmented_testX_10 = np.array(augmented_testX_10)
        augmented_testX_11 = np.array(augmented_testX_11)
        testX = np.concatenate((augmented_testX_1, augmented_testX_2, augmented_testX_3, augmented_testX_4, augmented_testX_5, augmented_testX_6, augmented_testX_7, augmented_testX_8, augmented_testX_9, augmented_testX_10, augmented_testX_11))
        # testXflipped = []
        # for img in testX:
        #     horizontal_flip = cv2.flip( img, 0 )
        #     testXflipped.append(horizontal_flip)
        # testXflipped = np.array(testXflipped)
        # testX = np.concatenate((testX, testXflipped))
        testY = np.repeat(testY, number)
        return (testX, testY)

    def create_montages(self, images, montage_image_number, image_size, full_montage_image_size):
        output = []
        if montage_image_number == 4:
            data = images.reshape(int(len(images)/montage_image_number), montage_image_number, image_size, image_size, 3)

            for iter in range(len(data)):
                img_set = data[iter]
                outputImage = np.zeros((full_montage_image_size, full_montage_image_size, 3))
                outputImage[0:image_size, 0:image_size, :] = img_set[0]
                outputImage[0:image_size, image_size:2*image_size, :] = img_set[1]
                outputImage[image_size:2*image_size, 0:image_size, :] = img_set[2]
                outputImage[image_size:2*image_size, image_size:2*image_size, :] = img_set[3]

                # cv2.imshow("Result", outputImage)
                # cv2.waitKey(0)
                # raise Exception('Exit')

                output.append(outputImage)
        else:
            raise Exception('Only implemented to montage 4 images into one image')

        return np.array(output)

    def process_cnn_data(self, images, aux_data, num_unique_stock_ids, num_unique_image_types, num_unique_time_days, image_size, keras_model_type, data_augmentation, data_augmentation_test, montage_image_number, full_montage_image_size, output_autoencoder_model_file_path, log_file_path):

        if log_file_path is not None:
            sys.stderr = open(log_file_path, 'a')

        def eprint(*args, **kwargs):
            print(*args, file=sys.stderr, **kwargs)

        trainX = []
        testX = []
        trainY = []
        testY = []

        datagen = self.get_imagedatagenerator()

        datagen.fit(images)
        images = datagen.standardize(images)

        aux_data["value"] = aux_data["value"].astype(float)
        output_image_file = aux_data["output_image_file"].tolist()

        # LSTM models group images by time, but are still ties to a single label e.g. X, Y = [img_t1, img_t2, img_t3], y1.
        if keras_model_type == 'densenet121_lstm_imagenet':
            images = images.reshape(num_unique_stock_ids * num_unique_image_types, num_unique_time_days, input_image_size, input_image_size, 3)

            (train_aux_data, test_aux_data, train_images, test_images) = train_test_split(aux_data, images, test_size=0.2)
            trainX_length = len(train_images) 
            testX_length = len(test_images)
            train_images = train_images.reshape(trainX_length * num_unique_time_days, input_image_size, input_image_size, 3)
            test_images = test_images.reshape(testX_length * num_unique_time_days, input_image_size, input_image_size, 3)
            trainX_length_flat = len(train_images)

            test_images = datagen.standardize(test_images)

            # (testX, testY) = self.generate_croppings(testX, testY, image_size, data_augmentation_test)
            testX_resized = []
            for img in test_images:
                testX_resized.append(cv2.resize(img, (image_size, image_size)))
            test_images = np.array(testX_resized)

            test_images = test_images.reshape(data_augmentation_test * testX_length, num_unique_time_days, image_size, image_size, 3)

            # trainX_aug = []
            # trainY_aug = []
            # augmented = datagen.flow(train_images, train_aux_data, batch_size=trainX_length_flat)
            # for i in range(0, data_augmentation):
            #     X, y = augmented.next()
            #     if len(trainX_aug) == 0:
            #         trainX_aug = X
            #         trainY_aug = y
            #     else:
            #         trainX_aug = np.concatenate((trainX_aug, X))
            #         trainY_aug = np.concatenate((trainY_aug, y))
            # 
            # trainX = trainX_aug
            # trainY = trainY_aug

            trainX_resized = []
            for img in train_images:
                trainX_resized.append(cv2.resize(img, (image_size, image_size)))
            train_images = np.array(trainX_resized)

            train_images = train_images.reshape(data_augmentation * trainX_length, num_unique_time_days, image_size, image_size, 3)
        else:
            images = self.create_montages(images, montage_image_number, image_size, full_montage_image_size)

            (encoder, decoder, autoencoder) = self.build_autoencoder(full_montage_image_size, full_montage_image_size, 3)
            opt = Adam(lr=1e-3)
            autoencoder.compile(loss="mse", optimizer=opt)

            (train_aux_data, test_aux_data, train_images, test_images) = train_test_split(aux_data, images, test_size=0.2)

            checkpoint = ModelCheckpoint(filepath=output_autoencoder_model_file_path, monitor='loss', verbose=1, save_best_only=True, mode='min', save_frequency=1, save_weights_only=False)
            callbacks_list = [checkpoint]

            # train the convolutional autoencoder
            H = autoencoder.fit(
                train_images, train_images,
                validation_data=(test_images, test_images),
                epochs=25,
                batch_size=32,
                callbacks=callbacks_list
            )
            decoded = autoencoder.predict(images)

            output_image_counter = 0
            for image in decoded:
                cv2.imwrite(output_image_file[output_image_counter], image*255)
                output_image_counter += 1

            (train_aux_data, test_aux_data, train_images, test_images) = train_test_split(aux_data, decoded, test_size=0.2)
            # testY_length = len(testY)

            # (testX, testY) = self.generate_croppings(testX, testY, image_size, data_augmentation_test)
            # testY = testY.reshape(data_augmentation_test * testY_length, 1)

            # augmented = datagen.flow(trainX, trainY, batch_size=len(trainX))
            # for i in range(0, data_augmentation):
            #     X, y = augmented.next()

        stock_id_binarizer = LabelBinarizer().fit(aux_data["stock_id"])
        train_stock_id_categorical = stock_id_binarizer.transform(train_aux_data["stock_id"])
        test_stock_id_categorical = stock_id_binarizer.transform(test_aux_data["stock_id"])

        accession_id_binarizer = LabelBinarizer().fit(aux_data["accession_id"])
        train_accession_id_categorical = accession_id_binarizer.transform(train_aux_data["accession_id"])
        test_accession_id_categorical = accession_id_binarizer.transform(test_aux_data["accession_id"])

        female_id_binarizer = LabelBinarizer().fit(aux_data["female_id"])
        train_female_id_categorical = female_id_binarizer.transform(train_aux_data["female_id"])
        test_female_id_categorical = female_id_binarizer.transform(test_aux_data["female_id"])

        male_id_binarizer = LabelBinarizer().fit(aux_data["male_id"])
        train_male_id_categorical = male_id_binarizer.transform(train_aux_data["male_id"])
        test_male_id_categorical = male_id_binarizer.transform(test_aux_data["male_id"])

        continuous = [col for col in aux_data.columns if 'aux_trait_' in col]
        cs = MinMaxScaler()
        if len(continuous) > 0:
            trainContinuous = cs.fit_transform(train_aux_data[continuous])
            testContinuous = cs.transform(test_aux_data[continuous])

            #trainX = np.hstack((train_stock_id_categorical, train_accession_id_categorical, train_female_id_categorical, train_male_id_categorical, trainContinuous))
            #testX = np.hstack((test_stock_id_categorical, test_accession_id_categorical, test_female_id_categorical, test_male_id_categorical, testContinuous))
            trainX = trainContinuous
            testX = testContinuous
        else:
            trainX = []
            testX = []
        trainx = np.array(trainX)
        testx = np.array(testX)

        max_label = aux_data["value"].max()
        trainY = train_aux_data["value"]/max_label
        testY = test_aux_data["value"]/max_label

        train_genotype_files = train_aux_data["genotype_file"].tolist()
        test_genotype_files = test_aux_data["genotype_file"].tolist()
        train_genotype_data = []
        for f in train_genotype_files:
            if log_file_path is not None:
                eprint(f)
            else:
                print(f)
            if pd.isna(f) is False:
                geno_data = pd.read_csv(f, sep="\t", header=None, na_values="NA")
                train_genotype_data.append(np.array(geno_data.iloc[:,0]))
        test_genotype_data = []
        for f in test_genotype_files:
            if log_file_path is not None:
                eprint(f)
            else:
                print(f)
            if pd.isna(f) is False:
                geno_data = pd.read_csv(f, sep="\t", header=None, na_values="NA")
                test_genotype_data.append(np.array(geno_data.iloc[:,0]))

        train_genotype_data = np.array(train_genotype_data)
        test_genotype_data = np.array(test_genotype_data)
        eprint(train_genotype_data)
        eprint(testX)
        eprint(trainX)

        return (test_images, np.array(testX), testY.to_numpy(), test_genotype_data, train_images, np.array(trainX), trainY.to_numpy(), train_genotype_data)

    def process_cnn_data_predictions(self, data, aux_data, num_unique_stock_ids, num_unique_image_types, num_unique_time_days, image_size, keras_model_type, input_autoencoder_model_file_path, training_data, data_augmentation_test, montage_image_number, full_montage_image_size):
        trainX = []
        testX = []
        trainY = []
        testY = []

        datagen = self.get_imagedatagenerator()
        datagen.fit(training_data)
        data = datagen.standardize(data)

        output_image_file = aux_data["output_image_file"].tolist()

        data = self.create_montages(data, montage_image_number, image_size, full_montage_image_size)

        autoencoder_model = load_model(input_autoencoder_model_file_path)
        data = autoencoder_model.predict(data)

        #ret = self.generate_croppings(data, None, image_size, data_augmentation_test)
        #augmented_data = ret[0]

        # LSTM models group images by time, but are still ties to a single label e.g. X, Y = [img_t1, img_t2, img_t3], y1.
        if keras_model_type == 'KerasCNNLSTMDenseNet121ImageNetWeights':
            data = data.reshape(data_augmentation_test * num_unique_stock_ids * num_unique_image_types, num_unique_time_days, image_size, image_size, 3)

        output_image_counter = 0
        for image in data:
            cv2.imwrite(output_image_file[output_image_counter], image*255)
            output_image_counter += 1

        stock_id_binarizer = LabelBinarizer().fit(aux_data["stock_id"])
        stock_id_categorical = stock_id_binarizer.transform(aux_data["stock_id"])

        accession_id_binarizer = LabelBinarizer().fit(aux_data["accession_id"])
        accession_id_categorical = accession_id_binarizer.transform(aux_data["accession_id"])

        female_id_binarizer = LabelBinarizer().fit(aux_data["female_id"])
        female_id_categorical = female_id_binarizer.transform(aux_data["female_id"])

        male_id_binarizer = LabelBinarizer().fit(aux_data["male_id"])
        male_id_categorical = male_id_binarizer.transform(aux_data["male_id"])

        continuous = [col for col in aux_data.columns if 'aux_trait_' in col]
        cs = MinMaxScaler()
        if len(continuous) > 0:
            fitContinuous = cs.fit_transform(aux_data[continuous])

            # fitX = np.hstack([stock_id_categorical, accession_id_categorical, female_id_categorical, male_id_categorical, fitContinuous])
            fitX = fitContinuous
        else:
            # fitX = np.hstack([stock_id_categorical, accession_id_categorical, female_id_categorical, male_id_categorical])
            fitX = []
        fitX = np.array(fitX)

        max_label = aux_data["value"].max()
        fitY = aux_data["value"]/max_label

        genotype_files = aux_data["genotype_file"].tolist()
        genotype_data = []
        for f in genotype_files:
            if pd.isna(f) is False:
                geno_data = pd.read_csv(f, sep="\t", header=None, na_values="NA")
                genotype_data.append(np.array(geno_data.iloc[:,0]))

        genotype_data = np.array(genotype_data)

        return (data, fitX, genotype_data, fitY.to_numpy())

    def build_autoencoder(self, width, height, depth, filters=(32, 64), latentDim=16):
        inputShape = (height, width, depth)
        chanDim = -1

        # define the input to the encoder
        inputs = Input(shape=inputShape)
        x = inputs

        # loop over the number of filters
        for f in filters:
            # apply a CONV => RELU => BN operation
            x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=chanDim)(x)

        # flatten the network and then construct our latent vector
        volumeSize = K.int_shape(x)
        x = Flatten()(x)
        latent = Dense(latentDim)(x)

        # build the encoder model
        encoder = Model(inputs, latent, name="encoder")

        # start building the decoder model which will accept the
        # output of the encoder as its inputs
        latentInputs = Input(shape=(latentDim,))
        x = Dense(np.prod(volumeSize[1:]))(latentInputs)
        x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

        # loop over our number of filters again, but this time in
        # reverse order
        for f in filters[::-1]:
            # apply a CONV_TRANSPOSE => RELU => BN operation
            x = Conv2DTranspose(f, (3, 3), strides=2, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=chanDim)(x)

        # apply a single CONV_TRANSPOSE layer used to recover the
        # original depth of the image
        x = Conv2DTranspose(depth, (3, 3), padding="same")(x)
        outputs = Activation("sigmoid")(x)

        # build the decoder model
        decoder = Model(latentInputs, outputs, name="decoder")

        # our autoencoder is the encoder + decoder
        autoencoder = Model(inputs, decoder(encoder(inputs)), name="autoencoder")

        # return a 3-tuple of the encoder, decoder, and autoencoder
        return (encoder, decoder, autoencoder)
