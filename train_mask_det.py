# import packages to train DL model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os


class MaskDetectionTrainer:
    def __init__(self, dataset, plot, model, debug=False) -> None:
        self.dataset = dataset
        self.plot = plot
        self.model = model

        self.data = []
        self.labels = []

    def load_images(self):
        self.debug_print("[INFO] loading images...")
        imagePaths = list(paths.list_images(self.dataset))

        # loop over the image paths
        for imagePath in imagePaths:
            # extract the class label from the filename
            label = imagePath.split(os.path.sep)[-2]
            # load the input image (224x224) and preprocess it
            image = load_img(imagePath, target_size=(224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)
            # update the data and labels lists, respectively
            self.data.append(image)
            self.labels.append(label)

        # convert the data and labels to NumPy arrays
        self.data = np.array(self.data, dtype="float32")
        self.labels = np.array(self.labels)

    def pre_process(self):
        # perform one-hot encoding on the labels
        lb = LabelBinarizer()
        self.labels = lb.fit_transform(self.labels)
        self.labels = to_categorical(self.labels)

    def create_model():
        # load the MobileNetV2 network, ensuring the head FC layer sets are
        # left off
        baseModel = MobileNetV2(weights="imagenet", include_top=False,
                                input_tensor=Input(shape=(224, 224, 3)))
        # construct the head of the model that will be placed on top of the
        # the base model
        headModel = baseModel.output
        headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(128, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(2, activation="softmax")(headModel)
        # place the head FC model on top of the base model (this will become
        # the actual model we will train)
        model = Model(inputs=baseModel.input, outputs=headModel)

        # loop over all layers in the base model and freeze them so they will
        # *not* be updated during the first training process
        for layer in baseModel.layers:
            layer.trainable = False

        return model

    def compile(self, model):
        # construct the training image generator for data augmentation
        aug = ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest")

        # partition the data into training and testing splits using 80% of
        # the data for training and the remaining 20% for testing
        (trainX, testX, trainY, testY) = train_test_split(self.data, self.labels,
                                                          test_size=0.20,
                                                          stratify=self.labels, random_state=42)

        # compile our model
        self.debug_print("[INFO] compiling model...")
        opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
        model.compile(loss="binary_crossentropy", optimizer=opt,
                      metrics=["accuracy"])
        # train the head of the network
        self.debug_print("[INFO] training head...")
        H = model.fit(
            aug.flow(trainX, trainY, batch_size=BS),
            steps_per_epoch=len(trainX) // BS,
            validation_data=(testX, testY),
            validation_steps=len(testX) // BS,
            epochs=EPOCHS)



    def debug_print(self, toPrint):
        """Prints if debug is enabled"""
        if (self.debug):
            print(toPrint)