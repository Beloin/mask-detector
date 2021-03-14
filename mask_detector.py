# import the necessary packages
import time
from typing import Tuple
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os


class MaskDetector:
    """ Detect face and if user is using mask, or not """

    def __init__(self, image, face, model, confidence, debug=False) -> None:
        self.debug = debug
        self.image = image
        self.face = face
        self.model = model
        self.confidence = confidence

    def debug_print(self, toPrint):
        """Prints if debug is enabled"""
        if (self.debug):
            print(toPrint)

    def predict(self):
        net, model = self.__load()
        image = self.__predict(net, model)
        self.__show_and_save(image)

    def __load(self):
        # load our serialized face detector model from disk
        self.debug_print("[INFO] loading face detector model...")
        prototxtPath = os.path.sep.join([self.face, "deploy.prototxt"])
        weightsPath = os.path.sep.join([self.face,
                                        "res10_300x300_ssd_iter_140000.caffemodel"])
        net = cv2.dnn.readNet(prototxtPath, weightsPath)
        # load the face mask detector model from disk
        self.debug_print("[INFO] loading face mask detector model...")
        model = load_model(self.model)

        return net, model

    def __predict(self, net, model):
        # load the input image from disk, clone it, and grab the image spatial
        # dimensions
        image = cv2.imread(self.image)
        orig = image.copy()
        (h, w) = image.shape[:2]
        # construct a blob from the image
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        self.debug_print("[INFO] computing face detections...")
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > self.confidence:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = image[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)
                # pass the face through the model to determine if the face
                # has a mask or not
                (mask, withoutMask) = model.predict(face)[0]

                # determine the class label and color we'll use to draw
                # the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                # include the probability in the label
                label = "{}: {:.2f}%".format(
                    label, max(mask, withoutMask) * 100)
                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(image, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

        return image

    def __show_and_save(self, image):
        # show the output image
        cv2.imshow("Output", image)
        cv2.waitKey(0)

        timestamp = round(time.time())
        cv2.imwrite(f'./checked/checked{timestamp}' + '.jpg', image)
