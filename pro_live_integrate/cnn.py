from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os


# loading face detector from the place where we stored it
detectorPath = r".\face_detector"

protoPath = os.path.sep.join([detectorPath, "deploy.prototxt"])
#Loading the caffe model
caffemodelPath = os.path.sep.join([detectorPath,"res10_300x300_ssd_iter_140000.caffemodel"])
#reading data from the model.
net = cv2.dnn.readNetFromCaffe(protoPath, caffemodelPath)


modelPath = r".\model"

model = load_model(modelPath)
le_pickle_path = r".\le.pickle"
#le = pickle.loads(open(args["le"], "rb").read())
le = pickle.loads(open(le_pickle_path, "rb").read())


def live(frame):
    (h, w) = frame.shape[:2]
    temp = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    net.setInput(temp)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

          #staisfying the union need of veryfying through ROI and blink detection.
        if confidence > 0.5:
            #detect a bounding box
        #take dimensions
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        #get the dimensions
            (startX, startY, endX, endY) = box.astype("int")
            #print(box)

            startX = max(0, startX)

            startY = max(0, startY)

            endX = min(w, endX)

            endY = min(h, endY)

    # extract the face ROI and then preproces it in the exact
    # same manner as our training data
            face = frame[startY:endY, startX:endX]
            face = cv2.resize(face, (32, 32))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

    #pass the model to determine the liveness
            preds = model.predict(face)[0]
            j = np.argmax(preds)
            label = le.classes_[j]




    try:
        return [startX,startY,endX,endY,label]
    except UnboundLocalError:
        pass
