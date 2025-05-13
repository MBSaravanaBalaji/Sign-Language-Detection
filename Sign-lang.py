import  cv2 
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

# holistic model
mp_holistic = mp.solutions.holistic
# drawaing utilities
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    # color convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image is no longer wrtieable 
    image.flags.writeable = False
    # make predictions
    results = model.process(image)
    # makes the image writeable again
    image.flags.writeable = True
    # color convert RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

cap = cv2.VideoCapture(0)
# set up the mediapipe model
with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
    while cap.isOpened():

        #read the feed
        ret, frame = cap.read()

        #make detections
        image, results = mediapipe_detection(frame, holistic)

    

        # show to screen
        cv2.imshow('OpenCV feed', frame)
        # break the loop properly 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    results.face_landmarks