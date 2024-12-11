#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:11:04 2024

@author: loris

"""


""" IMPORTS """

# pip install opencv-contrib-python==4.9.0.80
# pip install mediapipe==0.10.9

import os
import cv2

from HandRecognizer import HandRecognizer


""" VIDEO STREAM """

# Remove previous windows if still existing
cv2.destroyAllWindows()

# Initialize video capture
cap = cv2.VideoCapture(0)


""" VARIABLES """

# Path to gesture recognizer model
model_path = os.path.dirname(__file__) + '/gesture_recognizer.task'

# Initialize hand recognizer
hand_recognizer = HandRecognizer(model_path)


""" MAIN LOOP """

while True:

    # Get a new frame
    _, frame = cap.read()
    
    # Translate BGR frame to RGB frame for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run gesture recognizer on new frame
    hand_landmarks, gestures = hand_recognizer.detect(rgb_frame)
    
    # Process hands on original frame
    recognition_frame = hand_recognizer.show_hands(frame)
    
    # Display resulting frame (with hands)
    cv2.imshow('Hands', recognition_frame)

    # Stop the program if the ESC key is pressed
    if cv2.waitKey(1) == 27:
        break


""" CLOSE VIDEO STREAM """

hand_recognizer.close()
cap.release()
cv2.destroyAllWindows()
