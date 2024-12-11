#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:11:04 2024

@author: loris

derived from mediapipe implementation of hand tracking and gesture recognition of Google
 (https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer)

"""


"""
Go through the code carefully before running it
Read the annotations and ask questions (to the internet, chatGPT, or the assistents)

"""


""" IMPORTS """

# pip install opencv-contrib-python==4.9.0.80
# pip install mediapipe==0.10.9

import cv2
from HandRecognizer import HandRecognizer   # mediapipe dependency


""" VIDEO STREAM """

# Remove previous windows if still existing
cv2.destroyAllWindows()

# TODO: initialize video capture
## YOUR CODE HERE ##


""" VARIABLES """

# Path to gesture recognizer model
model_path = ''

# Initialize hand recognizer
hand_recognizer = HandRecognizer(model_path)


""" MAIN LOOP """

while True:

    # TODO: Get a new frame
    ## YOUR CODE HERE ##
    
    # TODO: Translate BGR frame to RGB frame for Mediapipe
    ## YOUR CODE HERE ##

    # Detect the hand by applying the hand recognizer on your new RGB frame
    hand_landmarks, gestures = hand_recognizer.detect(rgb_frame)
    # NOTE: Note that the hand_landmarks variable and gestures variable are defined here but remain unused. You can use them further for more advanced tasks. More information on those variables can be found at https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/python#handle_and_display_results
    
    # Process hands on the orginal frame
    recognition_frame = hand_recognizer.show_hands(frame)
    
    # Remove the following line to unblock the code:
    raise SystemExit("Go through the code carefully and remove this error to make the code run!\n")
    
    # TODO: Display resulting recognition frame
    ## YOUR CODE HERE ##

    # Stop the program if the ESC key is pressed
    if cv2.waitKey(1) == 27:
        break


""" CLOSE VIDEO STREAM """

hand_recognizer.close()
cap.release()
cv2.destroyAllWindows()
