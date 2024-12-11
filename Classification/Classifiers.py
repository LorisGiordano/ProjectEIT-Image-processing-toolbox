#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:04:59 2024

@author: loris
"""


""" IMPORTS """

# pre-installed with python
import os
import time

# instalation required
import numpy as np                  # copy the following in your console: conda install numpy
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier # copy the following in your console: pip install pip install scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import joblib                       # copy the following in your console: conda install joblib

# not included to not make installation too difficult
# import matplotlib.pyplot as plt     # copy the following in your console: conda install matplotlib


""" CLASSES """

class RandomForest():
    
    def __init__(self, n_estimators=100, max_depth=None):
        
        self.random_forest_classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.trained = False
        
        
    def dataset(self, data_directory, get_sample, test_size=0.2):
        
        # Initialize inputs and labels 
        X = []
        y = []
        i = 0
        # Go over directory
        for label in sorted(os.listdir(data_directory)):
            
            label_dir = os.path.join(data_directory, label)
            # Skip everything that is not a folder
            if not os.path.isdir(label_dir):
                continue
            # Print label of the folder
            print(f'Label: {label}')
            for data_sample in os.listdir(label_dir):
                data_sample_path = os.path.join(label_dir, data_sample)
                landmarks = get_sample(data_sample_path)
                if landmarks is not None:
                    X.append(landmarks)
                    y.append(label)
                i += 1
                if i == 10:
                    i = 0
                    break
    
        for i in range(3):
            Xi = X[i]
            yi = y[i]
            print('Example input and label:')
            print(f'{i+1}: \n{Xi} --> {yi}')
        print('\n')
        labels = np.unique(y)
        print('All different labels used for training:')
        print(labels)
        print('')
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=42)


    def train(self):
        
        if self.X_train is not None:
            print('Training...')
            self.random_forest_classifier.fit(self.X_train, self.y_train)
            
            start_time = time.time()
            y_pred = self.random_forest_classifier.predict(self.X_test)
            end_time = time.time()
            self.time_per_prediction = (end_time - start_time)/len(self.X_test)
            print('\nTime per prediction:')
            print(self.time_per_prediction)

            # Calculate confusion matrix
            self.conf_matrix = confusion_matrix(self.y_test, y_pred)
            print("\nConfusion Matrix:")
            print(self.conf_matrix)
            self.trained = True
        
            save_path= os.path.dirname(__file__) + '/random_forest_classifier.pkl'
            joblib.dump(self.random_forest_classifier, save_path)
            print('\nModel save to:')
            print(save_path)
            
        else:
            print('No dataset given')
           
            
    def get_classifier(self):
        
        if self.trained:
            return self.random_forest_classifier
    
    
    def get_confusion_matrix(self):
        
        if self.trained:
#            plt.imshow(self.conf_matrix, interpolation='nearest')
#            plt.title("Confusion matrix")
#            plt.colorbar()
#            tick_marks = np.arange(len(self.labels))
#            plt.xticks(tick_marks, self.labels, rotation=45)
#            plt.yticks(tick_marks, self.labels)
#
#            shape = np.shape(self.conf_matrix)
#            for i in range(shape[0]):
#                for j in range(shape[1]):
#                    plt.text(j, i, format(self.conf_matrix[i, j], 'd'),
#                             horizontalalignment="center",
#                             color="white")
#
#            plt.tight_layout()
#            plt.ylabel('True label')
#            plt.xlabel('Predicted label')
            return self.conf_matrix
        

class AdaBoost():
    
    def __init__(self, n_estimators=50):
        
        self.adaptive_boosting_classifier = AdaBoostClassifier(n_estimators=n_estimators)
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.trained = False
        
        
    def dataset(self, data_directory, get_sample, test_size=0.2):
        
        X = []
        y = []
        
        for label in sorted(os.listdir(data_directory)):
            
            label_dir = os.path.join(data_directory, label)
            # Skip everything that is not a folder
            if not os.path.isdir(label_dir):
                continue
            # Print label of the folder
            print(f'Label: {label}')
            for data_sample in os.listdir(label_dir):
                data_sample_path = os.path.join(label_dir, data_sample)
                landmarks = get_sample(data_sample_path)
                if landmarks is not None:
                    X.append(landmarks)
                    y.append(label)
    
        for i in range(3):
            Xi = X[i]
            if len(Xi) > 10:
                Xi = "Inpout vector (1x" + str(len(Xi)) + ")"
            yi = y[i]
            print('Example input and label:')
            print(f'{i+1}: \n{Xi} --> {yi}')
        print('\n')
        self.labels = np.unique(y)
        print('All different labels used for training:')
        print(self.labels)
        print('')
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=42)


    def train(self):
        
        if self.X_train is not None:
            print('Training...')
            self.adaptive_boosting_classifier.fit(self.X_train, self.y_train)
            
            start_time = time.time()
            y_pred = self.adaptive_boosting_classifier.predict(self.X_test)
            end_time = time.time()
            self.time_per_prediction = (end_time - start_time)/len(self.X_test)
            print('\nTime per prediction:')
            print(self.time_per_prediction)

            # Calculate confusion matrix
            self.conf_matrix = confusion_matrix(self.y_test, y_pred)
            print("\nConfusion Matrix:")
            print(self.conf_matrix)
            self.trained = True
        
            save_path = os.path.dirname(__file__) + '/adaptive_boosting_classifier.pkl'
            joblib.dump(self.adaptive_boosting_classifier, save_path)
            print('\nModel save to:')
            print(save_path)
            
        else:
            print('No dataset given')
           
            
    def get_classifier(self):
        
        if self.trained:
            return self.adaptive_boosting_classifier
    
    
    def get_confusion_matrix(self):
        
        if self.trained:
#                plt.imshow(self.conf_matrix, interpolation='nearest')
#                plt.title("Confusion matrix")
#                plt.colorbar()
#                tick_marks = np.arange(len(self.labels))
#                plt.xticks(tick_marks, self.labels, rotation=45)
#                plt.yticks(tick_marks, self.labels)
#                
#                shape = np.shape(self.conf_matrix)
#                for i in range(shape[0]):
#                    for j in range(shape[1]):
#                        plt.text(j, i, format(self.conf_matrix[i, j], 'd'),
#                                 horizontalalignment="center",
#                                 color="white")
#            
#                plt.tight_layout()
#                plt.ylabel('True label')
#                plt.xlabel('Predicted label')
            return self.conf_matrix


if __name__ == '__main__':
    
    """ INIATALIZATION OF TRAINER """
    
    """
    Things you should know before using these classifiers
        1. What is an input, what is a feature, what is an output?
        2. What is a parameter, hyperparameter, architecture?
        3. Which type of classifier are RandomForest and AdaBoost based on? Explain the principle.
        4. How does RandomForest extend this principle? And AdaBoost?
        5. How do we use data to train RandomForest and AdaBoost classifiers?
    """
    
    # Initialize Random Forest classification trainer
    random_forest = RandomForest()
    adaboost = AdaBoost()
    
    
    """ DATASET """
    
    """
    The dataset should be given as follows:
        DatasetFolder
            - label_1
                * sample_1
                * sample_2
                * ...
                * sample_P
            - label_2
                * sample_1
                * ...
                * sample_Q
            - ...
            - label_N
                * sample_1
                * ...
                * sample_R
    """
    
    # Define how to extract the information of one sample from a file in your dataset
    
    # example 1: text-file or csv-file
    import numpy as np
    def get_sample_example1(data_sample_path):
        # load sample
        sample = np.loadtxt(data_sample_path)
        # process sample, example here: z-scoring
        mean = np.mean(sample)
        std = np.std(sample)
        z_scored_sample = (sample-mean)/std
        # return sample
        return z_scored_sample
    
    # example 2: hand landmarks from images
    import mediapipe as mp
    import cv2
    mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    
    def get_sample_example2(data_sample_path):
        # load sample
        image = cv2.imread(data_sample_path)
        # process sample
        results = mp_hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            landmark_list = []
            for ldm in landmarks:
                landmark_list.extend([ldm.x, ldm.y, ldm.z])
            # return sample
            return landmark_list
        else:
            return None
    
    # define you own sample retrieval function
    def get_sample(data_sample_path):
        # load sample
        sample = ""
        # process sample
        processed_sample = ""
        # return sample
        return processed_sample
    
    # Path to dataset
    path_to_dataset = 'Absolute/path/to/dataset'
    
    # Load data
    random_forest.dataset(path_to_dataset, get_sample)
    
    
    """ TRAINING """
    
    # Train on data
    random_forest.train()
