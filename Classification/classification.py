#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:35:26 2024

@author: loris
"""


""" IMPORTS """

# Installl the packages if this is not done!
#   - numpy
#   - sklearn
#   - matplotlib
#   - joblib

import joblib
from Classifiers import RandomForest


""" TEST """

# Things you should know before using these classifiers
#    1. What is an input, what is a feature, what is an output?
#    2. What is a parameter, hyperparameter, architecture?
#    3. Which type of classifier are RandomForest and AdaBoost based on? Explain the principle.
#    4. How does RandomForest extend this principle? And AdaBoost?
#    5. How do we use data to train RandomForest and AdaBoost classifiers?


""" INITIALIZING TRAINER """

# TODO: Initialize Random Forest classification trainer

# Radnom Forest classification trainer
random_forest = "YOUR CODE HERE"


""" LOADING DATASET """

# TODO: Setup your dataset with the correct structure -> Check Canvas, Github or Classsifiers.py (line 268-283) for more details

# Path to dataset
path_to_dataset = 'Absolute/path/to/dataset'


# TODO: Define how to extract the information of one sample from a file in your dataset -> Check Canvas, Github or Classifiers.py (line 288-319) for more details

# Function to extract features
def get_sample(data_sample_path):
    # load sample
    sample = "YOUR CODE HERE"
    # process sample
    processed_sample = "YOUR CODE HERE"
    # return feature vector
    feature_vector = "YOUR CODE HERE"
    return feature_vector

# Load dataset
random_forest.dataset(path_to_dataset, get_sample)


""" TRAINING MODEL """

# Train on data
random_forest.train()


""" USING MODEL """

# Load model
model_path = '/Absolute/path/to/model'
random_forest_classifier = joblib.load(model_path)

# Get input
X_input = None

# Predict output
y_output = random_forest_classifier.predict(X_input)
print(str(y_output))
