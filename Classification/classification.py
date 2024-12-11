#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:35:26 2024

@author: loris

"""

""" """ """ """ """ """ """ """ """ """ """ """ """ """ """ """ """ """
""" """ """ """ """ """ """ """ """ """ """ """ """ """ """ """ """ """
"""                                                                 """
"""                 Please read the comments carefully              """
"""             Inform yourself on concepts you are using           """
"""                                                                 """
"""         Questions will be asked about machine learning!         """
"""                                                                 """
"""                                                                 """
""" """ """ """ """ """ """ """ """ """ """ """ """ """ """ """ """ """
""" """ """ """ """ """ """ """ """ """ """ """ """ """ """ """ """ """

""" IMPORTS """

import joblib       # copy the following in your console: conda install joblib
from Classifiers import RandomForest, AdaBoost # These classes require the following packages: numpy, sklearn. Details on installation in Classifiers.py (line 16-21)


""" INIATALIZATION OF TRAINER """

# TODO: Anser the questions on line 252-257 before using the rest of the code

# Initialize Random Forest classification trainer
random_forest = RandomForest()

# Initialize Adaptive Boosting classification trainer
adaboost = AdaBoost()


""" DATASET """

# TODO: The datset need to be given in a certain structure. Check Classsifiers.py (line 268-283) for more details

# Path to dataset
path_to_dataset = 'Absolute/path/to/dataset'

# TODO: Define how to extract the information of one sample from a file in your dataset. Check Classifiers.py (line 288-319) for examples

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

model_path = '/Absolute/path/to/model'
random_forest_classifier = joblib.load(model_path)

X_input = [ [0,0,0,0,0],
            [1,1,1,1,1] ]
            
y_output = random_forest_classifier.predict(X_input)
print(str(y_output))
