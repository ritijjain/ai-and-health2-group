# Diabetes Prediction using Neural Network(Ai-and-health 2)
## Overview:
This is a Python code using neural network to make diabetes predictions(CMPSC 442 Final project FNN and RNN.ipynb). The code reads a diabetes dataset (diabetes.csv), preprocesses the data, builds a neural network model with cross validation, and uses it to predict diabetes.

## Requirement:
Install and import necessary packages for the projects, such as sklearn, keras etc.
<h4 align="left">Libraries/Frameworks used:</h4>
<p align="left">* Tensorflow
* Keras API
* numpy
* pandas
* sklearn
* matplotlib</p>
<h4 align="left">Languages used:</h4>
<p align="left"> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> </p>

## Neural Network Model with Cross Validation:
The neural network model is built using the Keras library. The Sequential model is used to create a linear stack of layers. The Dense layer is used to create fully connected layers in the neural network. 

Cross validation is performed using the StratifiedKFold method from the sklearn.model_selection library. This method splits the data into k folds and ensures that each fold has a similar distribution of target classes. This helps to ensure that the model is trained on a representative sample of the data.

## Hyperparameter Tuning:
GridSearchCV method from sklearn.model_selection library is used to find the best hyperparameters for the neural network model. This method searches over a specified range of hyperparameters and selects the best combination based on the specified scoring method. In this code, the hyperparameters searched over include number of neurons, activation function, optimizer and number of epochs.
