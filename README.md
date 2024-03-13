# dog_cat_classifier_

This repository contains code for a Convolutional Neural Network (CNN) model trained to classify images as either dogs or cats. The model is implemented using TensorFlow and Keras, and it is served using Flask for making predictions on new images.

Dataset
The dataset used for training the model is the "Dogs vs. Cats" dataset available on Kaggle. It consists of a large number of images of dogs and cats.

Model Architecture
The CNN model architecture consists of multiple convolutional layers followed by batch normalization, max-pooling, and dropout layers to prevent overfitting. The model architecture is summarized as follows:

Input layer: Accepts images of size 256x256 pixels with 3 color channels (RGB).
Convolutional layers with ReLU activation.
Batch normalization after each convolutional layer.
Max-pooling layers for downsampling.
Flatten layer to convert the 2D output into a 1D vector.
Fully connected layers with ReLU activation and dropout for regularization.
Output layer with a sigmoid activation function to output probabilities for binary classification.
Flask Implementation
A Flask web application is provided to serve the trained model. The application exposes an endpoint /predict that accepts POST requests with image data. It preprocesses the image, makes predictions using the trained model, and returns the predicted class label along with class probabilities.

Files
main_code.py: Contains the code for training the CNN model, preprocessing data, and evaluating the model's performance.
flask_implementation.py: Implements a Flask web server to serve the trained model for making predictions.
model_test.py: Demonstrates how to load the trained model and make predictions on new images.
README.md: Instructions and information about the repository (this file).
Usage
Clone the repository to your local machine.
Train the model using the provided dataset or replace it with your own dataset.
Run the Flask application (flask_implementation.py) to serve the trained model.
Use the provided model_test.py script to make predictions on new images.
Requirements
Python 3.x
TensorFlow
Keras
Flask
NumPy
Matplotlib
OpenCV (cv2)
PIL (Python Imaging Library)
Note
Make sure to update file paths and configurations as per your system setup before running the code.
