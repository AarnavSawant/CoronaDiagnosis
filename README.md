# CoronaDiagnosis
Uses a Convolutional Neural Network to classify Chest X-Rays as COVID, NORMAL, or VIRAL PNEUMONIA.

About the Scripts Directory:
The ```covid.py``` file contains all of code for the Convolutional Neural Network. It has code to preprocess the images, create the layers of the neural network, and make predictions on either single images or sets of images. It also has code for a confusion matrix.

Instructions on How to Use this WebApp:
Step 1: Clone this repo
Step 2: Navigate to the webapp directory and create a virual environment using python3. If you do not have virtualenv installed. Do ```pip install virtualenv```. Once you have virtualenv, perform the command ```python3 -m venv venv```. This creates a virtual environment. Download the necessary libraries including flask, numpy, tensorflow. 
Step 3: Do ```FLASK_APP=app.py```
Step 4: Do ```flask run```
Step 5: The app should then be good to go!
