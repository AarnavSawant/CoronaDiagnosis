# CoronaDiagnosis
Uses a Convolutional Neural Network to diagnose COVID-19. The CNN classifies X-Rays to diagnose images as COVID-19, NORMAL, or VIRAL Pneumonia

About the Scripts Directory:
The ```covid.py``` file contains all of code for the Convolutional Neural Network. It has code to preprocess the images, create the layers of the neural network, and make predictions on either single images or sets of images. It also has code for a confusion matrix.
The ```COVID.ipynb``` file is just the Jupyter Notebook version of ```covid.py```. The Notebook contains important figures.


Instructions on How to Use this WebApp:
1. Clone this repo
2. Navigate to the webapp directory and create a virual environment using python3. If you do not have virtualenv installed. Do ```pip install virtualenv```. Once you have virtualenv, perform the command ```python3 -m venv venv```. This creates a virtual environment. Activate the virtual environment by writing ```source venv/bin/activate``` Download the necessary libraries including flask, numpy, tensorflow. 
3. Do ```FLASK_APP=app.py```
4. Do ```flask run```
5. The app should then be good to go!
