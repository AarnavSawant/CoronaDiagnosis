
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.models import load_model
import flask
from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf

app = Flask(__name__)


def get_model():
    global graph, model
    graph = tf.get_default_graph()
    model = load_model("3Conv2D/Model.h5")
    model.load_weights("3Conv2D/ModelWeights.h5")

get_model()
def prepare_single_image(filepath):
    img = image.load_img(filepath, target_size=(64, 64))
    img_tensor = image.img_to_array(img)
    img_tensor /= 255
    return img_tensor

@app.route('/predict', methods=['POST'])
def make_prediction():
    file = request.files['image']
    if not file:
        return render_template('index.html', label="No file")
    preproccesed_file = prepare_single_image(file)
    preproccesed_file = preproccesed_file.reshape([1, 64, 64, 3])
    with graph.as_default():
        probabilities = model.predict(preproccesed_file)
    label = np.argmax(probabilities)
    classes = ["COVID", "NORMAL", "Pneumonia"]
    return render_template("index.html", label=str(classes[label]), COVID=str(probabilities[0, 0] * 100), NORMAL=(probabilities[0, 1] * 100), Pneumonia=(probabilities[0, 2] * 100))

@app.route("/")
@app.route("/index")
def index():
   return flask.render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)