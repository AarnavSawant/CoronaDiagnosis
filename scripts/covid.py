import numpy as np
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import os

#Exploratory Data Analysis
plt.clf()
diagnosis = ["COVID", "NORMAL", "PNEUMONIA"]
num_COVID = len(os.listdir("dataset/COVID-19"))
num_Pneumonia = len(os.listdir("dataset/ViralPneumonia"))
num_NORMAL = len(os.listdir("dataset/NORMAL"))
y_pos = np.arange(len(diagnosis))
plt.barh(y_pos, [num_COVID, num_NORMAL, num_Pneumonia], align='center')
plt.title("Number of Images per Category")
plt.yticks(y_pos, diagnosis, fontsize=8)
plt.xlabel("Number of Images")
plt.savefig("charts/NumberOfImages.jpg")
plt.clf()


#Building the CNN
classifier = Sequential()
classifier.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(rate=0.1))
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(rate=0.1))
classifier.add(Dense(units=3, activation='softmax'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Reading Images from Training, Validation, and Test Directories and converting them to tensor format
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory("dataset/train_data", target_size=(64, 64), batch_size=32, class_mode='categorical')
validation_set = validation_datagen.flow_from_directory("dataset/validation_data", target_size=(64, 64), batch_size=32, class_mode='categorical')
test_set = test_datagen.flow_from_directory("dataset/test_data", target_size=(64, 64), batch_size=32, class_mode='categorical', shuffle=False)
history = classifier.fit_generator(training_set, epochs=25, steps_per_epoch=int(2048/32), validation_data=validation_set, validation_steps=436)
classifier.save("models/3Conv2D/Model.h5")
classifier.save_weights("models/3Conv2D/ModelWeights.h5")

#Option to Save Model as JSON
model_json = classifier.to_json()
with open("models/2Conv2DDropout/Model.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights("models/2Conv2DDropout/Weights.h5")
classifier.save("models/2Conv2DDropout/Model.h5")
classes = ["COVID", "NORMAL", "Pneumonia"]

#Preprocessing a list of filenames
def prepare(filepath):
    list = []
    for path in filepath:
        IMG_SIZE = 64
        img = image.load_img(filepath, target_size=(64, 64))
        img_tensor = image.img_to_array(img)
        img_tensor /= 255
        list.append(img_tensor)
    return list

#Preprocessing a single image
def prepare_single_image(filepath):
    img = image.load_img(filepath, target_size=(64, 64))
    img_tensor = image.img_to_array(img)
    img_tensor /= 255
    return img_tensor

#Making Single Predictions
img_path = "dataset/train_data/COVID-19/COVID-19(135).png"
img = prepare_single_image(img_path)
img = img.reshape([1, 64, 64, 3])
pred = classifier.predict(img)
prediction = np.argmax(pred)
# test_pred = classifier.predict_generator(test_set)
print("DefaultNN" + diagnosis[prediction])
np.set_printoptions(precision=3, suppress=True)
print(pred)

#Plotting Training Accuracy and Validation Accuracy vs Epoch Number
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig("charts/train_validation_accuracy.jpg")

#Plotting Training Loss and Validation Loss vs Epoch Number
plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig("charts/train_validation_loss.jpg")

#Confusion Matrix for untested Test Set
plt.clf()
true_classes = test_set.classes
class_labels = list(test_set.class_indices.keys())
test_pred = classifier.predict_generator(test_set)
predicted_classes = np.argmax(test_pred, axis=1)
cm = confusion_matrix(true_classes, predicted_classes)
df_cm = pd.DataFrame(cm, range(3), range(3))
pd.options.display.float_format = '%d'.format
sns.set(font_scale=1.4)
sns.heatmap(df_cm, annot=True, annot_kws={"size" : 16}, fmt='g')
plt.title("Confusion Matrix")
plt.savefig("charts/confusion_matrix.jpg")
