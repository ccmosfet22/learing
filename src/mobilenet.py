# Immport Libraries
from sklearn.utils import shuffle
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import cv2
import glob
import random

from IPython.display import Image
import matplotlib.pyplot as plt


def prepare_image(filepath):
    img = cv2.imread(filepath)
    img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img_result = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    return img_result


def data_proc(dataPath):

    dirList = glob.glob(dataPath+"*")  # list of all directories in dataPath
    print(dirList)
    dirList.sort()  # sorted in alphabetical order

    Y_data = []
    for i in range(len(dirList)):
        fileList = glob.glob(dirList[i]+'/*.png')
        [Y_data.append(i) for file in fileList]
    print(Y_data)
    X_data = []
    for i in range(len(dirList)):
        fileList = glob.glob(dirList[i]+'/*.png')
        [X_data.append(prepare_image(file))
         for file in fileList]
    X_data = np.asarray(X_data)

    # random shuffle
    X_data, Y_data = shuffle(X_data, Y_data, random_state=0)

    testNum = random.randint(0, len(X_data)-1)
    plt.imshow(X_data[testNum])

    # counting number of pictures of each class
    equilibre = []
    [equilibre.append(Y_data.count(i)) for i in range(len(dirList))]

    # Data Normalisation
    X_train = X_data / 255.0

    # One-hot encoding
    Y_train = to_categorical(Y_data)

    return X_train, Y_train


def train(X_train, Y_train, filename):

    base_model = MobileNetV2(input_shape=(224, 224, 3),
                             weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)  # FC layer 1
    x = Dense(64, activation='relu')(x)   # FC layer 2
    preds = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=preds)

    base_model.trainable = False

    model.summary()

    # Compile Model
    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the neural network
    model.fit(X_train, Y_train, batch_size=30, epochs=10, verbose=1)

    model.save("./"+str(filename)+".h5")


if __name__ == "__main__":

    dataPath = 'C:\\Users\\YF\\Desktop\\Chainwin\CNN_model_dvc\\data\\training_dvc_data\\'
    # TargetSize = (224, 224)

    proc_data = data_proc(dataPath)
    train(proc_data[0], proc_data[1], filename="test_model")
