import tensorflow as tf
import keras
from keras.applications.mobilenet_v2 import MobileNetV2
import numpy as np
from imageio import imread
from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.mobilenet_v2 import decode_predictions
from keras.layers import Dense, Flatten
from keras.models import Model
from numpy import asarray
import pandas as pd
from PIL import Image
import cv2
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras import layers
import matplotlib.pyplot as plt

# creates a base model of mobilenet that's able to predict different things
def create_base_model(input_shape_arr=(224, 224, 3)):
    mbn = tf.keras.applications.MobileNetV2(weights='imagenet', input_shape=input_shape_arr)
    return mbn

# creates a custom model that utilizes transfer learning and mobilenet as the base model
def create_model_scratch(input_shape_arr=(224, 224, 3), num_classes = 2, train_dataset = []):
    mbn = tf.keras.applications.MobileNetV2(weights='imagenet', input_shape=input_shape_arr, include_top=False)

    # make a reference to MobileNet's input layer
    inp = mbn.input

    x = Flatten()(mbn.output)  # making a flatten layer and that wraps around the output of mobilenet

    # make a new softmax layer with num_classes neurons
    new_classification_layer = Dense(num_classes, activation='softmax')(x)

    # create a new network between inp and out
    model_new = Model(inp, new_classification_layer)

    # make all layers untrainable by freezing weights (except for last layer)
    for l, layer in enumerate(model_new.layers[:-1]):
        layer.trainable = False

    # ensure the last layer is trainable/not frozen
    for l, layer in enumerate(model_new.layers[-1:]):
        layer.trainable = True

    # compile the actual model
    model_new.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 0 = daisy
    # 1 = dandelion
    #train_labels = {'file_name': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
    #                 'type':[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]}
    # ^^ above is an alternative format of the label data set

    # the label dataset is created here. when it's turned into a np array, it's able to be plugged into the model to
    # train off of

    train_labels = create_training_set_labels(50)

    X_train, X_test, y_train, y_test = train_test_split(train_dataset, train_labels, test_size=0.2, random_state=2)

    # the actual model is trained here. It's saved in a variable in case we wanted to print the history of it.
    history = model_new.fit(X_train, y_train, epochs=5)

    """
    y_predict = model_new.predict(X_test)

    print(y_predict)
    print(compile_predictions(y_predict))
    print(y_test)

    target_names = ['Daisies', 'Dandelions']
    print(classification_report(y_test, compile_predictions(y_predict), target_names=target_names))
    """
    # model_new.save("demo_model_TESTINGV1_01.h5") # saves the trained model as a file. get rid of the comment to enable
    return model_new


def make_prediction(model, file_name='test_dai.jpg', is_custom = False):

    # Image opening / formatting
    img = Image.open(file_name)
    img = cv2.resize(np.array(img), (224, 224))  # resize image to match model's expected sizing
    img = img.reshape(1, 224, 224, 3)  # return the image with shaping that TF wants.
    data = preprocess_input(img)

    predictions = model.predict(img)
    output_neuron = np.argmax(predictions)

    if is_custom: # image prediction
        print()
        print(file_name)
        print(predictions)
        print("Evens (left) are daisies / Odds (right) are dandelions")
    else:
        print('Shape: {}'.format(predictions.shape))

        print('Most active neuron: {} ({:.2f}%)'.format(
            output_neuron,
            100 * predictions[0][output_neuron]
        ))

        for name, desc, score in decode_predictions(predictions)[0]:
            print('- {} ({:.2f}%%)'.format(desc, 100 * score))

def make_batch_prediction(model, pred_dataset, true_dataset):
    y_predict = model.predict(pred_dataset)
    y_predict = compile_predictions(y_predict)

    print(y_predict)

    #print(compile_predictions(y_predict))
    print(true_dataset)



    #labels = [i for n, i in enumerate(y_predict) if i not in y_predict[:n]]
    #print(labels)

    #labels = [str(x) for x in labels]
    #print(labels)

    #[labels.append(x) for x in y_predict if x not in labels]

    #target_names = ['Daisies', 'Dandelions', "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17"]
    print(classification_report(true_dataset, y_predict))

def create_training_set(total = 50, augmented = False):
    file_name = 'Flowers_Classification.v2-raw\\train\\mix'+chr(92)+"1.jpg"
    img = Image.open(file_name)
    img = cv2.resize(np.array(img), (224, 224))  # resize image to match model's expected sizing
    img = img.reshape(1, 224, 224, 3)  # return the image with shaping that TF wants.

    # asarray() class is used to convert PIL images into NumPy arrays
    numpydata_original = asarray(img)

    if augmented:
        flipped = tf.image.flip_left_right(img)
        numpydata_flip = asarray(flipped)
        numpydata_original = np.concatenate([numpydata_original, numpydata_flip], axis=0)

    #print(numpydata_original)

    for x in range(2, total+1):
        file_name = 'Flowers_Classification.v2-raw/train/mix/' + str(x) + ".jpg"
        # new image is loaded with the name of the file
        img = Image.open(file_name)
        img = cv2.resize(np.array(img), (224, 224))  # resize image to match model's expected sizing
        img = img.reshape(1, 224, 224, 3)  # return the image with shaping that TF wants.

        numpydata = asarray(img)

        numpydata_original = np.concatenate([numpydata_original, numpydata], axis=0)

        if augmented:
            flipped = tf.image.flip_left_right(img)
            #grayscale = tf.image.rgb_to_grayscale(img)

            numpydata_flip = asarray(flipped)
            numpydata_original = np.concatenate([numpydata_original, numpydata_flip], axis=0)

    #aug_ds = numpydata_original.map(
    #    lambda x, y: (data_augmentation(x, training=True), y))

    return numpydata_original

def create_training_set_labels(total = 50, augmented = False):
    vals = []
    custom = True

    for x in range(1, total+1):
        if custom:
            if((x+1) % 2 == 0):
                vals.append(985)
                if augmented:
                    vals.append(985)
            else:
                vals.append(999)
                if augmented:
                    vals.append(999)
        else:
            vals.append((x+1) % 2)
            if augmented:
                vals.append((x + 1) % 2)
    train_labels = {'type': vals}
    train_labels = np.asarray(vals)

    return train_labels

def compile_predictions(predictions):
    predictions_formatted = [];
    for x in range(0, len(predictions)):
        predictions_formatted.append(np.argmax(predictions[x]))
    return predictions_formatted

def fine_tune_model(model, train_ds, label_ds):
    model.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    epochs = 10
    model.fit(train_ds, label_ds, epochs=epochs)

    return model

# --------------------------------------------------------------------

model = create_base_model()
#model = create_model_scratch((224, 224, 3), 2, create_training_set())

file_path = 'Flowers_Classification.v2-raw/train/mix/'
pic_name = '2'
is_custom = False

make_prediction(model, file_path+pic_name+".jpg", is_custom)
make_prediction(model, file_path+"test_dai"+".jpg", is_custom)
make_prediction(model, file_path+"test_dand"+".jpg", is_custom)

photo_range = 250

X_train, X_test, y_train, y_test = train_test_split(create_training_set(photo_range, True), create_training_set_labels(photo_range, True), test_size=0.2, random_state=2)

make_batch_prediction(model, X_test, y_test)

model = fine_tune_model(model, X_train, y_train)

make_prediction(model, file_path+pic_name+".jpg", is_custom)
make_prediction(model, file_path+"test_dai"+".jpg", is_custom)
make_prediction(model, file_path+"test_dand"+".jpg", is_custom)

make_batch_prediction(model, X_test, y_test)
