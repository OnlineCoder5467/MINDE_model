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


def create_base_model(input_shape_arr=(224, 224, 3)):
    mbn = tf.keras.applications.MobileNetV2(weights='imagenet', input_shape=input_shape_arr)
    return mbn

def create_model_scratch(input_shape_arr=(224, 224, 3), num_classes = 2, train_dataset = []):
    # Evens are dandelions
    # Odds are daisies

    # """ Comment out for creating the transfer learning model from scratch

    mbn = tf.keras.applications.MobileNetV2(weights='imagenet', input_shape=input_shape_arr, include_top=False)
    #mbn = tf.keras.applications.MobileNetV2(weights='imagenet', input_shape=input_shape_arr)

    # make a reference to MobileNet's input layer
    inp = mbn.input
    #inp = Flatten()(inp)

    x = Flatten()(mbn.output)  # our layers - you can add more if you want

    # make a new softmax layer with num_classes neurons
    new_classification_layer = Dense(num_classes, activation='softmax')(x)

    # connect our new layer to the second to last layer, and make a reference to it
    #out = new_classification_layer(mbn.layers[-2].output)

    # create a new network between inp and out
    model_new = Model(inp, new_classification_layer)

    # make all layers untrainable by freezing weights (except for last layer)
    for l, layer in enumerate(model_new.layers[:-1]):
        layer.trainable = False

    # ensure the last layer is trainable/not frozen
    for l, layer in enumerate(model_new.layers[-1:]):
        layer.trainable = True


    #model_new.summary()

    model_new.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 0 = daisy
    # 1 = dandelion
    #train_labels = {'file_name': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
    #                 'type':[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]}

    """
    train_ds, test_ds = tfds.load('cifar10', split=['train', 'test'], as_supervised=True, batch_size=-1)
    train_images, train_labels = tfds.as_numpy(train_ds)
    X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.1, random_state=2)

    #print(X_train.shape)
    print(X_test.shape)
    print(train_dataset.shape)
    #print(y_test.shape)
    #print(type(y_test))
    #print(y_test)
    """

    vals = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    train_labels = {'type': vals}

    train_labels = np.asarray(vals)

    """
    #train_labels = pd.DataFrame(asarray(train_labels))
    #train_labels = pd.DataFrame(train_labels)
    #print(type(train_labels))
    

    #print(train_labels)
    #print(train_labels.shape)
    #print(type(train_labels))

    #train_dataset = tf.cast(train_dataset, dtype=tf.float32)
    #train_labels = tf.cast(train_labels, dtype=tf.float32)

    
    # model_new.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # model_new.summary()

    # train_ds, test_ds = tfds.load('cifar10', split=['train','test'], as_supervised = True, batch_size = -1)
    # train_images, train_labels = tfds.as_numpy(train_ds)


    # X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.2, random_state=2)

    #df = pd.read_csv("oil-spill.csv")
    #print(df)

    # print(train_images.shape)
    # print(train_images[0])
    # print(train_images[0].shape)
    # print(train_labels)
    """

    history = model_new.fit(train_dataset, train_labels, epochs=5)

    # model_new.save("demo_model_TESTINGV1_01.h5")

    return model_new
    #"""

#"""

def make_prediction(model, file_name='test_dai.jpg'):

    # Image opening / formatting
    img = Image.open(file_name)
    img = cv2.resize(np.array(img), (224, 224))  # resize image to match model's expected sizing
    img = img.reshape(1, 224, 224, 3)  # return the image with shaping that TF wants.
    data = preprocess_input(img)

    # image prediction
    predictions = model.predict(img)
    output_neuron = np.argmax(predictions)

    print()
    print(file_name)
    print(predictions)

    """
    # print out results
    print('Shape: {}'.format(predictions.shape))

    print('Most active neuron: {} ({:.2f}%)'.format(
        output_neuron,
        100 * predictions[0][output_neuron]
    ))

    for name, desc, score in decode_predictions(predictions)[0]:
        print('- {} ({:.2f}%%)'.format(desc, 100 * score))
    """


def create_training_set():
    file_name = 'Flowers_Classification.v2-raw\\train\\mix'+chr(92)+"1.jpg"
    img = Image.open(file_name)
    img = cv2.resize(np.array(img), (224, 224))  # resize image to match model's expected sizing
    img = img.reshape(1, 224, 224, 3)  # return the image with shaping that TF wants.

    #img = Image.open('Digit/0.jpg')

    # asarray() class is used to convert PIL images into NumPy arrays
    numpydata_original = asarray(img)

    """
    print("--------------------------")
    print(len(numpydata))
    print(numpydata.shape)
    print(numpydata)

    file_name2 = 'Flowers_Classification.v2-raw\\train\\mix' + chr(92) + "2.jpg"
    img2 = Image.open(file_name2)
    img2 = cv2.resize(np.array(img2), (224, 224))  # resize image to match model's expected sizing
    img2 = img.reshape(1, 224, 224, 3)  # return the image with shaping that TF wants.

    numpydata2 = asarray(img2)

    merge_arr = np.concatenate([numpydata, numpydata2], axis=0)
    #numpydata.append(numpydata2)

    print("--------------------------")
    print(len(merge_arr))
    print(merge_arr.shape)
    print(merge_arr)
    """


    # the new dataframe is created using the values of the first image in the dataset as the base
    #converted_df = pd.DataFrame([flat_list])
    total = 31
    for x in range(2, total):
        #print(x)
        file_name = 'Flowers_Classification.v2-raw/train/mix/' + str(x) + ".jpg"
        # new image is loaded with the name of the file
        img = Image.open(file_name)
        img = cv2.resize(np.array(img), (224, 224))  # resize image to match model's expected sizing
        img = img.reshape(1, 224, 224, 3)  # return the image with shaping that TF wants.

        numpydata = asarray(img)

        numpydata_original = np.concatenate([numpydata_original, numpydata], axis=0)

    """
    for x in range(2, total):
        #print(x)
        file_name = 'Flowers_Classification.v2-raw/train/mix/' + str(x) + ".jpg"
        # new image is loaded with the name of the file
        img = Image.open(file_name)
        img = cv2.resize(np.array(img), (224, 224))  # resize image to match model's expected sizing
        img = img.reshape(1, 224, 224, 3)  # return the image with shaping that TF wants.

        numpydata = asarray(img)

        flat_list = [item for sublist in numpydata for item in sublist]

        # the new 64x1 array is converted into a dataframe and then concatonated onto the base dataframe made earlier
        temp_df = pd.DataFrame([flat_list])
        converted_df = pd.concat([converted_df, temp_df])
    """
    #print(converted_df)

    #print(len(converted_df))
    return numpydata_original

# --------------------------------------------------------------------

#model = create_base_model()
model = create_model_scratch((224, 224, 3), 2, create_training_set())

file_path = 'Flowers_Classification.v2-raw\\train\\mix'+chr(92)
pic_name = '2'

make_prediction(model, file_path+pic_name+".jpg")
make_prediction(model, file_path+"test_dai"+".jpg")
make_prediction(model, file_path+"test_dand"+".jpg")

print("Evens (left) are dandelions / Odds (right) are daisies")