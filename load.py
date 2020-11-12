import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense


def create_model(num_classes, dropout_rate):
    base = tf.keras.applications.InceptionV3(include_top=False, weights=None, input_shape=(256, 256, 3))
    x = base.get_layer("mixed10").output
    x = AveragePooling2D(pool_size=(6, 6), padding="valid")(x)
    x = Dropout(dropout_rate)(x)
    x = Flatten()(x)
    x = Dense(units=num_classes, activation="softmax")(x)

    model = tf.keras.Model(base.input, x)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model


# Loading class labels
classes = []
for dir in sorted(os.listdir("data/training_data")):
    classes.append(dir)

# Loading model
model = create_model(num_classes=196, dropout_rate=0.2)
model.load_weights("train/cp.ckpt")

# Writing predictions
with open("predictions.csv", "w") as file:
    writer = csv.writer(file, lineterminator="\n")
    writer.writerow(["id", "label"])

    i = 0
    for filename in os.listdir("data/testing_data"):
        image_path = "data/testing_data/" + filename
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(256, 256), interpolation='bilinear')
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # convert single image to a batch

        prediction = model.predict(input_arr)[0]
        idx = prediction.argmax()
        label = classes[idx]

        writer.writerow([filename[:-4], label])

        i += 1
        if i % 100 == 0:
            print(i)
