from dataset.dataset import get_images_path_and_name, get_labels
from retinaface.RetinaFace import build_model, extract_faces, detect_faces, get_image
from retinaface.commons import preprocess, postprocess

from face_recognition_model import FaceRecognitionModel

import tensorflow as tf
import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import os

# get sccript path
BASE_PATH = os.path.dirname(os.path.realpath(__file__))


def load_images_and_labels(dataframe, retina_model, common_size=(224, 224)):
    images = []
    labels = []

    for index, row in dataframe.iterrows():
        image_path = row['path']
        label = row['name']

        # Extract faces using RetinaFace
        try:
            extracted_face = extract_faces(img_path=image_path, model=retina_model)
        except:
            path_without_extension = os.path.splitext(image_path)[0]
            new_path = path_without_extension + ".jpeg"
            new_path = os.path.join(BASE_PATH, new_path)
            image_path = get_image(new_path)
            extracted_face = extract_faces(img_path=image_path, model=retina_model)
        
        for face in extracted_face:
            rezied_face = cv2.resize(face, common_size)
            images.append(rezied_face)
            labels.append(label)

    return np.array(images), np.array(labels)

def main():
    train_img_df = get_images_path_and_name("./dataset/images/")

    test_img_path = os.path.join(BASE_PATH, "dataset/Test/images/")
    test_img_df = get_images_path_and_name(test_img_path)
    labels = get_labels(train_img_df)

    print(train_img_df)
    print(labels)

    retina_model = build_model()
    retina_detect_faces = detect_faces(img_path = train_img_df["path"][0], model = retina_model)

    print(retina_detect_faces)


    test_images, test_labels = load_images_and_labels(test_img_df, retina_model)
    train_images, train_labels = load_images_and_labels(train_img_df, retina_model)
    


    face_recognition_model = FaceRecognitionModel(num_classes = len(labels))
    face_recognition_model.build((None, None, None, 3))
    face_recognition_model.summary()

    face_recognition_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    face_recognition_model.fit(
        train_images,
        train_labels,
        epochs=5,
        batch_size=32,
        validation_split=0.1
    )

    face_recognition_model.evaluate(
        test_images,
        test_labels
    )

    face_recognition_model.save_weights('./face_recognition_model_weights.h5')



if __name__ == '__main__':
    main()
