import tensorflow as tf
from retinaface.model import retinaface_model
from retinaface.RetinaFace import build_model

class FaceRecognitionModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(FaceRecognitionModel, self).__init__()
        self.retinaface_model = build_model()
        self.num_classes = num_classes
        self.recognition_model = self.build_recognition_model(num_classes)

    def build_recognition_model(self, num_classes):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        return model

    def call(self, inputs):
        # Detect faces using RetinaFace
        face_detections = self.retinaface_model(inputs)
        
        # Initialize an empty list to store face labels
        face_labels = []

        for feature_map in face_detections:
            # Reshape feature maps to (batch_size, num_features)
            feature_map = tf.reshape(feature_map, (feature_map.shape[0], -1))

            # Pass the reshaped feature map through the recognition model
            face_label = self.recognition_model(feature_map)
            face_labels.append(face_label)

        return face_labels
