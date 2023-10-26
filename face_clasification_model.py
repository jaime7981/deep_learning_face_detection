import tensorflow as tf

class FaceClasificatorModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(FaceClasificatorModel, self).__init()
        self.retinaface_model = build_model()  # Assuming you have a RetinaFace model
        self.recognition_model = self.build_recognition_model(num_classes)

    def build_recognition_model(self, num_classes):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        return model

    def call(self, inputs):
        # Extract features from RetinaFace
        retinaface_features = self.retinaface_model(inputs)

        # Concatenate the RetinaFace features if needed
        # You can skip this if you've already concatenated the features

        # Pass features through the recognition model
        recognition_output = self.recognition_model(retinaface_features)

        return recognition_output
