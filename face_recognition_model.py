import tensorflow as tf
from retinaface.model import retinaface_model
from retinaface.RetinaFace import build_model

class FaceRecognitionModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(FaceRecognitionModel, self).__init__()

        self.num_classes = num_classes

        self.retinaface_model = build_model()
        self.concatenated_retinaface_output = self.build_concatenate_retinaface_output(self.retinaface_model)
        
        self.recognition_model = self.recognition_model_from_retina(self.concatenated_retinaface_output)
        
        self.model = tf.keras.Model(
            inputs=self.retinaface_model.inputs, 
            outputs=self.recognition_model.outputs
        )


    def build_concatenate_retinaface_output(self, retinaface_model):
        feature_maps = retinaface_model.outputs
        concatenated = tf.keras.layers.Concatenate(axis=-1)(feature_maps)
        return concatenated


    def recognition_model_from_retina(self, retinaface_features):
        flatten_layer = tf.keras.layers.Flatten()(retinaface_features)
        dense1 = tf.keras.layers.Dense(128, activation='relu')(flatten_layer)
        output = tf.keras.layers.Dense(self.num_classes, activation='softmax')(dense1)
        return output


    def call(self, inputs):
        retinaface_features = self.retinaface_model(inputs)
        recognition_output = self.recognition_model(retinaface_features)
        return recognition_output

'''
def call(self, inputs):
    retinaface_features = self.retinaface_model(inputs)

    flatten_layer = tf.keras.layers.Flatten()(retinaface_features)
    dense1 = tf.keras.layers.Dense(128, activation='relu')(flatten_layer)
    recognition_output = tf.keras.layers.Dense(self.num_classes, activation='softmax')(dense1)

    return recognition_output
'''
