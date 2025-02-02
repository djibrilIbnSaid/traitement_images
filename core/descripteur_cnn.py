# Description: This file contains the class DescripteurCNN which is used to extract features from images using a pre-trained CNN model. return the flatten features extracted from the image.
import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
from keras.applications import VGG16, MobileNet
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers import Flatten
from PIL import Image

class DescripteurCNN:
    """
        Classe permettant d'extraire les caractéristiques d'une image à l'aide d'un modèle CNN pré-entrainé
    """
    
    def __init__(self, model=VGG16(weights='imagenet', include_top=False), shape=(448, 448)):
        """
            Constructeur de la classe DescripteurCNN
            Usage:
                DescripteurCNN()
            Arguments:
                None
            Returns:
                None
        """
        self.model_vgg16 = Model(inputs=model.input, outputs=model.output)
        self.model_mobilenet = MobileNet(weights='imagenet', include_top=False)
        # self.model = Model(inputs=self.model.input, outputs=self.model.get_layer('block5_pool').output)
        self.shape = shape
        
        
    def __str__(self):
        return "DescripteurCNN"
    
    def __repr__(self):
        return self.__str__()
    
    def extract_features(self, image, type='VGG16'):
        """
            Extrait les caractéristiques d'une image à l'aide d'un modèle CNN pré-entrainé
            Usage:
                DescripteurCNN.extract_features(image_path)
            Arguments:
                image_path: str - Chemin de l'image
            Returns:
                np.array - Caractéristiques extraites de l'image
        """
        img = np.expand_dims(image, axis=0)
        img = preprocess_input(img)
        features = self.model_vgg16.predict(img) if type == 'VGG16' else self.model_mobilenet.predict(img)
        features = features.flatten()
        
        return features
    