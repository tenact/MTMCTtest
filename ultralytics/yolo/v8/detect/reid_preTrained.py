import numpy as np
import torch
from torchvision import models, transforms
import json
import torch.nn as nn
from PIL import Image
import cv2

from torchreid.utils import FeatureExtractor

import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity


extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='a/b/c/model.pth.tar',
    )



class REID:


    

    
    

    
    
    def extract_features(self, img, id):
            
        return extractor(img)
    
    def euclidian_distance(self, features1, features2):
        """
        Calculates the Euclidean distance between two feature vectors.

        Args:
            features1 (np.array): 1D array of features.
            features2 (np.array): 1D array of features.

        Returns:
            distance (float): Euclidean distance between the two feature vectors.
        """

        #features1_norm = features1 / np.linalg.norm(features1)
        #features2_norm = features2 / np.linalg.norm(features2)
        # Calculate Euclidean distance
        #distance = np.linalg.norm(features1_norm - features2_norm)

        #return distance
        similarity = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))
        return similarity


    

    #IN Predict.py

    #Inittialisieurng der ReID-File
    # Durchführung der ReID-Operationen und der Erstellung der globalen FIles.





