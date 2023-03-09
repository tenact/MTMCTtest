import numpy as np
import torch
from torchvision import models, transforms
import json
import torch.nn as nn
from PIL import Image
import cv2





class REID:

    # lesen der Text-File mit den JSON-Daten
    #oder des Hierarchical Clusterings

    # die Daten müssen schon processed sein, und 


    # Load the ResNet50 model
    # Load ResNet50 model
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval()

    # Define transform for preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load ResNet50 model
    
    def extract_features(self, img):
            
           # Load image using cv2
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Define transforms
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        # Preprocess image
        img_pil = Image.fromarray(img)
        img_tensor = preprocess(img_pil)

        # Create ResNet50 model
        resnet50 = models.resnet50(pretrained=True)

        # Remove last layer (softmax) from model
        resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-1])

        # Extract features
        with torch.no_grad():
            features = resnet50(img_tensor.unsqueeze(0))

        # Return features
        return features.squeeze().numpy()
    
    def euclidian_distance(self, features1, features2):
        """
        Calculates the Euclidean distance between two feature vectors.

        Args:
            features1 (np.array): 1D array of features.
            features2 (np.array): 1D array of features.

        Returns:
            distance (float): Euclidean distance between the two feature vectors.
        """

        features1_norm = features1 / np.linalg.norm(features1)
        features2_norm = features2 / np.linalg.norm(features2)
        # Calculate Euclidean distance
        distance = np.linalg.norm(features1_norm - features2_norm)

        return distance





    #IN Predict.py

    #Inittialisieurng der ReID-File
    # Durchführung der ReID-Operationen und der Erstellung der globalen FIles.





