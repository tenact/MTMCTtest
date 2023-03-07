#imports



import numpy as np
import torch
from torchvision import models, transforms
import json



class REID:

    # lesen der Text-File mit den JSON-Daten
    #oder des Hierarchical Clusterings

    # die Daten müssen schon processed sein, und 


    # Load the ResNet50 model
    # Load ResNet50 model

    
    

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

    def extract_features(img_path):
        """
        Extracts ResNet50 features from an image file.

        Args:
            img_path (str): Path to image file.

        Returns:
            features (np.array): 1D array of ResNet50 features.
        """
        # Load image and preprocess
        img = transform(Image.open(img_path))
        # Add batch dimension
        img = img.unsqueeze(0)
        # Forward pass through model
        with torch.no_grad():
            features = model(img)
        # Convert to numpy array and flatten
        features = features.numpy().flatten()
        return features

    def euclidean_distance(features1, features2):
        """
        Calculates the Euclidean distance between two feature vectors.

        Args:
            features1 (np.array): 1D array of features.
            features2 (np.array): 1D array of features.

        Returns:
            distance (float): Euclidean distance between the two feature vectors.
        """
        # Calculate Euclidean distance
        distance = np.linalg.norm(features1 - features2)
        return distance





    #IN Predict.py

    #Inittialisieurng der ReID-File
    # Durchführung der ReID-Operationen und der Erstellung der globalen FIles.





