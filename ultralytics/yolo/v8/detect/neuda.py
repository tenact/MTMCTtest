# Ultralytics YOLO 🚀, GPL-3.0 license

import multiprocessing as mp
import os
import queue

from sklearn.cluster import AgglomerativeClustering
import hydra
import torch
import argparse
import time
from pathlib import Path
from PIL import Image
from reid import REID
import json
import networkx as nx
import time


from collections import deque

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

import cv2
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np
from ultralytics.yolo.v8.detect.hardFeatures import SIFTFeatures

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)




'''
Graph muss beim Aufruf des MultiProcessing definiert werden, Also ist die Abfrage hier unnötig, da immer ein Graph
übergeben wird.


'''

def create_or_update_graph(new_feature_array, camera_name, existing_graph=None, threshold=0.8):


    # wenn kein Graph übergeben kriegen wir einen zurück
    """
    Method to create or update a graph based on a new feature array.

    Args:
    new_feature_array (numpy array): New feature array to be added to the graph.
    camera_name (str): Name of the camera associated with the new feature array.
    existing_graph (NetworkX Graph): Existing graph to be updated. If None, a new graph will be created.
    threshold (float): Threshold for edge creation based on Euclidean distance. Default is 0.8.

    Returns:
    NetworkX Graph: Updated graph after adding the new feature array.
"""
    '''
    # If existing graph is not provided, create a new graph
    if existing_graph is None:
        graph = nx.Graph()
    else:
        graph = existing_graph

    '''

    # Get current timestamp
    timestamp = time.time()

    # Check if graph is empty, add a new node and continue to the next iteration
    if len(existing_graph) == 0:
        existing_graph.add_node(0, feature=new_feature_array, timestamp=timestamp, camera=camera_name)
        return existing_graph

    # Calculate Euclidean distance between the new feature and existing features in the graph
    distances = np.linalg.norm(new_feature_array - np.array([existing_graph.nodes[node]['feature'] for node in graph]), axis=1)

    # Find nodes in the graph that are within the threshold distance
    close_nodes = np.where(distances <= threshold)[0]



    # wir müssen noch die node mit der höchsten Distanz filtern, also die liste ordnen und das höchste ausgeben





    if len(close_nodes) > 0:
        # If there are close nodes, add an edge between the new node and the closest node
        closest_node = close_nodes[np.argmin(distances[close_nodes])]
        existing_graph.add_node(closest_node+1, feature=new_feature_array, timestamp=timestamp, camera=camera_name)
        existing_graph.add_edge(closest_node, closest_node+1)
        # die edge soll die Distance darstellen
    else:
        # If there are no close nodes, add a new node in the graph
        existing_graph.add_node(len(existing_graph), feature=new_feature_array, timestamp=timestamp, camera=camera_name)

    return max(close_nodes)
    # rückgabe des Graphen und der Node mit der höchsten Euclidean Distanze zum Abfrage Element (wenn es existier, sonst None)
    #momemntan nur Ausgabe der max Distanz
    # noch nach Heursitiken prüfen, ob Kamera Assoziation überhaupt möglich etc.



class Neudas():
    data_deque = None
    deepsort = None
    features_dict = {}


    # ich Initalisiere die Deepsort Klasse
    # ich übergebe deepSort
    # dann muss die drawBoxes Funktion ausgeführt werden 

    # data_deque = {} wird beim Init initalisiert
    #data_deque = {} auch ausgeben, und in drawBoxes übergeben?
    # ich müsste die dann aber auch wieder zurückgeben



    def init_tracker(self):
        global deepsort
        
        global data_deque
        global features_dict

        data_deque = {}
        features_dict = {}

        cfg_deep = get_config()
        cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

        deepsort= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                                max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                                nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                                max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                                use_cuda=True)
        #return deepsort
    ##########################################################################################

    def getDeepSort(self):
        return deepsort
    def getData_deque(self):
        return data_deque

    def compute_color_for_labels(self, label):
        """
        Simple function that adds fixed color depending on the class
        """
        if label == 0: #person
            color = (85,45,255)
        elif label == 2: # Car
            color = (222,82,175)
        elif label == 3:  # Motobike
            color = (0, 204, 255)
        elif label == 5:  # Bus
            color = (0, 149, 255)
        else:
            color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
        return tuple(color)

    

    def draw_border(self, img, pt1, pt2, color, thickness, r, d):
        x1,y1 = pt1
        x2,y2 = pt2
        # Top left
        cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        # Top right
        cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        # Bottom left
        cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
        cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        # Bottom right
        cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
        cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
        
        cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
        cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
        cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
        cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
        
        return img

    def UI_box(self,x, img, color=None, label=None, line_thickness=None): #added self
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

            img =self.draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2) #added self

            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


    




    #Logik zur Erstellung des Graphen



    



    # draw boxes, and takes the id of the object

    def draw_boxes(self, img, bbox, names,object_id, dict, identities=None,  offset=(0, 0) ):

        self.lock = mp.Lock()
        with self.lock:

            #Entry und Exit-Plain 
            
            p1,p2,q1,q2 = dict["entryPlain"] 
            v1,v2,f1,f2 = dict["exitPlain"] 
            
            '''  
            '''


            # ganz simpler Weg per internal queue: Alle Detectoin in der Entry werden dort hinzugefügt (aus der ExitQueue gelesen)
            #: Alle Detections in der Exit werden aus der internalQueue gelesen und der ExitQueue hinzugefügt

            internalQueue = deque()
            entryQueue = dict["entryQueue"]
            exitQueue = dict["exitQueue"]
            camName = dict["camName"]
            graph = dict["graph"]


            # übergebne Liste wo kameraübergreifend die Identitäten + BILD (hier noch ResNet50 Array)gespeichert werden
            features_dict = dict["list"]
            
            height, width, _ = img.shape  
            neuId = 0
            addIdToJsonFile = True 

            for key in list(data_deque):
                if key not in identities:
                    print("key not in identities deque" + str(key))
                    '''
                    if x1 >= v1 and x2 <= v2 and y1 >= f1 and y2 <= f2: 
                        print("key insides Exit-Zone and can be removed")

                    # DAS MACHT ÜBERHAUPT KEINEN SINN, DA ES JA KEINE DETECTOIN GIBT
                    # ALSO WÄRE DAS NIEMALS WAHR

                    #ICH MUSS DAS ENTWEDER ZULASSEN
                    #ODER EINE NEUE LÖSUNG FINDEN,
                    #WENN YOLO GUT GENUG IST, SOLLTE ES IMMMER DIE OBJEKTE IN DER EXIT-ZONE ENTDECKEN UND ES SOLLTE
                    #JEDE IDENTITÄ GEPOPT WERDEN.

                    #TODO:  ALSO DAS poppen hier entfernen und in der ExitZone hinzufügen.

                    # hier das mit der ExitQueue und InternalQueue machen
                    #nur dann ist es möglich, dass ein Objekt entfernt wird.
                        data_deque.pop(key)
                        exitQueue.append(key)
                    
                    '''
            for i, box in enumerate(bbox):

                x1, y1, x2, y2 = [int(i) for i in box]
                x1 += offset[0]
                x2 += offset[0]
                y1 += offset[1]
                y2 += offset[1]
                
                # code to find center of bottom edge
                center = (int((x2+x1)/ 2), int((y2+y2)/2))
            
                # get ID of object
                id = int(identities[i]) if identities is not None else 0
                # checks if its a new ID
                if id not in data_deque:  

                    # DEFINITION der Entry und Exit Points und erst dann neue Elemente hinzufügen
                    #check if its inside the entry or exit plain


                    '''
                    Feature-Extraktion (Hard-Crafted-Features: TODO: pre-trained-Model nutzen)
                    '''
                  
                    point1 = (x1, y1)
                    point2 = (x2, y2)

                    # Calculate the sub-image dimensions
                    sub_image_width = point2[0] - point1[0]
                    sub_image_height = point2[1] - point1[1]

                    # Extract the sub-image using array indexing
                    sub_image = np.zeros((sub_image_height, sub_image_width, 3), dtype=np.uint8)
                    for y in range(sub_image_height):
                        for x in range(sub_image_width):
                            sub_image[y, x] = img[point1[1]+y, point1[0]+x]
                    
                    reid = REID()   

                    #Durchführung der Re-Identification und Auslesen der Id
                    # oder hinzufügen der neuen ID mit Feature zum Dictionary

                    feature_bild = reid.extract_features(sub_image, id)





                    
                    currentObject = None
                    if x1 >= p1 and x2 <= p2 and y1 >= q1 and y2 <= q2: # Entry Plain
                        # befindet sich im Entry Plain
                        print("befindet sich im Entry Plain")
                        


                        '''
                        Erstellung des Grapehn
                        und Grapehn-Operationen
                        '''
                        # 1. Extracktion der Features
                        # 2. Aufruf der Graph Funktion und 
                        # 3. noch die Track ID in der NOde speichern, wenn die Euclidean Distance den Threshold überschreitet ID hinzufünge
                        #    und und hier in DrawBoxes verwenden / als ReID darstellen.
                        # ! ich speicher den gleichen Track mehrmals, aber das ist ja nicht schlimm? - bringt ja eine bessere Assoziation
                        # 
                        #  
                        
                        maxNode = create_or_update_graph(feature_bild, camName, existing_graph=graph)

                        if maxNode is not None:
                            # get id from node
                            print("maxNode is not None")



                        if len(entryQueue) > 0:
                            currentObject = entryQueue.pop() # nur das Bild ist gespeichert, wenn die nächste ID kommt, dann 
                            #kann ich das abfragen, aber 
                            #mehr Daten wären besser
                            # der Trail, um die ganzen Koordinaten zu haben
                            # und vielleciht ein zwei Bilder mehr


                            #Änderung der ganzen Speicherung


                            internalQueue.append(currentObject)
                            print("die Länge der entryQueue ist: " + str(len(entryQueue)))
                        else:
                            internalQueue.append(currentObject)

                    
                    elif x1 >= v1 and x2 <= v2 and y1 >= f1 and y2 <= f2:  # Exit Plain
                        
                        if len(exitQueue) > 0:
                            currentObject = internalQueue.pop() #Aus InternalQueue Bild lesen und ExitQueue hinzufügen
                            exitQueue.append(currentObject)
                            print("die Länge der entryQueue ist: " + str(len(exitQueue)))
                        else:
                            internalQueue.pop()
                            
                    print("Die Länge der internalQueue ist: " + str(len(internalQueue)))
                    data_deque[id] = deque(maxlen= 256)

                    point1 = (x1, y1)
                    point2 = (x2, y2)

                    # Calculate the sub-image dimensions
                    sub_image_width = point2[0] - point1[0]
                    sub_image_height = point2[1] - point1[1]

                    # Extract the sub-image using array indexing
                    sub_image = np.zeros((sub_image_height, sub_image_width, 3), dtype=np.uint8)
                    for y in range(sub_image_height):
                        for x in range(sub_image_width):
                            sub_image[y, x] = img[point1[1]+y, point1[0]+x]
                    
                    reid = REID()   

                    #Durchführung der Re-Identification und Auslesen der Id
                    # oder hinzufügen der neuen ID mit Feature zum Dictionary

                    #Erstellung des Camera-Link-Mode

                    feature_bild = reid.extract_features(sub_image, id)

                    #print("Länge der Featutes: " + str(len(features_dict)))

                    if len(features_dict) == 0: # Frame 58, Exception
                        features_dict[0] = feature_bild
                    else:
                        for fe, arr in features_dict.items():
                            dist = reid.euclidian_distance(arr, feature_bild)
                            print(dist)
                            if dist < 0.3:
                                neuId = fe
                                addIdToJsonFile = False
                                break
                        else:
                            features_dict[id] = feature_bild

                if neuId != 0:
                    id = neuId  
                    
                color = (255,255,255) #white/ 
                
                if(addIdToJsonFile):
                    color = self.compute_color_for_labels(object_id[i]) 
                    
                obj_name = names[object_id[i]]
                label = '{}{:d}'.format("", id) + ":"+ '%s' % (obj_name)

                # add center to buffer
                data_deque[id].appendleft(center) 

                self.UI_box(box, img, label=label, color=color, line_thickness=2)
                # draw trail
                for i in range(1, len(data_deque[id])):

                    # check if on buffer value is none
                    if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                        continue           
                    cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color) #, thickness

                    #image = img
                '''
                overlay = img.copy()
                cv2.rectangle(overlay, (p1, q1), (p1+(p2-p1), q1+(q2-q1)), (0, 0, 255, 128), -1)
                cv2.rectangle(overlay, (v1, f1), (v1+(v2-v1), f1+(f2-f1)), (0, 0, 255, 128), -1)

                alpha = 0.5
                img = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)

                print("x1: " + str(x1))
                print("x2: " + str(x2))
                print("y1: " + str(y1))
                print("y2: " + str(y2))
                '''



            color = (255, 0, 0)  # blue
            alpha = 0.5  # 50% transparency

            mask = np.zeros_like(img)
            cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
            # Apply the mask to the image
            img = cv2.addWeighted(mask, alpha, img, 1 - alpha, 0)


            #Exit Plain
            color = (0, 0, 255)  # red
            alpha = 0.5  # 50% transparency

            mask = np.zeros_like(img)
            cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
            # Apply the mask to the image
            img = cv2.addWeighted(mask, alpha, img, 1 - alpha, 0)
    #cv2.imshow('image', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
            
            return img