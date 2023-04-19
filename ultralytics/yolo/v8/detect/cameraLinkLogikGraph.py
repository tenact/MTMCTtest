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
from sklearn.metrics.pairwise import euclidean_distances
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

def euclidean_distance(feature_array1, feature_array2):
    """
    Calculate the Euclidean distance between two ResNet50 feature arrays.
    
    Args:
        feature_array1 (np.ndarray): The ResNet50 feature array for the first image.
        feature_array2 (np.ndarray): The ResNet50 feature array for the second image.
        
    Returns:
        float: The Euclidean distance between the two feature arrays.
    """
    # Convert the feature arrays to NumPy arrays if they are not already
    feature_array1 = np.asarray(feature_array1)
    feature_array2 = np.asarray(feature_array2)
    
    # Calculate the Euclidean distance using the euclidean_distances function from sklearn
    distance = euclidean_distances(feature_array1.reshape(1, -1), feature_array2.reshape(1, -1))
    
    # The result is a 2D array with one value, so we extract and return the single value
    return distance[0, 0]



'''
TODO: Der Graph bleibt leer, da nie detection inside der Zonen sind.




'''

'''
def create_or_update_graph(new_feature_array, camera_name, id, timestap,  existing_graph, threshold=0.8):


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





    # Get current timestamp
    timestamp = time.time()

    print("die länge des Graphen beim Hinzufügen ist: " + str(len(existing_graph)))

    # Check if graph is empty, add a new node and continue to the next iteration
    if len(existing_graph) == 0:
        existing_graph.add_node(0, feature=new_feature_array, timestamp=timestamp, camera=camera_name, id=id)
        return existing_graph

    # Calculate Euclidean distance between the new feature and existing features in the graph
    distances = np.linalg.norm(new_feature_array - np.array([existing_graph.nodes[node]['feature'] for node in existing_graph]), axis=1)

    # Find nodes in the graph that are within the threshold distance
    close_nodes = np.where(distances <= threshold)[0]
    

   
   # TODO: 1. Filtern der Nodes/Tracks, die von der gleichen Kamera stammen.
    #TODO: 2. Filtern der Nodes, die einen Zeitunterschied von weniger als x- Sekungen haben (5)
   


    #DIE neue Node bekommt Edges zu den Nodes die in der Nähe sind
    if len(close_nodes) > 0:
        # If there are close nodes, add an edge between the new node and the closest node
        closest_node = close_nodes[np.argmin(distances[close_nodes])]
        existing_graph.add_node(closest_node+1, feature=new_feature_array, timestamp=timestamp, camera=camera_name, id=id)
        existing_graph.add_edge(closest_node, closest_node+1)
        # die edge soll die Distance darstellen
    else:
        # If there are no close nodes, add a new node in the graph
        existing_graph.add_node(len(existing_graph), feature=new_feature_array, timestamp=timestamp, camera=camera_name, id=id)


    print("die Id von MaxNodes is" + str(max(close_nodes)["id"]))
    print("die gebenen Id is" + str(id))
    maxNode = max(close_nodes)
#    return int(existing_graph.nodes[maxNode]['id'])
    
    id = int(existing_graph.nodes[maxNode]['id'])
    return int(id['id'])
    # rückgabe des Graphen und der Node mit der höchsten Euclidean Distanze zum Abfrage Element (wenn es existier, sonst None)
    #momemntan nur Aus  gabe der max Distanz
    # noch nach Heursitiken prüfen, ob Kamera Assoziation überhaupt möglich etc.

'''

def add_to_graph(new_feature_array, camera_name, id, timestamp, existing_graph, threshold=0.8):

    print("The length of the graph when adding is: " + str(len(existing_graph)))

    # Check if graph is empty, add a new node and continue to the next iteration
    if len(existing_graph) == 0:
        existing_graph.add_node(0, feature=new_feature_array, timestamp=timestamp, camera=camera_name, id=id)
        return existing_graph

    # Calculate Euclidean distance between the new feature and existing features in the graph
    distances = np.linalg.norm(new_feature_array - np.array([existing_graph.nodes[node]['feature'] for node in existing_graph]), axis=1)

    # Find nodes in the graph that are within the threshold distance
    close_nodes = np.where(distances <= threshold)[0]


    print("die länge von close nodes ist: " + str(len(close_nodes)))
    '''
    TODO: 1. Filter nodes/tracks that come from the same camera.
    TODO: 2. Filter nodes that have a time difference of less than x seconds (5).
    '''

    # The new node gets edges to the nodes that are close
    if len(close_nodes) > 0:
        # If there are close nodes, add an edge between the new node and the closest node
        closest_node = close_nodes[np.argmin(distances[close_nodes])]
        existing_graph.add_node(closest_node+1, feature=new_feature_array, timestamp=timestamp, camera=camera_name, id=id)
        existing_graph.add_edge(closest_node, closest_node+1)
        # The edge represents the distance
    else:
        # If there are no close nodes, add a new node in the graph
        existing_graph.add_node(len(existing_graph), feature=new_feature_array, timestamp=timestamp, camera=camera_name, id=id)





def get_max_element2(feature_arrays, graph):
    """
    Method to get the ID of the node with the highest Euclidean similarity in the given graph.

    Args:
    feature_arrays (numpy array): Feature array for which the maximum similarity is to be found.
    graph (NetworkX Graph): Graph containing nodes with feature arrays.

    Returns:
    int: ID of the node with the highest Euclidean similarity.
    """
    max_id = None
    max_distance = 0

    for node in graph.nodes:
        distance = np.linalg.norm(feature_arrays - graph.nodes[node]['feature'])
        if distance > max_distance:
            max_distance = distance
            max_id = graph.nodes[node]['id']


    return int(max_id) if max_id >= 0.8 else None



# TODO: DIESE FUNKTION WURDE NICHT AUF IHRE FUNKTIONALITÄT GETESTET
def get_max_element(feature_arrays, graph, timestamp_dict):
    """
    Method to get the ID of the node with the highest Euclidean similarity in the given graph.
    Additionally, checks if the camera name attribute is the same and if the timestamp
    between two nodes from different cameras is greater than 5 seconds.

    Args:
    feature_arrays (numpy array): Feature array for which the maximum similarity is to be found.
    graph (NetworkX Graph): Graph containing nodes with feature arrays.
    timestamp_dict (dict): Dictionary containing timestamp information for each node.

    Returns:
    int: ID of the node with the highest Euclidean similarity, considering camera name attribute
    and timestamp differences.
    """
    max_id = None
    max_distance = 0

    for node in graph.nodes:
        camera_name = graph.nodes[node]['camera_name']
        timestamp = timestamp_dict[node]
        distance = np.linalg.norm(feature_arrays - graph.nodes[node]['feature'])
        if distance > max_distance and distance >= 0.8:
            if all(graph.nodes[neighbor]['camera_name'] != camera_name for neighbor in graph.neighbors(node)):
                min_timestamp_diff = float('inf')
                for neighbor in graph.neighbors(node):
                    neighbor_camera_name = graph.nodes[neighbor]['camera_name']
                    neighbor_timestamp = timestamp_dict[neighbor]
                    if neighbor_camera_name != camera_name and abs(neighbor_timestamp - timestamp) < 5:
                        break
                    else:
                        min_timestamp_diff = min(min_timestamp_diff, abs(neighbor_timestamp - timestamp))
                else:
                    max_distance = distance
                    max_id = graph.nodes[node]['id']

    return int(max_id) if max_id is not None else None


class Neudas():
    data_deque = None
    deepsort = None
    features_dict = {}


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

    '''
    TODO: Erklräung der Funktion hier
    '''
    def draw_boxes(self, img, bbox, names,object_id, timestamp, dict, identities=None,  offset=(0, 0) ):

        self.lock = mp.Lock()
        with self.lock:

            #Entry und Exit-Plain 
            
            p1,p2,q1,q2 = dict["entryPlain"] 
            v1,v2,f1,f2 = dict["exitPlain"] 
            


            # ganz simpler Weg per internal queue: Alle Detectoin in der Entry werden dort hinzugefügt (aus der ExitQueue gelesen)
            #: Alle Detections in der Exit werden aus der internalQueue gelesen und der ExitQueue hinzugefügt

            internalQueue = deque()
            entryQueue = dict["entryQueue"]
            exitQueue = dict["exitQueue"]
            camName = dict["camName"]
            node_graph = dict["graph"]
            
            print(node_graph)
            #print("Der GRaph ist lang: " + str(len(node_graph.nodes)))
            #print("Adress des Graphs: " + str(id(node_graph)))


#


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
                    '''
                    Feature-Extraktion (solo ResNet50 Backbone: TODO: pre-trained-Model nutzen)
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
                    feature_bild = reid.extract_features(sub_image, id)
                    
                    currentObject = None
                    if x1 >= p1 and x2 <= p2 and y1 >= q1 and y2 <= q2: # Entry Plain
                        print(" Detection des Tracks befindet sich im Entry Plain")

                        '''
                        Erstellung des Grapehn
                        und Grapehn-Operationen


                        TODO: Informationen über das Tracking im Log speichern
                        TODO: 
                        TODO: Prüfen, ob es sich um ein Objekt handelt, dass gelöscht werden muss.
                            momentan einfache Lineare Version: Komplexer auch in Entry-Zone ist das Verlassen des FOVs möglich
                            und der Track müsste gelöscht werden

                        TODO: Wir löschen Track aus ExitZone und danach wird der Track nochmal getrackt im Exit-Zone?
                            Time-Constraint nutzen, dass das nicht möglich ist
                            Sollte mit einer guten Größe der Planens nicht passieren.
                        
                        
                        '''


                        #IMPORTANT: Logik für EntryPlane
                        if len(entryQueue) > 0:

                            internalQueue.append((feature_bild, id))

                            #prüfen, ob Objekt im Graph vorhanden ist
                            maxNode = get_max_element(feature_bild, node_graph)
                            if maxNode is not None:
                                print("Objekt ist im Graph vorhanden")
                                feature = maxNode["feature"]
                                if euclidean_distance(feature, feature_bild) > 0.8:
                                    entryQueue.pop()
                                    addIdToJsonFile = False
                                    id = maxNode["id"]
                                    add_to_graph(feature_bild, camName, id, timestamp, node_graph)
                            else:
                                for i in range(len(entryQueue)):
                                    if euclidean_distance(i , feature_bild) > 0.8 or  euclidean_distance(i, feature) > 0.8:
                                        entryQueue.pop(i)
                                        addIdToJsonFile = False
                                        id = maxNode["id"]
                                        add_to_graph(feature_bild, camName, id, timestamp, node_graph)

                                        break
                                    
                            #else Objekte nicht im Grah und nicht in der EntryQueue => im LOG Vermerken => es gab irgendeinen Fehler
                            #Fehler, dass die entryQueue leer ist kann nicht sein, da dies nur ausgeführt wird, wenn die entryQueue größer als 0 ist

                        else: # Keine detections aus der vorherigen Kamera, also werden objekte einfach der internalQueue hnizugefügt
                            internalQueue.append(currentObject)
                            print("erstes Objekt der Queue")
                            add_to_graph(feature_bild, camName, id, timestamp, node_graph)



                    elif x1 >= v1 and x2 <= v2 and y1 >= f1 and y2 <= f2:  # Exit Plain
                        
                        if exitQueue is not None:
                            #TODO: Detection in ExitQueue, die vorher nicht detected wurde: => Fehler

                            maxNode = add_to_graph(feature_bild, camName, id, timestamp, node_graph)
                            if maxNode is not None:
                                internalQueue.pop() #Aus InternalQueue Bild lesen und ExitQueue hinzufügen    
                                exitQueue.pop() # keine Prüfung, ob es auch das richtige ist
                                # => aber Annahme, dass es keine Überholungen in der Produktion gibt, 
                                # Deshabl wurde dieser Case hier nicht berücksichtigt
                                addIdToJsonFile = False
                                id = maxNode["id"]
                        else:
                            maxNode = add_to_graph(feature_bild, camName, id, timestamp, node_graph)
                            # es sollte ja kein Track nochmals detected werden in dn Zonen
                                #=> ja aber es weredn alle TRACKS gezeichnet und wenn DeepSort
                            #mir sagt, zeichne dort, dann mache ich das.
                            internalQueue.pop()

                            addIdToJsonFile = False
                            #TODO: Im lOG VERMERKEN
                            
                    print("Die Länge der internalQueue ist: " + str(len(internalQueue)))
                    data_deque[id] = deque(maxlen= 256)


                '''
                Fuck, dass ist ja nur für Tracks und nicht Detections: Für Tracks einfach nur die ReId machen#
                und bei den Detections immer aud die Graph-Struktur zugreifen, ob es sich um existierende Tracks handelt
                die auch in den InternalQueues vorhanden sind (DeepSort Level Single-Camera-Tracking: Falsher Ort.
                TODO: nein
                hier wird ja auch gezeichnet, und bei jeder Zeichnung prüfe ich, ob es sich im exit/entry befindet und dann entferne ich das

                    ich zeichne also was updateSort mir gibt. Also die aktuellen Tracks Idenitfikationen
                    also ist das doch alles richtig.
                    keine Änderung nötig.
                )
                '''


                print(type(id))
                abs = id +1
                print(type(abs))
                acd = abs +1
                cfd = acd-2
                id = cfd
                print(type(id))

                

                    
                color = (255,255,255) #white/ 
                
                if(addIdToJsonFile):
                    color = self.compute_color_for_labels(object_id[i]) 
                    
                obj_name = names[object_id[i]]
                label = '{}{:d}'.format("", id) + ":"+ '%s' % (obj_name)
                #label = '{}{:d}'.format("", id) + ":" + obj_name
                #label = f'{id}: {obj_name}'



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
        pass        
        return img