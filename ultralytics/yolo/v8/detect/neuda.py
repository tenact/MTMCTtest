# Ultralytics YOLO 🚀, GPL-3.0 license

import multiprocessing
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

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


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


    # draw boxes, and takes the id of the object

    def draw_boxes(self, img, bbox, names,object_id, identities=None,  offset=(0, 0) ):

        #Entry und Exit-Plain 
        '''
        p1,p2,q1,q2 = dict["entryPlain"] 
        v1,v2,f1,f2 = dict["ExitPlain"] 
        entryQueue = dict["entryQueue"]
        exitQueue = dict["exitQueue"]

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
        '''  
        internalQueue = []
        entryQueue = []
        exitQueue = []
        
        height, width, _ = img.shape  
        neuId = 0
        addIdToJsonFile = True 

        print("draw_boxes wird gezeichnet") 
        print("identities: " + str(identities))
        print("object_id: " + str(object_id))
        


        print("Die Länge der data: queue " + str(len(data_deque)))
        #print(hex(id)(data_deque))

        for key in list(data_deque):
            if key not in identities:
                print("key not in identities deque" + str(key))
                # aber ich habe die ganzen Infos nicht mehr
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

            print("wir sind in der Schleife")
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
                #übergabe von zwei Punkten

                #entryPoint = (p1, q1), (p2, q2) # andere Punkte, nicht die Bounding Box
                #exitPoint = (v1, f1), (v2, f2)


                #check if its inside the entry or exit plain
                '''
                if x1 >= p1 and x2 <= p2 and y1 >= q1 and y2 <= q2: 
                    # befindet sich im Entry Plain
                    print("befindet sich im Entry Plain")

                    internalQueue.append({'id': id, 'feature': data_deque[id], 'label': names[id], 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}) 


                    #if entryQueue is not None:
                        #   entryQueue.get() 
                        # momentan speicher ich nur das Feature Array und die ID
                        # aber wenn ich hier das Speichern will, dann muss ich das alles speichern
                            #die Koordinaten
                            # die ID
                            # Feature   Array
                            # einfach das gannze Label speicerhn
                            
                        

                    # hinzufügen einer Queue 
                    

                #if x1 >= v1 and x2 <= v2 and y1 >= f1 and y2 <= f2: 
                    # befindet sich im Exit Plain

                    #   if exitQueue is not None:
                        # aus der eigenen Queue rausnehmene und in die ExitQueue hinzufügen
                    #      exitQueue.append({'id': id, 'feature': data_deque[id], 'label': names[id], 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2})                                         
                    #hinzufügen der Queue für die nächste Kamera/Video
                    #dequeue interene Queue
                    # print("befindet sich im Exit Plain")
                #entfernen des Elements aus der Queue / ich checke ja sowieso nur, ob es neue Elemente sind, also deque ich die gar nciht heir

                internalQueue.get() 
                '''
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

                print("Länge der Featutes: " + str(len(features_dict)))

                if len(features_dict) == 0:
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
#cv2.imshow('image', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
        
        return img