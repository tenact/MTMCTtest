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
from neuda import Neudas

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
#data_deque = {}
id_list = []
features = []
dicti = {}
#features_dict = {}
test = True
basic_infos = {}

entryQueue = []
exitQueue = []

#queue 0 nur 1 und ggf4? - nien
#queue 1 1 und 2
#aueue 2 2 und 3
#queue 3 3 und 4
#queue 4 nur 4

queue1 = []
queue2 = []
queue3 = []
queue4 = []


#deepsort = None

object_counter = {}


object_counter1 = {}



#draw boxex
#xyxy_to_xywh
#werden von write_results genutzt, also von DetectionPredictor aufgerufen
# die andeeren Methoden wie drawBox, verwenden DeepSort
# ich kann eine neue Klasse erstellen, enthält alle Methoden. Ich erstelle eine Instanz und DeepSort arbeitet 
# in der Instanz von alleine und ich kann somit 4 Instanzen erstellen
# und die methoden drawBox undwrite_results von der Instanz aufrufen

def xyxy_to_xywh(*xyxy):
        """" Calculates the relative bounding box from absolute pixel values. """
        bbox_left = min([xyxy[0].item(), xyxy[2].item()])
        bbox_top = min([xyxy[1].item(), xyxy[3].item()])
        bbox_w = abs(xyxy[0].item() - xyxy[2].item())
        bbox_h = abs(xyxy[1].item() - xyxy[3].item())
        x_c = (bbox_left + bbox_w / 2)
        y_c = (bbox_top + bbox_h / 2)
        w = bbox_w
        h = bbox_h
        return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

def compute_color_for_labels(label):
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




class DetectionPredictor(BasePredictor):

    infos2 = {}
    deepsort = None
    c = None
    def __init__(self, config=DEFAULT_CONFIG, overrides=None, infos={}): # zusätzliche Übergabe des Dictionaries
        """
        Initializes the BasePredictor class.

        Args:
            config (str, optional): Path to a configuration file. Defaults to DEFAULT_CONFIG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
            queues (dict, optional): A dictionary of queues. Defaults to an empty dictionary.
        """
        print("im dectectoi predictior")
        super().__init__(config, overrides)
        print("die Results werden geschrieben")
        b = Neudas()
        b.init_tracker()
        global deepsort
        global c
        c = b
        deepsort = b.getDeepSort()
        
        #self.infos2 = infos
        print(infos)

        # ich muss eine Dequeu übergeben, aber nicht bei drawboxes, dann würde ich immer die selbe übergeben
        # Definieren hier einer drawbox, und üvergabe an drawboxes?
        # set dequeu in deepSort hinzufügen, und dann hinzufügen, dann sollte das stimmen.
    
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):

        

        print("is deepsort None? " + str(deepsort is None))
        print("deepSort: " + str(deepsort))
        
        p, im, im0 = batch
        all_outputs = []
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}') # print in die Konsole?
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        all_outputs.append(det)
        if len(det) == 0:
            return log_string # wenn nichts detected wird, einfach log string in terminal ausgeben
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        xywh_bboxs = []
        confs = []
        oids = []
        outputs = []
        for *xyxy, conf, cls in reversed(det):
            print("hier wird gezeichnet")
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
            xywh_bboxs.append(xywh_obj)
            confs.append([conf.item()])
            oids.append(int(cls))
        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)

      
        #printen der yolo-Werte, die an DeepSort übergeben werden
        print("xywhs: " + str(xywhs))
        print("confss: " + str(confss))
        print("oids: " + str(oids))
        print("im0: " + str(im0))  
        outputs = deepsort.update(xywhs, confss, oids, im0) 
        
        
        print("outputs: " + str(outputs))

        # outputs bleiben immmer leer, also muss da irgendwie ein Problem bei DeepSort sein
        # DeppSort Initalisiert, und die Infos werden übergeben?
        #Return Wert von DeepSort bleibt leer, leere Liste

        if len(outputs) > 0: 
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]      
            print("identities: " + str(identities))
            print("object_id: " + str(object_id))  
            print(c)
            c.draw_boxes(im0, bbox_xyxy, self.model.names, object_id,identities)

            '''
            print( "das dict: " + str(self.infos[0]))
            if test:
                deepsort.draw_boxes(im0, bbox_xyxy, self.model.names, object_id,identities, self.infos[0])
                test = False
            else:
                deepsort.draw_boxes(im0, bbox_xyxy, self.model.names, object_id,identities, self.infos[1])
   
            '''
        return log_string
lock = multiprocessing.Lock()


def process_video(video_path, cfg, counter, dicti):
        
        with lock:
           
            

            print(f"Processing video: {video_path}")
            cfg.source = video_path
            #deepSort_tmp = init_tracker()
            print("Initalisierung des Predictors")
            if counter == 1:
                predictor = DetectionPredictor(cfg, dicti)
            else:
                predictor = DetectionPredictor(cfg, dicti) # Initailizioerung des Predictors
            predictor()
            pass

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):   
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = 'videos' #cfg.source if cfg.source is not None else ROOT / "assets"
   
    # Load all videos in source directory
    video_files = []
    if os.path.isdir(cfg.source):
        for file in os.listdir(cfg.source):
            if file.endswith('.mp4') or file.endswith('.avi'):
                video_files.append(os.path.join(cfg.source, file))
    else:
        video_files.append(cfg.source)

    counter = 0
    for video_path in video_files:
        
        counter += 1
        processes = []
        print(counter)


         #EntryPlain
        p1 = 50 # x value
        p2 = 1230 #x value
        q1 = 10 #y value
        q2 = 150 #y value

        #Exit Plain
        v1 = 50 # x value
        v2  =1230 #x value
        f1  = 750   
        f2 = 650

        entryQ = list(deque())
        exitQ = list(deque())
        
        #dicti = {}
        dicti2 = {}
        entryPLain = (p1,p2,q1,q2)
        exitPlain = (v1,v2,f1,f2)

        if counter == 1:
            dicti[0] = {0: {"entryPlain": entryPLain, "exitPlain": exitPlain, "entryQueue": entryQ, "exitQueue": exitQ}}
        elif counter == 2:
            dicti2[2] = {1: {"entryPlain": entryPLain, "exitPlain": exitPlain, "entryQueue": entryQ, "exitQueue": exitQ}}
        else:
            dicti.update({"entryPlain": entryPLain, "exitPlain": exitPlain, "entryQueue": entryQ, "exitQueue": exitQ})

        #übergabe 
        if counter ==1:
            process = multiprocessing.Process(target=process_video, args=(video_path, cfg, counter, dicti))
        else:
            process = multiprocessing.Process(target=process_video, args=(video_path, cfg, counter, dicti2))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()


if __name__ == "__main__":
    predict()


