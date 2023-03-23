
from cameraLink import DetectionPredictor


def process_video(video_path, cfg):
        print(f"Processing video: {video_path}")
        cfg.source = video_path
        predictor = DetectionPredictor(cfg) # i changed this to video_path from cfg // but now exceptions in all threads
        print("Initalisierung des Predictors")
        predictor()