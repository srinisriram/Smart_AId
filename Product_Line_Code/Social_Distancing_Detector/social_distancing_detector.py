import time
import cv2
import numpy as np
import math
from constants import CLASSES, COLORS, prototxt_path, model_path
from distance_calculations import calcDistance, get_angle, finalDist
import threading


class SocialDistancing:
    preferableTarget = None

    def __init__(self):
        self.net = None
        self.vs = None
        self.faces = []
        self.frame = None
        self.h = None
        self.w = None
        self.blob = None
        self.detections = None
        self.confidence = None
        self.idx = None
        self.label = None
        self.box = None
        self.box2 = None
        self.xCorr = None
        self.yCorr = None
        self.width = None
        self.height = None
        self.dist = None
        self.labels = None
        self.y = None
        self.arr = []
        self.ang = None
        self.ang2 = None
        self.finalDistance = None
        self.Asum = None
        self.key = None

        self.load_models()
        self.start_video_stream()

    @classmethod
    def perform_job(cls, preferableTarget=cv2.dnn.DNN_TARGET_MYRIAD):
        SocialDistancing.preferableTarget = preferableTarget
        t1 = threading.Thread(target=SocialDistancing.thread_for_social_distancing_detection)
        t1.start()

    def load_models(self):
        self.net = cv2.dnn.readNetFromCaffe(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            prototxt_path),
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                model_path))
        self.net.setPreferableTarget(SocialDistancing.preferableTarget)

    def start_video_stream(self):
        print("[INFO] starting video stream...")
        self.vs = cv2.VideoCapture(0)

