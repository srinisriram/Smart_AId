import sys
import threading
import time

import cv2
import imutils
import numpy as np
from constants import prototxt_path, model_path, frame_width_in_pixels, CLASSES, COLORS
from distance_calculations import calcDistance, get_angle, finalDist
from imutils.video import VideoStream
from play_audio_social import PlayAudio


class SocialDistancing:
    preferableTarget = None
    run_program = True

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
        self.AudioPlay = True
        self.humanIndex = 15

        self.load_models()
        self.start_video_stream()

    @classmethod
    def perform_job(cls, preferableTarget=cv2.dnn.DNN_TARGET_MYRIAD):
        """
        This method performs the job expected from this class.
        :key
        """
        SocialDistancing.preferableTarget = preferableTarget
        t1 = threading.Thread(target=SocialDistancing().thread_for_social_distancing_detection)
        t1.start()

    def load_models(self):
        """
        This method will load the caffe model that we will use for detecting humans, and then set the preferable target to the correct target.
        :key
        """
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        self.net.setPreferableTarget(SocialDistancing.preferableTarget)

    def start_video_stream(self):
        """
        This method will initialize the camera.
        :key
        """
        print("[INFO] starting video stream...")
        self.vs = VideoStream(src=0).start()
        time.sleep(2.0)

    def grab_next_frame(self):
        """
        This method extracts the next frame from the video stream.
        :key
        """
        self.frame = self.vs.read()
        if self.frame is None:
            return
        self.frame = imutils.resize(self.frame, width=frame_width_in_pixels)

    def set_dimensions_for_frame(self):
        """
        This method will set the frame dimensions, which we will use later on.
        :key
        """
        if not self.h or not self.w:
            (self.h, self.w) = self.frame.shape[:2]

    def rotate_frame(self):
        """
        This method will rotate the frame as the camera was installed upside down.
        :key
        """
        self.frame = cv2.rotate(self.frame, cv2.ROTATE_180)

    def create_frame_blob(self):
        """
        This method will create a blob for our human detector to detect a human.
        :key
        """
        self.blob = cv2.dnn.blobFromImage(cv2.resize(self.frame, (300, 300)), 0.007843, (300, 300), 127.5)

    def extract_human_detections(self):
        """
        This method will extract each human detection that our human detection model provides.
        :return:
        """
        self.net.setInput(self.blob)
        self.detections = self.net.forward()

        for i in np.arange(0, self.detections.shape[2]):
            self.confidence = self.detections[0, 0, i, 2]
            if self.confidence > 0.5:
                self.idx = int(self.detections[0, 0, i, 1])
                self.label = round(self.idx)

                if self.label == self.humanIndex:
                    self.find_dist_each_human(i)

        self.find_dist_between_humans()

    def find_dist_each_human(self, i):
        """
        This method will find the distance for each human from the camera and draw the bounding boxes onto the frame.
        :return:
        """
        self.box = (self.detections[0, 0, i, 3:7] * np.array([self.w, self.h, self.w, self.h])).astype('int')
        self.box2 = (self.box[0], self.box[1], self.box[2] - self.box[0], self.box[3] - self.box[1])
        self.faces.append(self.box2)
        (self.xCorr, self.yCoor, self.width, self.height) = (
            self.box[0], self.box[1], self.box[2] - self.box[0], self.box[3] - self.box[1])
        self.dist = calcDistance(self.width)

        self.labels = "{}: {:.2f}%".format(CLASSES[self.idx],
                                           self.confidence * 100)
        cv2.rectangle(self.frame, (self.box[0], self.box[1]), (self.box[2], self.box[3]),
                      COLORS[self.idx], 2)
        self.y = self.box[1] - 15 if self.box[1] - 15 > 15 else self.box[1] + 15
        cv2.putText(self.frame, str(self.label) + " " + str(self.dist), (self.box[0], self.y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[self.idx], 2)

    def find_dist_between_humans(self):
        """
        This method will find the distance between all the humans in the frame
        and if the distance is less than 6 ft, it will play audio.
        :return:
        """
        if len(self.faces) > 1:
            for i in range(len(self.faces) - 1):
                self.ang, self.ang2 = get_angle(a=(self.faces[i][0], self.faces[i][1]), b=(320, 480),
                                                c=(self.faces[i + 1][0], self.faces[i + 1][1]))
                self.dist1 = calcDistance(self.faces[i][2])
                self.dist2 = calcDistance(self.faces[i + 1][2])
                print("a=", self.dist1, " b=", self.ang2, " c=", self.dist2)
                self.finalDistance = finalDist(self.dist1, self.ang2, self.dist2)
                if len(self.arr) <= 20:
                    self.arr.append((self.finalDistance / 12))
                else:
                    self.Asum = sum(self.arr) / len(self.arr)
                    print("ASUM: ", self.Asum)
                    self.Asum -= 1
                    print("Face", i, " & ", i + 1, ": ", round(self.Asum, 1), " feet apart.")
                    self.arr.clear()

                    if self.Asum <= 5:
                        print("Trying to play Audio...")
                        self.play_audio()
                        self.AudioPlay = False
        self.display_frame()

    def play_audio(self):
        """
        This method is used for playing the social distancing alarm.
        :return:
        """
        print("Inside the function")
        SoundThread = threading.Thread(target=PlayAudio.play_audio_file)
        if not self.AudioPlay:
            self.AudioPlay = True
            print("[INFO]: Starting Sound Thread")
            SoundThread.start()
            if SoundThread.is_alive() == False:
                self.AudioPlay = False
            print("[INFO]: Stopping Sound Thread")

    def display_frame(self):
        """
        This method will display the frame and wait for the user to quit.
        :key
        """
        self.faces.clear()
        cv2.imshow("Frame", self.frame)
        self.key = cv2.waitKey(1) & 0xFF

        if self.key == ord("q"):
            self.clean_up()
            sys.exit()

    def clean_up(self):
        """
        Clean up the cv2 video capture.
        :return:
        """
        cv2.destroyAllWindows()
        self.vs.release()

    def thread_for_social_distancing_detection(self):
        """
        Callable function that will run the social distancing and can be invoked in a thread.
        :return:
        """
        while SocialDistancing.run_program:
            try:
                self.grab_next_frame()
                self.set_dimensions_for_frame()
                self.rotate_frame()
                self.create_frame_blob()
                self.extract_human_detections()
            except ValueError:
                self.clean_up()
                time.sleep(10)
        self.clean_up()


if __name__ == "__main__":
    SocialDistancing.perform_job(preferableTarget=cv2.dnn.DNN_TARGET_MYRIAD)
