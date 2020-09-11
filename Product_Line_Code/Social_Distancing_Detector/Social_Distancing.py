import time
import cv2
import numpy as np
import math
import simpleaudio as sa


def calcDistance(x):
    y = 900 / x * 12
    # y = 275 / x * 12
    y = round(y)
    return y


def get_angle(a, b, c):
    angle1 = (abs(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])))
    angle2 = (abs(math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))))
    return round(angle1, 0), round(angle2, 0)


def finalDist(firstSide, angle, secondSide):
    thirdSide = math.sqrt((secondSide ** 2) + (firstSide ** 2) - 2 * secondSide * firstSide * math.cos(angle))
    return round(thirdSide)


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe(prototxt="models/MobileNetSSD_deploy.prototxt.txt",
                               caffeModel="models/MobileNetSSD_deploy.caffemodel")
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

vs = cv2.VideoCapture(0)

filename = 'speech.wav'
wave_obj = sa.WaveObject.from_wave_file(filename)
arr = []

while True:

    faces = []

    ret, frame = vs.read()

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = round(idx)

            if label == 15:
                box = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype('int')
                box2 = (box[0], box[1], box[2] - box[0], box[3] - box[1])
                faces.append(box2)
                (xCorr, yCoor, width, height) = (box[0], box[1], box[2] - box[0], box[3] - box[1])
                dist = calcDistance(width)

                labels = "{}: {:.2f}%".format(CLASSES[idx],
                                              confidence * 100)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]),
                              COLORS[idx], 2)
                y = box[1] - 15 if box[1] - 15 > 15 else box[1] + 15
                cv2.putText(frame, str(label) + " " + str(dist), (box[0], y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    if len(faces) > 1:
        for i in range(len(faces) - 1):
            ang, ang2 = get_angle(a=(faces[i][0], faces[i][1]), b=(320, 480), c=(faces[i + 1][0], faces[i + 1][1]))
            dist1 = calcDistance(faces[i][2])
            dist2 = calcDistance(faces[i + 1][2])
            finalDistance = finalDist(ang, dist2, dist1)
            if len(arr) <= 20:
                arr.append((finalDistance / 12))
            else:
                Asum = sum(arr) / len(arr)
                print(Asum)
                Asum -= 1
                print("Face", i, " & ", i + 1, ": ", round(Asum, 1), " feet apart.")
                arr.clear()

                if Asum <= 5:
                    play_obj = wave_obj.play()
                    play_obj.wait_done()

    faces.clear()
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.release()
