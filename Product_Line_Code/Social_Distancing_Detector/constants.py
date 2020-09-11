import numpy as np

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

prototxt_path = 'models/MobileNetSSD_deploy.prototxt.txt'

model_path = 'models/MobileNetSSD_deploy.caffemodel'

sound_file = 'sound_files/warning.wav'

frame_width_in_pixels = 320
