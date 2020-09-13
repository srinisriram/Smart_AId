prototxt_path = "face_detection_model/deploy.prototxt"

model_path = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"

embedder_path = "face_embedding_model/openface_nn4.small2.v1.t7"

recognizer_path = "face_recognizer_pickle/recognizer.pickle"

labels_path = "face_recognizer_pickle/labels.pickle"

abhisar_sound_path = "sound_files/abhisar.wav"

srinivas_sound_path = "sound_files/srinivas.wav"

aditya_sound_path = "sound_files/aditya.wav"

arsh_sound_path = "sound_files/arsh.wav"

COLORS = [(255, 0, 0), (0, 191, 255), (50, 205, 50)]

LABELS = ["Abhisar", "Srinivas", "Aditya"]

frame_width_in_pixels = 320

OPEN_DISPLAY = True

MIN_CONFIDENCE = 0.7

MIN_CONFIDENCE_FOR_FACE = 0.95

MOTOR1_FORWARD_GPIO = 14

ON = 1

OFF = 0

