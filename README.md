# Smart_AId


## Introduction

We at Smart_AId, (Abhisar Anand, Aditya Anand, Arsh Sharma, Srinivas Sriram) participated in the Girls Computing League's 4th AI Innovation Business Summit. We decided to create 4 different products in response to the COVID-19 pandemic going on in the world right now. These products aim to make reopening of businesses and schools in this unfamiliar time as easy as possible, with the help of AI, Deep Learning, Raspberry Pi's + Intel Movidius Sticks, and Cameras. These are our four products as well as how our code works.


## 1. Mask Detector

### Why we did it 
One of the most simple ways to prevent the spread of the virus is simply to wear a mask or face covering. However, many people forget to do this simple task, and this can easily spread the virus. In response to this, we created a mask detector that can be fitted at the entrance of a building to detect if people are wearing a face mask as they enter the building. 

### How we built it.
To start with, we had to be able to choose a model that was able to run on the Raspberry Pi 4 + Intel Movidius Stick. So for that, we used a face detection model caffe model for detecting if a face is there, a pytorch model for extracting and embedding the face, and finally, a linear SVM scikit-learn machine used for recognizing and predicting the class of the face (with or without mask). Here are some code snippets containing how we trained/used these models. 

Loading face detection caffe model:
```
self.detector = cv2.dnn.readNetFromCaffe(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            prototxt_path),
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                model_path))

        self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
 
```

Loading face extraction/embedding pytorch model:
```
self.embedder = cv2.dnn.readNetFromTorch(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            embedder_path))

        self.embedder.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
```

Training and using SVM scikit-learn model
```
        params = {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
          "gamma": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]}
        model = GridSearchCV(SVC(kernel="rbf", gamma="auto",
          probability=True), params, cv=3, n_jobs=-1)
        model.fit(data["embeddings"], labels)

        self.recognizer = pickle.loads(open(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            recognizer_path), "rb").read())

        self.le = pickle.loads(open(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            labels_path), "rb").read())

        self.predictions = self.recognizer.predict_proba(self.vec)[0]
        self.j = np.argmax(self.predictions)
        self.probability = self.predictions[self.j]
        self.name = self.le.classes_[self.j]

```

Now that you know what Deep Learning models we are using, let's talk about how we use them in real time. To start with, we use one of python's best open source computer vision libraries, OpenCV. We use OpenCV to start a video stream with the camera, and then constantly read each frame of the camera in a while True loop. In each frame, we detect for a face using our caffe model, and if a face is detected, we use our pytorch model to extract the face. We then pass this extracted face into our SVM Classifier, and then the Classifier predicts whether the person is or is not wearing a face mask. 

After the class label is outputted, we then take corresponding action. If the person is not wearing a face mask, we play a computer-generated warning asking the person to wear a mask. If the person is wearing a mask, we use Raspberry Pi's GPIO pins to send a signal to our motor controller, which then causes the motor to spin, which will open the door to allow the person inside. 


## 2. Attendance Tracker.

### Why we did it
The reason why we want to create an automatic attendance tracker is because we believe that companies and schools can use this technology in a lot of ways. For example, if a worker or employee has contracted the virus, the manager can use our data to see who was in the building at the time, and then take action to help quarantine them so they can keep going. Overall, we just feel like having an automatic attendance system not only saves the manager time, but also can provide valuable data to them, especially during the pandemic.

### How we built it
The Deep Learning involved in the Attendance System and the Mask Detector are very similar. They both utilize the face detection caffe model and the face embedding/extracting pytorch model. However, our SVM model was trained a bit differently. Instead of recognizing mask or no mask, we trained a model off of our faces so that it would recognize each of us. 

We loop through the video stream (mentioned on Mask Detector) and receive the person's name. Using that name, we play a customized message which says welcome specifically for that person. (ex. If Srinivas is detected, it will play Welcome Srinivas). After that, we open the door using Raspberry Pi GPIO pins (mentioned on Mask Detector) to let the person inside.











