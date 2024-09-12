NO I DID NOT STEAL THIS I AM GOING TO USE IT TO HELP SOLVE SOME PROBLEMS GIVE ME A MINUTE TO FIX IT AND I WILL TAKE THIS DOWN AND PUT UP MY OWN PERSONAL ONE 
import numpy as np
import cv2 as cv
import mss
import mss.tools
from PIL import Image
import win32gui  # Import the win32gui module
import os
import random
import time

# WindowCapture class captures screenshots from the specified window using MSS
class WindowCapture:
    w = 0
    h = 0
    monitor = None
    hwnd = None

    def __init__(self, window_title):
        # Initialize MSS for screen capture
        self.sct = mss.mss()

        # Find the window handle by title
        self.hwnd = win32gui.FindWindow(None, window_title)
        if not self.hwnd:
            raise Exception(f'Window not found: {window_title}')

        # Get the window’s position and size
        rect = win32gui.GetWindowRect(self.hwnd)
        self.monitor = {
            "top": rect[1],
            "left": rect[0],
            "width": rect[2] - rect[0],
            "height": rect[3] - rect[1]
        }
        self.w = self.monitor["width"]
        self.h = self.monitor["height"]

    def get_screenshot(self):
        # Capture the screen using MSS
        screenshot = self.sct.grab(self.monitor)
        
        # Convert to a format usable by OpenCV
        img = np.array(Image.frombytes("RGB", (screenshot.width, screenshot.height), screenshot.rgb))
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        return img

    def get_window_size(self):
        # Get the size of the window
        return (self.w, self.h)

# ImageProcessor class processes the images and performs object detection
class ImageProcessor:
    W = 0
    H = 0
    net = None
    ln = None
    classes = {}
    colors = []

    def __init__(self, img_size, cfg_file, weights_file):
        # Initialize the neural network
        np.random.seed(42)
        self.net = cv.dnn.readNetFromDarknet(cfg_file, weights_file)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i-1] for i in self.net.getUnconnectedOutLayers()]
        self.W = img_size[0]
        self.H = img_size[1]

        # Load the class labels
        with open('yolov4-tiny/obj.names', 'r') as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            self.classes[i] = line.strip()

        # Define the colors for the bounding boxes
        self.colors = [
            (0, 0, 255), 
            (0, 255, 0), 
            (255, 0, 0), 
            (255, 255, 0), 
            (255, 0, 255), 
            (0, 255, 255)
        ]

    def process_image(self, img):
        # Process the image and perform object detection
        blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.ln)
        outputs = np.vstack(outputs)

        coordinates = self.get_coordinates(outputs, 0.5)

        self.draw_identified_objects(img, coordinates)

        return coordinates

    def get_coordinates(self, outputs, conf):
        # Get the coordinates of the detected objects
        boxes = []
        confidences = []
        classIDs = []

        for output in outputs:
            scores = output[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > conf:
                x, y, w, h = output[:4] * np.array([self.W, self.H, self.W, self.H])
                p0 = int(x - w//2), int(y - h//2)
                boxes.append([*p0, int(w), int(h)])
                confidences.append(float(confidence))
                classIDs.append(classID)

        indices = cv.dnn.NMSBoxes(boxes, confidences, conf, conf-0.1)

        if len(indices) == 0:
            return []

        coordinates = []
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            coordinates.append({'x': x, 'y': y, 'w': w, 'h': h, 'class': classIDs[i], 'class_name': self.classes[classIDs[i]]})

        return coordinates

    def draw_identified_objects(self, img, coordinates):
        # Draw the bounding boxes and class labels on the image
        for coordinate in coordinates:
            x = coordinate['x']
            y = coordinate['y']
            w = coordinate['w']
            h = coordinate['h']
            classID = coordinate['class']

            color = self.colors[classID]

            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv.putText(img, self.classes[classID], (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv.imshow('window', img)

# Main program execution

window_title = "BloonsTD6"  # Update this to match the exact title of the game window
cfg_file_name = "./yolov4-tiny/yolov4-tiny-custom.cfg"
weights_file_name = "yolov4-tiny-custom_last.weights"
wincap = WindowCapture(window_title)
improc = ImageProcessor(wincap.get_window_size(), cfg_file_name, weights_file_name)

while(True):
    # Get a screenshot from the game window
    ss = wincap.get_screenshot()

    # If 'q' is pressed on the keyboard, stop the loop
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

    # Process the screenshot and perform object detection
    coordinates = improc.process_image(ss)

    # Print the coordinates of the detected objects
    for coordinate in coordinates:
        print(coordinate)
    print()

    # Optionally add a delay to reduce CPU usage
    time.sleep(0.2)

print('Finished.')


[![Watch the video](https://img.youtube.com/vi/RSXgyDf2ALo/maxresdefault.jpg)](https://youtu.be/RSXgyDf2ALo)

```

With the data above, you can create your own logic for automating character actions. You can use libraries like [**pynput**](https://pypi.org/project/pynput/) to control your mouse or keyboard based on the detections.

In this video below I demonstrate how you can use the model predictions with pynput to automate actions in a game.

[![Watch the video](https://img.youtube.com/vi/gdIVHdRbhOs/maxresdefault.jpg)](https://youtu.be/gdIVHdRbhOs)

## Getting Started

To get started with this tutorial, follow these steps:

1. Clone this repository.

2. Open a command line in the repository folder.

3. Install the required dependencies by running:

```pip install -r requirements.txt```

4. Launch Jupyter Notebook by executing:

```jupyter notebook .```

5. Execute the step-by-step instructions provided in the Jupyter notebooks.

## References

This guide would not be possible without other great tutorials that I used as references:

1. [**OpenCV Object Detection in Games** - Learn Code by Gaming](https://www.youtube.com/playlist?list=PL1m2M8LQlzfKtkKq2lK5xko4X-8EZzFPI)
    - In this tutorial, you will learn how to use OpenCV for object detection in images using Template matching. It's a great tutorial, very well explained and I highly recommend watching it and also the channel other playlists to learn more about OpenCV.

2. [**YOLO Object Detection with OpenCV** - PyImageSearch](https://pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)
    - If you want to dive deeper into using YOLO for object detection in images or video streams using Python, I recommend reading this article for more details on this topic.

3. [**TRAIN A YOLOv4 DETECTOR USING GOOGLE COLAB** - Techzizou](https://medium.com/analytics-vidhya/train-a-custom-yolov4-tiny-object-detector-using-google-colab-b58be08c9593)
    - In this tutorial, you can find instructions on running the detector in a live video and also obtaining metrics for your trained model performance. If you want more details about the training process, I recommend reading his Medium article.



## Contact

If you have any questions or want to share your projects based on this tutorial, please find me on [LinkedIn](https://www.linkedin.com/in/moisesdias/). I'd love to see your results.

