import cv2
import os

class LoadImages:
    def __init__(self, path):
        self.path = path
        self.files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __iter__(self):
        for file in self.files:
            img = cv2.imread(file)
            yield img, file

class LoadStreams:
    def __init__(self, source):
        self.source = source

    def __iter__(self):
        cap = cv2.VideoCapture(self.source)
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break
            yield img, self.source
        cap.release()
