from tkinter.constants import E
from imutils import face_utils
import dlib
import cv2
import numpy as np
import time
from PIL import ImageTk, Image

face_vs_screen_ratio = 0
predictor_path = 'C:/Users/manis/Documents/opencv/Detectors/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()
# cap = cv2.VideoCapture(0)


def get_face_area(safe_area, cap):
    try:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            area = cv2.contourArea(np.reshape(shape, (68, 1, 2)))
            if area <= safe_area:
                print(area)
                print('You are in safe Zone')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return area
        return area
    except Exception as e:
        print(e)
