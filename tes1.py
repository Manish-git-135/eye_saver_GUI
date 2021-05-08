"""
	This file is for reference
"""
import tkinter
from tkinter.constants import SE, SEL, TRUE
from PIL.Image import preinit
import cv2
import PIL.Image
import PIL.ImageTk
import time
import dlib
from numpy.core import shape_base
from numpy.core.fromnumeric import shape
import numpy as np
import imutils
from scipy.spatial import distance as dist
from imutils import face_utils


class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.predictor_path = 'C:/Users/manis/Documents/opencv/Detectors/shape_predictor_68_face_landmarks.dat'
        self.predictor = dlib.shape_predictor(self.predictor_path)
        self.detector = dlib.get_frontal_face_detector()
        self.lStart, self.lEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        self.rStart, self.rEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.COUNTER = 0
        self.TOTAL = 0
        self.EYE_AR_CONSEC_FRAMES = 3
        self.EYE_AR_THRESH = 0.24

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(
            window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot = tkinter.Button(
            window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)

        # Label for distance measure
        self.dist_label = tkinter.Label(
            window, text="", padx=34, pady=12, bg='black', fg='white')
        self.dist_label.pack()

        # Label for count eye blink
        self.count_label = tkinter.Label(
            window, text="", padx=34, pady=12, bg='black', fg='white')
        self.count_label.pack()

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 1
        self.update()

        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") +
                        ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        
        try:
            if ret:
                self.photo = PIL.ImageTk.PhotoImage(
                    image=PIL.Image.fromarray(frame))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = self.detector(gray, 0)
                for rect in rects:
                    shape = self.predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    self.dist_label['text'] = cv2.contourArea(
                        np.reshape(shape, (68, 1, 2)))
                if True:
                    leftEAR = 0
                    rightEAR = 0
                    ear = 0
                    leftEye = 0
                    rightEye = 0
                    for rect in rects:
                        shape = self.predictor(gray, rect)
                        shape = face_utils.shape_to_np(shape)

                        leftEye = shape[self.lStart:self.lEnd]
                        rightEye = shape[self.rStart:self.rEnd]
                        leftEAR = self.eye_aspect_ratio(leftEye)
                        rightEAR = self.eye_aspect_ratio(rightEye)

                        ear = (leftEAR + rightEAR)/2.0
                        try:
                            leftEyeHull = cv2.convexHull(leftEye)
                            rightEyeHull = cv2.convexHull(rightEye)
                            cv2.drawContours(
                                frame, [leftEyeHull], -1, (0, 255, 0), 1)
                            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                            if ear < self.EYE_AR_THRESH:
                                self.COUNTER += 1
                            else:
                                if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                                    self.TOTAL += 1
                                self.COUNTER = 0
                                cv2.putText(frame, "Blinks: {}".format(self.TOTAL), (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                self.count_label['text'] = self.TOTAL
                        except Exception as e:
                            print(e, 'error')
        except Exception as e:
            print(e)

        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)

        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)


    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            frame = cv2.flip(frame, 1)
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            # return (ret, None)
            pass

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


# Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV")
