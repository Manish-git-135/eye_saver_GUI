from main3 import Distance
import tkinter
from PIL.Image import preinit
from PIL import Image
import cv2
import PIL.Image
import PIL.ImageTk
import time
import dlib
from numpy.core import shape_base
from numpy.core.fromnumeric import shape
import numpy as np
import imutils
from scipy import spatial
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
        # Booline values
        self.dist_Bool = False
        self.count_Bool = False
        self.start_Bool = False
        # class var
        self.eye = Eye()

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.window.geometry('850x550')
        self.window.configure(bg="white")
        self.canvas = tkinter.Canvas(
            window, width=self.vid.width, height=self.vid.height)
        self.canvas.place(x=505, y=23)

        # Button for counting eyeblink
        tkinter.Button(window, text='Distance', bg='#ff9d00', border=0, borderwidth=0, highlightthickness=0,
                       fg='white', command=Eye().dist_bool_change).place(x=110, y=210)

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if ret:
            newframe = cv2.resize(
                frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
            self.photo = PIL.ImageTk.PhotoImage(
                image=PIL.Image.fromarray(newframe))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

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
            return (False, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


class Eye(App):

    def __init__(self, window, window_title, video_source):
        super().__init__(window, window_title, video_source=video_source)

    def ebc_bool_change(self):
        self.dist_Bool = False
        self.ebc_bool = True
        self.start_Bool = False

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def eye_blink_count(self, frame):
        leftEAR = 0
        rightEAR = 0
        ear = 0
        leftEye = 0
        rightEye = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
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
                cv2.drawContours(
                    frame, [rightEyeHull], -1, (0, 255, 0), 1)
                if ear < self.EYE_AR_THRESH:
                    self.COUNTER += 1
                else:
                    if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                        self.TOTAL += 1
                        self.COUNTER = 0
                        self.blink_count_label['text'] = self.TOTAL
            except Exception as e:
                print(e, 'eye blink count error')


App(tkinter.Tk(), "Tkinter and OpenCV")
