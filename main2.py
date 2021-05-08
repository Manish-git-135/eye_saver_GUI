"""
	This file is for reference
"""
from tkinter import *
from PIL import ImageTk, Image
import cv2
import numpy as np
import imutils
from imutils import face_utils
import dlib
from scipy.spatial import distance as dist
import time
import threading

root = Tk()
root.geometry('850x550')
root.resizable(width=0, height=0)
root.title("Eye Saver")
root.configure(bg='white')

# Create a frame
app = Frame(root, bg="white")
app.place(x=505, y=23)
lmain = Label(app)  # Create a label in the frame
lmain.grid()
# Capture from camera
cap = cv2.VideoCapture(0)

# function for video streaming


def video_stream():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(1, video_stream)


video_stream()

face_vs_screen_ratio = 0
predictor_path = 'C:/Users/manis/Documents/opencv/Detectors/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()


def distance():
    try:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            area = cv2.contourArea(np.reshape(shape, (68, 1, 2)))
            if area <= 15000:
                dist_label['bg'] = 'green'
                dist_label['fg'] = 'white'
                dist_label['text'] = area
            elif area > 15000:
                dist_label['bg'] = 'red'
                dist_label['fg'] = 'white'
                dist_label['text'] = area
                print('You are in safe Zone')
    except Exception as e:
        print(e)
    root.after(100, distance)


# Distance btn
distance_btn_image = PhotoImage(file='./images/distance.png')
dist_label = Label(root, text="", pady=12, padx=18,
                   bg='#B9C7C4', fg='white')
dist_label.place(x=420, y=380)

Button(root, image=distance_btn_image, border=0, borderwidth=0, highlightthickness=0,
       fg='white', command=distance).place(x=110, y=110)

# * blink count code start from here
global COUNTER, TOTAL, EYE_AR_CONSEC_FRAMES, EYE_AR_THRESH
EYE_AR_THRESH = 0.24
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0
TOTAL = 0
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def blink_count():
    try:
        global COUNTER, TOTAL, EYE_AR_CONSEC_FRAMES, EYE_AR_THRESH
        _, frame = cap.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(frame, 0)

        leftEAR = 0
        rightEAR = 0
        ear = 0
        leftEye = 0
        rightEye = 0
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1

        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            COUNTER = 0
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return TOTAL
        print(TOTAL)
        count_label['text'] = TOTAL
    except Exception as e:
        print(e)
    root.after(80, blink_count)


# Eye blink count btn
eyeBlinkCount_btn_image = PhotoImage(file='./images/eyeCount.png')
count_label = Label(root, text="", pady=12, padx=18,
                    bg="#B9C7C4")
count_label.place(x=590, y=380)
Button(root, image=eyeBlinkCount_btn_image, border=0, borderwidth=0, highlightthickness=0,
       fg='white', command=blink_count).place(x=110, y=210)

mainloop()
