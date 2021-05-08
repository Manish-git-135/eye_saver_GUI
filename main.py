"""
	This file is for reference
"""
import cv2
from tkinter import *
from PIL import ImageTk, Image
import cv2
import brightness
import distance
import eye
import numpy as np
import imutils
from imutils import face_utils
import dlib
from scipy.spatial import distance as dist
import time
import threading
from multiprocessing import Process

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
# images for buttons
distance_btn_image = PhotoImage(file='./images/distance.png')
eyeBlinkCount_btn_image = PhotoImage(file='./images/eyeCount.png')
start_btn_image = PhotoImage(file="./images/start.png")

# function for find face area


def get_area():
    dist1 = distance.get_face_area(15000, cap)
    dist_label['text'] = dist1
    if dist1 <= 15000:
        dist_label['bg'] = "#008000"
    elif dist1 >= 15000:
        dist_label['bg'] = '#FF0000'


# Button for face area
Button(root, image=distance_btn_image, border=0, borderwidth=0, highlightthickness=0,
       fg='white', command=get_area).place(x=110, y=110)

# function for count blink


def get_eye_blink_count():
	startTime = time.time()
	count_label['text'] = 'Plase wait'
	count = eye.eye_blink_counter(cap)
	count_label['text'] = count
	endTime = time.time()
	print(endTime-startTime)


# button for count eye blink count
Button(root, image=eyeBlinkCount_btn_image, border=0, borderwidth=0, highlightthickness=0,
       fg='white', command=get_eye_blink_count).place(x=110, y=210)


face_vs_screen_ratio = 0
predictor_path = 'C:/Users/manis/Documents/opencv/Detectors/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

EYE_AR_THRESH = 0.24
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0
TOTAL = 0
AREA = 0

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def update_label(AREA, TOTAL):
    print(AREA, TOTAL)
    dist_label.config(text=AREA)
    count_label.config(text=TOTAL)
    # update_label(1000, update_label)


def distance_and_eye_blink_count(COUNTER, TOTAL, EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES, AREA):
	CAP = cv2.VideoCapture(0)
	while True:
		_, frame = CAP.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		rects = detector(gray, 0)
		for rect in rects:
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)
			area = cv2.contourArea(np.reshape(shape, (68, 1, 2)))
			if area <= 15000:
				print(area)
				# update_label(area, TOTAL)
				AREA = area
				print('You are in safe Zone')

		# frame = imutils.resize(frame, width=450)
		# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# rects = detector(frame, 0)

		# leftEAR = 0
		# rightEAR = 0
		# ear = 0
		# leftEye = 0
		# rightEye = 0
		# for rect in rects:
		# 	shape = predictor(gray, rect)
		# 	shape = face_utils.shape_to_np(shape)

		# 	leftEye = shape[lStart:lEnd]
		# 	rightEye = shape[rStart:rEnd]
		# 	leftEAR = eye_aspect_ratio(leftEye)
		# 	rightEAR = eye_aspect_ratio(rightEye)

		# 	ear = (leftEAR + rightEAR) / 2.0
		# try:
		# 	leftEyeHull = cv2.convexHull(leftEye)
		# 	rightEyeHull = cv2.convexHull(rightEye)
		# 	cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		# 	cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# 	if ear < EYE_AR_THRESH:
		# 		COUNTER += 1

		# 	else:
		# 		if COUNTER >= EYE_AR_CONSEC_FRAMES:
		# 			TOTAL += 1

		# 		COUNTER = 0

		# 		# cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
		# 		# 			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		# 		# cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
		# 		# 			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# 	cv2.imshow("Frame", frame)

		# 	if cv2.waitKey(1) & 0xFF == ord("q"):
		# 		return TOTAL
		# 	print(TOTAL)
		# 	# update_label(area, TOTAL)
		# except Exception as e:
		# 	print(e)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			return


def done():
    t1 = threading.Thread(target=distance_and_eye_blink_count, args=(
	    COUNTER, TOTAL, EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES, AREA))
    t2 = threading.Thread(target=update_label, args=(AREA, TOTAL))
    t2.start()
    t1.start()
	# p1 = Process(target=distance_and_eye_blink_count, args=(
    #             COUNTER, TOTAL, EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES, AREA))
	# p2 = Process(target=update_label, args=(AREA, TOTAL))
	# p2.start()
	# p1.start


# button for start
Button(root, image=start_btn_image, border=0, borderwidth=0, highlightthickness=0,
       fg='white', command=done).place(x=90, y=380)

v1 = DoubleVar()
v1.set(brightness.current_brightness)
Scale(root, from_=1, to=100, bg="white", label="Brightness",
      highlightthickness=0, length=200, command=lambda val: brightness.set_brightness(val), variable=v1).place(x=300, y=100)

Scale(root, from_=1, to=100, bg="white", label="Yellow",
      highlightthickness=0, length=200).place(x=400, y=100)

dist_label = Label(root, text="", pady=12, padx=18,
                   bg='#B9C7C4', fg='white')
dist_label.place(x=420, y=380)
count_label = Label(root, text="", pady=12, padx=18,
                    bg="#B9C7C4")
count_label.place(x=590, y=380)


root.mainloop()
