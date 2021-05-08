from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import time


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


EYE_AR_THRESH = 0.24
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0
TOTAL = 0
shape_pred = "C:/Users/manis/Documents/opencv/Detectors/shape_predictor_68_face_landmarks.dat"
# cap = cv2.VideoCapture(0)

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_pred)


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


def blink(cap, COUNTER, TOTAL, EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES):
    while True:
        start = time.time()
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
        try:
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

                cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Frame", frame)
            if TOTAL >= 2:
                return 
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return TOTAL
            print(TOTAL)
        except Exception as e:
            print(e)
        end = time.time()
        print(end-start, 'time')


def eye_blink_counter(cap):
    return blink(cap, COUNTER, TOTAL, EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES)

cv2.destroyAllWindows()
