#USAGE
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav

#import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2


def sound_alarm(path):

    playsound.playsound(path)


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)

    return ear

ap = argparse.ArgumentParser()
ap.add_argument('-p', "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument('-a', "--alarm", type=str, default="",
                help="path alarm .WAV file")
ap.add_argument('-w', "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())


EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48


COUNTER = 0
ALARM_ON = False


print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)


while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    rects = detector(gray, 0)


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

            cv2.putText(frame, "winkle COUNT : {}".format(COUNTER), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255),
                        2)

            if COUNTER >= EYE_AR_CONSEC_FRAMES:

                if not ALARM_ON:
                    ALARM_ON = True


                    if args["alarm"] != "":
                        t = Thread(target=sound_alarm,
                                   args=(args["alarm"],))
                        t.deamon = True
                        t.start()


                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0
            ALARM_ON = False


        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF



    if key == ord("q"):
        break


cv2.destroyAllWindows()
vs.stop()

# python3 detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav --cascade haarcascade_frontalface_default.xml
# import the necessary packages
from subprocess import call
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

from threading import Thread
import playsound
#import RPi.GPIO as GPIO


def sound_alarm(path):

    playsound.playsound(path)



def euclidean_dist(ptA, ptB):
    return np.linalg.norm(ptA - ptB)


def eye_aspect_ratio(eye):
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])
    C = euclidean_dist(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)

    return ear


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
                help="path to where the face cascade resides")
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", required=True,
                help="boolean used to indicate if TraffHat should be used")
#쓰레드 적용시 알람
#ap.add_argument('-a', "--alarm", type=str, default="",
                #help="path alarm .WAV file")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48

COUNTER = 0
ALARM_ON = False

# GPIO.setwarnings(False)
#GPIO.setmode(GPIO.BCM)
#GPIO.setup(21, GPIO.OUT)

print("[INFO] loading facial landmark predictor...")
detector = cv2.CascadeClassifier(args["cascade"])
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] starting video stream thread...")
vs = VideoStream(usePiCamera=True).start()
time.sleep(1.0)

while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

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

            cv2.putText(frame, "winkle COUNT : {}".format(COUNTER), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255),
                        2)

            if COUNTER >= EYE_AR_CONSEC_FRAMES:

                if not ALARM_ON:
                    ALARM_ON = True

                    if args["alarm"] > 0:
                        call(["aplay", "/home/pi/Downloads/alarm.wav"])
                    #쓰레드 이용한 알람
                    #if args["alarm"] != "":
                        #t = Thread(target=sound_alarm,
                                 #args=(args["alarm"],))
                        #t.deamon = True
                        #t.start()

                cv2.putText(frame, "WAKE UP! WAKE UP!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

               #LED점멸
               # while(True):
                    #GPIO.output(21, True)
                    #time.sleep(1)
                    #GPIO.output(21, False)
                    #time.sleep(1)


        else:
            COUNTER = 0
            ALARM_ON = False
            #GPIO 초기화
            #GPIO.cleanup()

            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()

