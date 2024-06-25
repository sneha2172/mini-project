import cv2
import numpy as np
import tensorflow as tf
import pygame
import dlib
from scipy.spatial import distance
from imutils import face_utils
import imutils
from pygame import mixer    q

model = tf.keras.models.load_model('C:/RLDD/model_weights.h5')

cap = cv2.VideoCapture(0)

eye_cascade = cv2.CascadeClassifier('C:/RLDD/haarcascade_eye.xml')

alarm_on = False
closed_eye_frames = 0
closed_eyes_limit = 25

mixer.init()
mixer.music.load("C:/RLDD/music.wav")

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh = 0.15
frame_check = 30
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("C:/RLDD/shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    subjects = detect(gray, 0)

    if len(eyes) == 0 and alarm_on:
        mixer.music.stop()
        alarm_on = False

    for (x, y, w, h) in eyes:
        eye_img = gray[y:y+h, x:x+w]
        eye_img = cv2.resize(eye_img, (224, 224))
        eye_img_rgb = cv2.cvtColor(eye_img, cv2.COLOR_GRAY2RGB)
        eye_img_rgb = np.expand_dims(eye_img_rgb, axis=0)
        prediction = model.predict(eye_img_rgb)

        if np.any(prediction < 0.4):
            closed_eye_frames += 1
        else:
            closed_eye_frames = 0

        if closed_eye_frames >= closed_eyes_limit:
            if not alarm_on:
                mixer.music.play()
                alarm_on = True

                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            if alarm_on:
                mixer.music.stop()
                alarm_on = False

    for subject in subjects:
        shape = predict(gray, subject)
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
        if ear < thresh:
            if not alarm_on:
                mixer.music.play()
                alarm_on = True

                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            if alarm_on:
                mixer.music.stop()
                alarm_on = False

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
mixer.quit()
