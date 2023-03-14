import cv2
import mediapipe as mp
import numpy as np
import time
import math
import HandTrackingModule as htm
import osascript as osa


# Parameters
wCam, hCam = 1280, 720
minVol, maxVol = 0, 100
minLen, maxLen = 30, 300

# Camera
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Detector
detector = htm.handDetector(mode=False, maxHands=1, model_complexity=1, detectionCon=0.8, trackCon=0.8)

# FPS counter
pTime = 0
cTime = 0
setvol = 0
vol = 0

# Distance const
handPointsDist = [260, 228, 193, 144, 122, 103, 75, 68, 60, 58]
distFromCamera = [20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
coff = np.polyfit(handPointsDist, distFromCamera, 2)

while True:
    success, img = cap.read()
    img = detector.findHands(img)

    if detector.handDetected():

        lmList = detector.findPosition(img)
        # Distance
        xd1, yd1 = lmList[5][1], lmList[5][2]
        xd2, yd2 = lmList[17][1], lmList[17][2]
        handDist = math.hypot(xd2 - xd1, yd2 - yd1)

        a, b, c = coff
        distance = a*handDist**2 + b*handDist + c

        print(distance, handDist)


        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        radius = int(length / 2)

        # Drawing lines
        if length < minLen:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        elif length > maxLen:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        else:
            cv2.line(img, (x1, y1), (x2, y2), (111, 150, 50), 3)

        # Drawing circle
        cv2.circle(img, (cx, cy), radius, (111, 150, 50), 3)

        vol = int(np.interp(length, [minLen, maxLen], [minVol, maxVol]))

        if detector.pointsAreClose() and 20 < distance < 80:
            detector.drawLineBetweenPoints(img, color=(0, 0, 255), thickness=3)
            # Volume
            setvol = vol
            stringVol = "set volume output volume " + str(setvol)
            # osa.osascript(stringVol)


    # Volume bar
    slider = int(np.interp(vol, [minVol, maxVol], [350, 50]))

    cv2.rectangle(img, (50, 50), (85, 350), (111, 150, 50), 3)
    cv2.recxtangle(img, (50, slider), (85, 350), (111, 150, 50), cv2.FILLED)
    cv2.putText(img, f'{vol}', (50, 450), cv2.FONT_HERSHEY_PLAIN, 3, (111, 150, 50), 3)
    cv2.putText(img, f'{setvol}', (50, 500), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)



    # FPS counter:
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps))+' FPS', (10, hCam-10), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)