import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm


cap = cv2.VideoCapture(0)
detector = htm.handDetector(maxHands=4)
pTime = 0
cTime = 0


while True:
    success, img = cap.read()
    img = detector.findHands(img)

    lmList = detector.findPosition(img, draw=False)

    if lmList:
        print(lmList[8])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)