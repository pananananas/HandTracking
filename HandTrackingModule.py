"""
Hand tracking module
Developed 26.01.2023 by @pananananas
"""

import cv2
import mediapipe as mp
import time
import math


class handDetector:
    def __init__(self, mode=False, maxHands=2, model_complexity=1, detectionCon=0.5, trackCon=0.5):

        self.lmList = None
        self.results = None

        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = model_complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)   # prints None if no hands are detected or a list of landmarks

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=False):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 255, 255), cv2.FILLED)
            return self.lmList

    def handDetected(self):
        if self.results.multi_hand_landmarks:
            return True
        else:
            return False

    def pointsAreClose(self, finger=5):
        if finger == 5:
            first, second = 20, 17
        elif finger == 4:
            first, second = 16, 13
        elif finger == 3:
            first, second = 12, 9
        else:
            return False

        x1, y1 = self.lmList[first][1], self.lmList[first][2]
        x2, y2 = self.lmList[second][1], self.lmList[second][2]
        length = math.hypot(x2 - x1, y2 - y1)

        if length < 50:
            return True
        else:
            return False

    def drawLineBetweenPoints(self, img, finger=5, color=(0, 0, 255), thickness=3):
        if finger == 5:
            first, second = 20, 17
        elif finger == 4:
            first, second = 16, 13
        elif finger == 3:
            first, second = 12, 9
        else:
            return False

        x1, y1 = self.lmList[first][1], self.lmList[first][2]
        x2, y2 = self.lmList[second][1], self.lmList[second][2]
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        return True

def main():
    cap = cv2.VideoCapture(0)  # webcam 0, the main one :>
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        # FPS counter
        cTime = time.time()
        pTime = cTime
        fps = 1 / (cTime - pTime)
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
