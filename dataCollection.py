import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

folder = "Data/C"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRadio = h / w

        if aspectRadio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgRize = cv2.resize(imgCrop, (wCal, imgSize))
            imgRizeShape = imgRize.shape
            wGap = math.ceil((imgSize - wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgRize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgRize = cv2.resize(imgCrop, (imgSize, hCal))
            imgRizeShape = imgRize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgRize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImgWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
