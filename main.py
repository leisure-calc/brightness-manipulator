import cv2
import mediapipe as mp
from math import hypot
import numpy as np
import screen_brightness_control as sbc

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0

while True:
    success, img = cap.read()
    if not success:
        break

    # ↓ Resize frame to reduce CPU load
    img = cv2.resize(img, (640, 480))

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    lmList=[]

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape 
                cx, cy = int(lm.x * w), int(lm.y * h) 
                lmList.append([id, cx, cy])
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
        
    if lmList!=[]:
        x1,y1 = lmList[4][1], lmList[4][2]
        x2,y2 = lmList[8][1], lmList[8][2]
        cv2.circle(img,(x1,y1),radius=4,thickness=4,color=(255,0,0))
        cv2.circle(img,(x2,y2),radius=4,thickness=4,color=(255,0,0))
        cv2.line(img,(x1,y1),(x2,y2),color=(0,255,0),thickness=4)

        distance = hypot(x2-x1,y2-y1)
        bright = np.interp(distance,[12,200],[0,100])
        sbc.set_brightness(int(bright))
    



    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()