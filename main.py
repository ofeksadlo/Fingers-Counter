import cv2
import numpy as np
import math
from pyautogui import press

cap = cv2.VideoCapture(0)

whT = 320
confThreshold = 0.5
nmsThreshold = 0.3
classesNames = ['fist']

modelConfiguration = 'fist_model.cfg'
modelWeights = 'fist_model.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

handROI = None

def findROI(outputs, img):
    global handROI
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - (w / 2)), int((det[1] * hT) - (h / 2))
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        handROI = (x, y, w, h)
        cv2.rectangle(img, (x-40,y-40), (x+w+30, y+h+30),(0,0,255),thickness=1)
        cv2.putText(img, f'{classesNames[0]} {int(confs[i] * 100)}%', (x, y -10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), thickness=1)

foundROI = False

isPaused = False

while True:
    ret, frame = cap.read()

    blob = cv2.dnn.blobFromImage(frame, 1/255, (whT, whT), [0,0,0], 1, crop=False)

    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)
    
    
    count_finger_gaps = 0
    if handROI is not None:
        x ,y ,w ,h = handROI
        
        cv2.rectangle(frame, (x-40,y-40), (x+w+30, y+h+30),(0,0,255),thickness=5)

        # frameROI = frame.copy()
        frameROI = frame[y-35:y+h+25, x-35:x+w+25]

        gray = cv2.cvtColor(frameROI, cv2.COLOR_BGR2GRAY)
        
        blur = cv2.GaussianBlur(gray, (45,45), sigmaX=0)

        ret, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cnt = max(contours, key=lambda x: cv2.contourArea(x))

        hull = cv2.convexHull(cnt, returnPoints=False)

        defects = cv2.convexityDefects(cnt, hull)

        
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

            if angle <= 110:
                count_finger_gaps += 1
                cv2.circle(frameROI, far, 5, (255,0,0), -1)
        if count_finger_gaps == 0:
            cv2.putText(frame, 'Fingers found: ' + str(count_finger_gaps), (x-40,y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), thickness=2)
        else:
            cv2.putText(frame, 'Fingers found: ' + str(count_finger_gaps + 1), (x-40,y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), thickness=2)
    else:
        findROI(outputs, frame)

    cv2.imshow('Frame', frame)

    cv2.waitKey(1)