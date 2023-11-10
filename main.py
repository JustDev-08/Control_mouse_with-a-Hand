from cvzone.HandTrackingModule import HandDetector
import cv2
import pyautogui
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
# Initialize the webcam to capture video
# The '2' indicates the third camera connected to your computer; '0' would usually refer to the built-in camera
cap = cv2.VideoCapture(0)

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)
detector_face = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640,360,[20,50])
idList = [22,23,24,26,110,157,158,159,161,130 ,243]
ratioList = []
blink_count = 0
counter = 0

first = True
origin = (800,450)
kum = False
pre_focus_1 = []
pre_focus_2 = []
# Continuously get frames from the webcam
while True:
    # Capture each frame from the webcam
    # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
    success, img = cap.read()
    img = cv2.resize(img,(1600,900))
    # Find hands in the current frame
    # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
    # The 'flipType' parameter flips the image, making it easier for some detections
    img, faces = detector_face.findFaceMesh(img,draw=False)
    hands, img = detector.findHands(img, draw=True, flipType=True)
    cv2.circle(img,origin,5,(0,255,255))
    # Check if any hands are detected
    if hands:
        # Information for the first hand detected
        hand1 = hands[0]  # Get the first hand detected
        lmList1 = hand1["lmList"]  # List of 21 landmarks for the first hand
        bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)
        center1 = hand1['center']  # Center coordinates of the first hand
        handType1 = hand1["type"]  # Type of the first hand ("Left" or "Right")
        focus_1 = lmList1[8][0:2]
        focus_2 = lmList1[6][0:2]
        cv2.circle(img,focus_2,5,(0,255,255))
        length, info, img = detector.findDistance(focus_1,origin, img, color=(255, 0, 255),scale=10)
        
        # pre_focus_1.append(focus_1)
        # pre_focus_2.append(focus_2)
        # if len(pre_focus_1) >= 4 :
        #     pre_focus_1.pop(0)
        # if len(pre_focus_2) >= 4 :
        #     pre_focus_2.pop(0)
        # focus_1_mean = [0,0]
        # focus_2_mean = [0,0]
        # for i in pre_focus_1 :
        #     focus_1_mean[0] += i[0]
        #     focus_1_mean[1] += i[1]
        # focus_1_mean[0] /= 4
        # focus_1_mean[1] /= 4
        # for i in pre_focus_1 :
        #     focus_2_mean[0] += i[0]
        #     focus_2_mean[1] += i[1]
        # focus_2_mean[0] /= 4
        # focus_2_mean[1] /= 4
        
        # print(focus_1_mean)
        if focus_2[1] <= focus_1[1]  :
            kum = True
        else :
            kum = False    
        if kum == False :
            pyautogui.moveTo(focus_1)
    if faces :
        face = faces[0]
        for id in idList:
            cv2.circle(img, face[id],5,(255,0,255),cv2.FILLED)
        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]
        lenghtVer,_ = detector_face.findDistance(leftUp,leftDown)
        lenghtHor,_ = detector_face.findDistance(leftLeft,leftRight)
        cv2.line(img,leftUp,leftDown,(0,200,0),3)
        cv2.line(img,leftLeft,leftRight,(0,200,0),3)
        ratio = (lenghtVer/lenghtHor)*100
        ratioList.append(ratio)
        if len(ratioList) > 2:
            ratioList.pop(0)
        ratioAvg = sum(ratioList)/len(ratioList)
        if ratioAvg < 36 and counter == 0:
            blink_count += 1
            counter = 1
            pyautogui.click()
        if counter != 0 :
            counter += 1
            if counter > 5:
                counter = 0
        cv2.putText(img,f'count : {blink_count}',org=(50, 50), 
                fontFace = cv2.FONT_HERSHEY_DUPLEX,
                fontScale = 1,
                color = (125, 246, 55),
                thickness = 2
        )
    img = cv2.resize(img,(533,300))
    cv2.imshow("Image", img)

    # Keep the window open and update it for each frame; wait for 1 millisecond between frames
    cv2.waitKey(1)