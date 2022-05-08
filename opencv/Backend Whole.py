import time
import mediapipe as mp
import matplotlib.pyplot as plt
from numpy.lib.function_base import median
from scipy.interpolate import UnivariateSpline

import math
import cv2 as cv
import numpy as np
import squats as pm

import tkinter as tk
from tkinter import ttk
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
form = 0
feedback = "Fix Form"
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

left_lip_corner = 61
right_lip_corner = 91
chin_anchor = 152

left_eyebrow = 105
left_eye = 145
right_eyebrow = 334
right_eye = 374

nose_top = 6
nose_bottom = 4

# choose workout
def click(x):
    global flag
    win.destroy()
    flag = x


win = tk.Tk()
win.title("AI Trainer")
Lbl = ttk.Label(win, text="Choose your workout")
Lbl.pack()
flag = ""

action1 = ttk.Button(win, text="Bicep Curl", command=lambda: click("Bicep Curl"))
action2 = ttk.Button(win, text="Squat", command=lambda: click("Squat"))
action3 = ttk.Button(win, text="Pushup", command=lambda: click("Pushup"))
action4 = ttk.Button(win, text="Stroke", command=lambda: click("Stroke"))
action5 = ttk.Button(win, text="Bells Palsy", command=lambda: click("Bells Palsy"))
action1.pack()
action2.pack()
action3.pack()
action4.pack()
action5.pack()

win.mainloop()

def detect_bicep_curls ():
    cap = cv.VideoCapture(0)
    fps = cap.get(cv.CAP_PROP_FPS)
    delay = round(1000 / fps)

    detector = pm.PoseDetector()
    count = 0
    direction = 0
    sec = 5

    while cap.isOpened(): 

        #countdown before it starts
        prev = time.time()
        while sec > 0:
            ret, img = cap.read()
            cv.putText(img, str(sec), (img.shape[1]//2, img.shape[0]//2), cv.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)
            cv.imshow("image", img)
            cur = time.time()
            if cur - prev >= 1:
                prev = cur
                sec -= 1
            cv.waitKey(1)

        # detect workout
        ret, img = cap.read()

        if not ret:
            break

        img = detector.get_pose(img, False)
        lmList = detector.get_position(img,False)

        if len(lmList) != 0:
            global per, bar
            # left arm
            angle = detector.get_angle(img, 11, 13, 15, draw=True)
            # right arm
            detector.get_angle(img, 12, 14, 16, draw=True)
            per = np.interp(angle, (50, 160), (100, 0))
            bar = np.interp(angle, (50, 160), (300, 600))

            # counting reps
            color = (255,255,0)
            if per == 100:
                color = (0, 0, 255)
                if direction == 0:
                    count += 0.5
                    direction = 1
            if per == 0:
                color = (0, 0, 255)
                if direction == 1:
                    count += 0.5
                    direction = 0
            cv.putText(img, str(int(count)), (50,100), cv.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 5)

            cv.rectangle(img, (50, 300), (150,600),(255,255,255),cv.FILLED)
            cv.rectangle(img, (50, int(bar)), (150,600),color,cv.FILLED)
            cv.putText(img, f'{int(per)}%', (50,280), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv.imshow("image", img)

        if cv.waitKey(1) == 27:
            break
def detect_squats():
    cap = cv.VideoCapture(0)
    fps = cap.get(cv.CAP_PROP_FPS)
    delay = round(1000 / fps)

    detector = pm.PoseDetector()
    count = 0
    direction = 0
    sec = 5

    while cap.isOpened(): 

        #countdown before it starts
        prev = time.time()
        while sec > 0:
            ret, img = cap.read()
            cv.putText(img, str(sec), (img.shape[1]//2, img.shape[0]//2), cv.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)
            cv.imshow("image", img)
            cur = time.time()
            if cur - prev >= 1:
                prev = cur
                sec -= 1
            cv.waitKey(1)

        # detect workout
        ret, img = cap.read()

        if not ret:
            break

        img = detector.get_pose(img, False)
        lmList = detector.get_position(img,False)

        if len(lmList) != 0:
            global per, bar
            # left side
            angle = detector.get_angle(img, 11, 23, 25, draw=True)
            per = np.interp(angle, (65, 165), (100, 0))
            bar = np.interp(angle, (65, 165), (300, 600))
            # counting reps
            color = (255,255,0)
            if per == 100:
                color = (0, 0, 255)
                if direction == 0:
                    count += 0.5
                    direction = 1
            if per == 0:
                color = (0, 0, 255)
                if direction == 1:
                    count += 0.5
                    direction = 0
            cv.putText(img, str(int(count)), (50,100), cv.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 5)

            cv.rectangle(img, (50, 300), (150,600),(255,255,255),cv.FILLED)
            cv.rectangle(img, (50, int(bar)), (150,600),color,cv.FILLED)
            cv.putText(img, f'{int(per)}%', (50,280), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv.imshow("image", img)

        if cv.waitKey(1) == 27:
            break
def detect_Pushup():
    cap = cv.VideoCapture(0)
    detector = pm.PoseDetector()
    count = 0
    direction = 0
    form = 0
    feedback = "Fix Form"


    while cap.isOpened():
        ret, img = cap.read() #640 x 480
        #Determine dimensions of video - Help with creation of box in Line 43
        width  = cap.get(3)  # float `width`
        height = cap.get(4)  # float `height`
        # print(width, height)
        
        img = detector.get_pose(img, False)
        lmList = detector.get_position(img, False)
        # print(lmList)
        if len(lmList) != 0:
            elbow = detector.get_angle(img, 11, 13, 15)
            shoulder = detector.get_angle(img, 13, 11, 23)
            hip = detector.get_angle(img, 11, 23,25)
            
            #Percentage of success of pushup
            per = np.interp(elbow, (90, 160), (0, 100))
            
            #Bar to show Pushup progress
            bar = np.interp(elbow, (90, 160), (380, 50))

            #Check to ensure right form before starting the program
            if elbow > 160 and shoulder > 40 and hip > 160:
                form = 1
        
            #Check for full range of motion for the pushup
            if form == 1:
                if per == 0:
                    if elbow <= 90 and hip > 160:
                        feedback = "Up"
                        if direction == 0:
                            count += 0.5
                            direction = 1
                    else:
                        feedback = "Fix Form"
                        
                if per == 100:
                    if elbow > 160 and shoulder > 40 and hip > 160:
                        feedback = "Down"
                        if direction == 1:
                            count += 0.5
                            direction = 0
                    else:
                        feedback = "Fix Form"
                            # form = 0
                    
                        
        
            print(count)
            
            #Draw Bar
            if form == 1:
                cv.rectangle(img, (580, 50), (600, 380), (0, 255, 0), 3)
                cv.rectangle(img, (580, int(bar)), (600, 380), (0, 255, 0), cv.FILLED)
                cv.putText(img, f'{int(per)}%', (565, 430), cv.FONT_HERSHEY_PLAIN, 2,
                            (255, 0, 0), 2)


            #Pushup counter
            cv.rectangle(img, (0, 380), (100, 480), (0, 255, 0), cv.FILLED)
            cv.putText(img, str(int(count)), (25, 455), cv.FONT_HERSHEY_PLAIN, 5,
                        (255, 0, 0), 5)
            
            #Feedback 
            cv.rectangle(img, (500, 0), (640, 40), (255, 255, 255), cv.FILLED)
            cv.putText(img, feedback, (500, 40 ), cv.FONT_HERSHEY_PLAIN, 2,
                        (0, 255, 0), 2)

            
        cv.imshow('Pushup counter', img)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv.destroyAllWindows()
def detect_Stroke():
    tipIds = [4, 8, 12, 16, 20]
    state = None
    Gesture = None
    wCam, hCam = 720, 640
    instr_array = ["Get ready!", "9 more times to go", "8 more times to go", "8 more times to go", "7 more times to go", "6 more times to go", "5 more times to go", "4 more times to go", "3 more times to go", "2 more times to go", "1 more times to go", "Smile :)"]
    instr_index = 0

    def fingerPosition(image, handNo=0):
        lmList = []
        if results.multi_hand_landmarks:
            myHand = results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
    
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy]) 
        return lmList

    cap = cv.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    i = 0

    with mp_hands.Hands(
        min_detection_confidence=0.8,
        min_tracking_confidence=0.5) as hands:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")

            continue

        image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        

        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
         for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        lmList = fingerPosition(image)
        imageText = image.copy()

        if len(lmList) != 0:
            


            
            imageText = cv.putText(img=np.copy(image), text=instr_array[instr_index], org=(50,50),fontFace=3, fontScale=1, color=(255,255,255), thickness=3)
                    
            fingers = []
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
            
                    fingers.append(1)
                if (lmList[tipIds[id]][2] > lmList[tipIds[id] - 2][2] ):
            
                    fingers.append(0)
            totalFingers = fingers.count(1)
            print(totalFingers)

            
            if totalFingers == 4:
                state = "open"
            # fingers.append(1)
            if totalFingers == 0 and state == "open":
                state = "close"
                instr_index+=1    
        cv.imshow("Media Controller", imageText )
        key = cv.waitKey(1) & 0xFF

        if key == ord("q"):
            break
      cv.destroyAllWindows()
def detect_bells_palsy():
    instr_array = ["Get ready!", "Wrinkle your nose", "Puff your cheeks and blow", "Smile :)"]

    wrinkled = False
    puff = False
    kiss = False
    smile = False

    def getCoord(image, normalx, normaly):
        image_rows, image_cols, _ = image.shape
        x_px = min(math.floor(normalx * image_cols), image_cols - 1)
        y_px = min(math.floor(normaly * image_rows), image_rows - 1)
        return x_px, y_px

    def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
            """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.

            `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
            """
            # Image ranges
            y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
            x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

            # Overlay ranges
            y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
            x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

            # Exit if nothing to do
            if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
                return

            # Blend overlay within the determined ranges
            img_crop = img[y1:y2, x1:x2]
            img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
            alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
            alpha_inv = 1.0 - alpha

            img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop


        # For webcam input:
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv.VideoCapture(0)
    overlay = cv.imread('C:/Users/annie/Downloads/PikPng.com_face-png-transparent_2717214.png')

    timestamp = time.time()
    timeelapsed = timestamp
    instr_index = 0

    mouth_width = 0
    new_mouth_width = 0

    nose_length = 0
    new_nose_length = 0

    colr = (255, 255, 255)

    with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
     while cap.isOpened():

        timeelapsed = time.time()
        
        seconds = 10 - abs(timestamp - timeelapsed)
        
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
        cimage = image
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_mesh.process(image)
        
        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
          for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
                    
            chin_anchor_x_val, chin_anchor_y_val = getCoord(image, face_landmarks.landmark[chin_anchor].x, face_landmarks.landmark[chin_anchor].y)
            left_lip_corner_x_val, left_lip_corner_y_val = getCoord(image, face_landmarks.landmark[left_lip_corner].x, face_landmarks.landmark[left_lip_corner].y)
            right_lip_corner_x_val, right_lip_corner_y_val = getCoord(image, face_landmarks.landmark[right_lip_corner].x, face_landmarks.landmark[right_lip_corner].y)
            left_eyebrow_x_val, left_eyebrow_y_val = getCoord(image, face_landmarks.landmark[left_eyebrow].x, face_landmarks.landmark[left_eyebrow].y)
            right_eyebrow_x_val, right_eyebrow_y_val = getCoord(image, face_landmarks.landmark[right_eyebrow].x, face_landmarks.landmark[right_eyebrow].y)
            left_eye_x_val, left_eye_y_val = getCoord(image, face_landmarks.landmark[left_eye].x, face_landmarks.landmark[left_eye].y)
            right_eye_x_val, right_eye_y_val = getCoord(image, face_landmarks.landmark[right_eye].x, face_landmarks.landmark[right_eye].y)
            
            nose_top_x_val, nose_top_y_val = getCoord(image, face_landmarks.landmark[nose_top].x, face_landmarks.landmark[nose_top].y)
            nose_bottom_x_val, nose_bottom_y_val = getCoord(image, face_landmarks.landmark[nose_bottom].x, face_landmarks.landmark[nose_bottom].y)

            left_lip = math.sqrt((chin_anchor_x_val-left_lip_corner_x_val)**2 + (chin_anchor_y_val-left_lip_corner_y_val)**2)
            right_lip = math.sqrt((chin_anchor_x_val-right_lip_corner_x_val)**2 + (chin_anchor_y_val-right_lip_corner_y_val)**2)
            
            left_eye_dist = math.sqrt((left_eyebrow_x_val-left_eye_x_val)**2 + (left_eyebrow_y_val-left_eye_y_val)**2)
            right_eye_dist = math.sqrt((right_eyebrow_x_val-right_eye_x_val)**2 + (right_eyebrow_y_val-right_eye_y_val)**2)

            eye_diff = abs(left_eye_dist - right_eye_dist)/min(left_eye_dist, right_eye_dist)
            
            lip_diff = abs(left_lip - right_lip)/min(left_lip, right_lip)
            
            nose_diff = math.sqrt((nose_top_x_val-nose_bottom_x_val)**2 + (nose_top_y_val-nose_bottom_y_val)**2)
            
            texted_image = image
            
            image = cv.circle(image, (20,40), 10, color=colr, thickness=5)
            texted_image = cv.putText(img=np.copy(image), text=str(int(seconds)), org=(50,50),fontFace=3, fontScale=1, color=(255,255,255), thickness=3)
            texted_image = cv.putText(img=np.copy(texted_image), text=instr_array[instr_index], org=(100,50),fontFace=3, fontScale=1, color=(255,255,255), thickness=3)
            
            if (seconds < 0):
                instr_index+=1
                seconds = 10
                timestamp=time.time()
                timeelapsed = timestamp
                
            new_mouth_width = abs(left_lip_corner_x_val - right_lip_corner_x_val)
            new_nose_length = nose_diff
            
            if (instr_index == 0):
                # winsound.PlaySound("SystemExit", winsound.SND_ALIAS)
                colr = (255,255,255)
                mouth_width = abs(left_lip_corner_x_val - right_lip_corner_x_val)
                nose_length = nose_diff
                
            elif (instr_index == 1 and (nose_length * 1.1) < new_nose_length):
                # alpha = 0.5
                # texted_image = np.uint8(texted_image*alpha + overlay*(1-alpha))
                colr = (0,255,0)
                print("bunny ears")
                
            elif (instr_index == 2 and (mouth_width * 0.9) > new_mouth_width):
                colr = (0,255,0)
                print("firebreath")
                
            elif (instr_index == 3 and (mouth_width * 1.1) < new_mouth_width):
                colr = (0,255,0)
                print("rainbow")
                
            else:
                colr = (0,255,255)
            
        cv.imshow('MediaPipe FaceMesh', texted_image)
        if cv.waitKey(5) & 0xFF == 27:
         break
    cap.release()

if (flag == "Bicep Curl"):
    detect_bicep_curls()
elif flag == "Squat":
    detect_squats()
elif flag == "Pushup":
    detect_Pushup()
elif flag == "Stroke":
    detect_Stroke()
elif flag == "Bells Palsy":
    detect_bells_palsy()