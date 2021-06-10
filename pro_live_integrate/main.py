from cnn import live
import dlib
import cv2
from face_position import face_pos
import f_detector


dlib_detector = dlib.get_frontal_face_detector()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
detector = f_detector.detect_face_orientation()

cap  = cv2.VideoCapture(0)
start = 0
inactive =0
while True:
    yes, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    list = live(frame)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3,minNeighbors=5,minSize=(64,64),
            flags=cv2.CASCADE_FIND_BIGGEST_OBJECT)

    for (x, y, w, h) in faces:
        dst = round(6421 / w)
    #print(f'dst:{dst}')
    face = dlib_detector(gray,0)
    #print(len(face))
    if len(face)==0:
        start += 1
        if start>15:
            cap.release()
            print("no faces detected, plz try again")
        else:
            start = 0
    else:
        if len(face)>1:
            start += 1
            if start<15:
                cv2.putText(frame, 'multiple faces detected', (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif start>=15:
                cv2.putText(frame, 'multiple faces detected,plz try again', (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        else:
            start = 0

    if dst in range(30,50) and len(face)==1:
        boxes,names = detector.face_orientation(gray)
        pos = face_pos(boxes,names)
        if list[4]=='real':
            cv2.rectangle(frame, (list[0], list[1]), (list[2], list[3]),
                                (255, 0, 0), 2)
            cv2.putText(frame, list[4], (list[0], (list[1] - 25)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Distance from camera:{str(dst)} cm', (list[0], (list[1] - 50)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            cv2.putText(frame,f'Facing :{pos}',(list[0],(list[1]-70)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        elif list[4]=='fake':
            cv2.rectangle(frame, (list[0], list[1]), (list[2], list[3]),
                                  (255, 0, 0), 2)
            cv2.putText(frame, list[4], (list[0], (list[1] - 25)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, f'Distance from camera:{str(dst)} cm', (list[0], (list[1] - 50)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            cv2.putText(frame,f'Facing :{pos}',(list[0],(list[1]-70)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


    else:
        if dst<30:
            cv2.putText(frame, 'WARNING:"Too close, plz move back"', (50,50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        elif dst>50:
            cv2.putText(frame, 'WARNING:"Too far, plz move closer"', (50,50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
